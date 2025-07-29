"""
Multi-provider Chat-LLM client

Supported providers
-------------------
* OpenAI
* DeepSeek
* Sonnet
* Ollama (local)
* Groq (OpenAI-style)
* Gemini 2 ( Google Generative Language API – flash/pro models )

Usage
-----
client = Client("gemini:gemini-2.0-flash")
resp    = client.chat.completions.create(
             messages=[{"role": "user", "content": "Explain how AI works"}],
             temperature=0.3
         )
print(resp)   # → assistant text, OpenAI-style
"""

import os
import base64
import requests
from dotenv import load_dotenv
from vidavox.utils.script_tracker import log_processing_time

# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #

load_dotenv()

PROVIDER_ENDPOINTS = {
    "openai":   "https://api.openai.com/v1/chat/completions",
    "deepseek": "https://api.deepseek.ai/v1/chat/completions",
    "sonnet":   "https://api.sonnet.ai/v1/chat/completions",
    "ollama":   "http://localhost:11434/api/generate",
    "groq":     "https://api.groq.com/openai/v1/chat/completions",
    # Gemini needs the model name appended later:  <base>/<model>:generateContent
    "gemini":   "https://generativelanguage.googleapis.com/v1beta/models/",
}

API_KEY_ENV_VARS = {
    "openai":   "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "sonnet":   "SONNET_API_KEY",
    "gemini":   "GEMINI_API_KEY",       # ⬅️ added
}

# --------------------------------------------------------------------------- #
#  Public façade                                                              #
# --------------------------------------------------------------------------- #


class Client:
    """
    Example:  Client("openai:gpt-4o-mini")   or   Client("gemini:gemini-2.0-pro")
    """

    def __init__(self, model: str, api_key: str | None = None):
        try:
            provider, model_name = model.split(":", 1)
        except ValueError:
            raise ValueError("Model must be '<provider>:<model_name>'")

        self.provider = provider.lower()
        self.default_model = model_name.strip()

        # Ollama ≠ API key
        if self.provider == "ollama":
            self.api_key = None
        else:
            self.api_key = api_key or self._get_api_key_from_env()
            if not self.api_key:
                raise ValueError(
                    f"No API key provided for {self.provider} and none found in environment variables"
                )

        self.chat = Chat(self.provider, self.api_key, self.default_model)

    # --------------------------------------------------------------------- #
    #  Internals                                                            #
    # --------------------------------------------------------------------- #

    def _get_api_key_from_env(self) -> str | None:
        if self.provider == "ollama":
            return None
        env_var = API_KEY_ENV_VARS.get(self.provider)
        return os.getenv(env_var) or os.getenv("API_KEY")


class Chat:
    def __init__(self, provider, api_key, default_model=None):
        self.completions = Completions(provider, api_key, default_model)


class Completions:
    def __init__(self, provider, api_key, default_model):
        self.provider = provider
        self.api_key = api_key
        self.default_model = default_model

    # --------------------------------------------------------------------- #
    #  Unified create()                                                     #
    # --------------------------------------------------------------------- #

    @log_processing_time
    def create(self, messages, temperature=0.7, model=None, **kwargs):
        model_to_use = model or self.default_model
        if not model_to_use:
            raise ValueError("No model specified")

        # ---------- Provider-specific request build ---------------------- #
        if self.provider == "ollama":
            url, headers, payload = self._build_ollama(messages, model_to_use,
                                                       temperature, **kwargs)

        elif self.provider == "gemini":
            url, headers, payload = self._build_gemini(messages, model_to_use,
                                                       temperature, **kwargs)

        else:  # OpenAI-style providers (OpenAI, DeepSeek, Sonnet, Groq)
            url = PROVIDER_ENDPOINTS[self.provider]
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": model_to_use,
                "messages": messages,
                "temperature": temperature,
                **kwargs
            }

        # ---------- Dispatch -------------------------------------------- #
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        # ---------- Provider-specific response adaptation ---------------- #
        if self.provider == "ollama":
            data = self._adapt_ollama_response(data)
        elif self.provider == "gemini":
            data = self._adapt_gemini_response(data)

        return Response(data)

    # --------------------------------------------------------------------- #
    #  Provider helpers                                                     #
    # --------------------------------------------------------------------- #

    # -------------- Ollama (text + VLM) ----------------------------------- #
    def _build_ollama(self, messages, model_to_use, temperature, **kwargs):
        # Check VLM?
        has_images = any(
            isinstance(m.get("content"), list)
            for m in messages
        )

        if has_images:
            proc_msgs = self._process_ollama_vlm_messages(messages)
            payload = {
                "model": model_to_use,
                "messages": proc_msgs,
                "stream": False,
                "options": {"temperature": temperature, **kwargs},
            }
        else:
            prompt = "\n".join(m["content"]
                               for m in messages if m["role"] == "user")
            payload = {
                "model": model_to_use,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, **kwargs},
            }

        headers = {"Content-Type": "application/json"}
        url = PROVIDER_ENDPOINTS["ollama"]
        return url, headers, payload

    def _adapt_ollama_response(self, data: dict) -> dict:
        if "response" in data:
            text = data.pop("response")
            data["choices"] = [{
                "index": 0,
                "finish_reason": data.get("done_reason"),
                "message": {"role": "assistant", "content": text}
            }]
        return data

    # -------------- Gemini ------------------------------------------------ #
    def _build_gemini(self, messages, model_to_use, temperature, **kwargs):
        """
        Convert an OpenAI-style chat history (system / user / assistant)
        into Gemini’s required format (role = "user" | "model").

        Mapping rules
        -------------
        • "user"      →  "user"
        • "assistant" →  "model"
        • "system"    →  prepend to the *first* user message
                         (Gemini has no dedicated system role)
        """
        # Endpoint
        url = f"{PROVIDER_ENDPOINTS['gemini']}{model_to_use}:generateContent"

        contents = []
        system_buffer = ""

        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")

            # 1️⃣  System messages → keep in a buffer, prepend to next user msg
            if role == "system":
                system_buffer += text + "\n"
                continue

            # 2️⃣  Map roles for Gemini
            if role == "assistant":
                gemini_role = "model"
            else:  # "user" or anything else
                gemini_role = "user"

            # 3️⃣  If there is buffered system text and we’re at the first user msg,
            #     prepend it once, then clear the buffer.
            if system_buffer and gemini_role == "user":
                text = system_buffer.strip() + "\n" + text
                system_buffer = ""

            # 4️⃣  Assemble Gemini part
            contents.append(
                {
                    "role": gemini_role,
                    "parts": [{"text": text}],
                }
            )

        # If all messages were “system” (rare), turn them into one user turn
        if not contents and system_buffer:
            contents.append(
                {
                    "role": "user",
                    "parts": [{"text": system_buffer.strip()}],
                }
            )

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                **kwargs,
            },
        }

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key,
        }
        return url, headers, payload

    def _adapt_gemini_response(self, data: dict) -> dict:
        """
        Convert Gemini's {"candidates":[{"content":{"parts":[{"text":...}]}}]}
        to OpenAI-style {"choices":[{"message":{"content":...}}]}
        """
        if "candidates" not in data:
            return data

        choices = []
        for idx, cand in enumerate(data["candidates"]):
            # Get first part text (if empty, fallback to whole string)
            try:
                text = cand["content"]["parts"][0]["text"]
            except Exception:
                text = cand.get("output", "")

            choices.append({
                "index": idx,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            })

        data["choices"] = choices
        return data

    # -------------- Shared VLM helper ------------------------------------- #
    def _process_ollama_vlm_messages(self, messages):
        processed = []
        for msg in messages:
            role, content = msg["role"], msg["content"]

            if isinstance(content, str):
                processed.append({"role": role, "content": content})
                continue

            # content is list of {type,text|image}
            buf = ""
            for part in content:
                if part.get("type") == "text":
                    buf += part["text"]
                elif part.get("type") == "image":
                    img_data = part.get("image_url", {}).get("url", "")
                    buf += f"\n![](data:image/jpeg;base64,{self._to_base64(img_data)})\n"
            processed.append({"role": role, "content": buf})
        return processed

    @staticmethod
    def _to_base64(ref: str):
        if ref.startswith("data:image"):
            return ref.split(",", 1)[1]
        if ref.startswith("http"):
            resp = requests.get(ref)
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode()
        if os.path.exists(ref):
            with open(ref, "rb") as f:
                return base64.b64encode(f.read()).decode()
        # assume already b64
        return ref


# --------------------------------------------------------------------------- #
#  Response abstraction                                                       #
# --------------------------------------------------------------------------- #


class Response:
    def __init__(self, data):
        self.raw_response = data
        self.choices = [Choice(c) for c in data.get("choices", [])]
        self.id = data.get("id")
        self.created = data.get("created")
        self.model = data.get("model")
        self.usage = data.get("usage", {})

    def __str__(self):
        if self.choices:
            return self.choices[0].message.content
        return str(self.raw_response)


class Choice:
    def __init__(self, data):
        self.index = data.get("index")
        self.finish_reason = data.get("finish_reason")
        self.message = Message(data.get("message", {}))


class Message:
    def __init__(self, data):
        self.role = data.get("role")
        self.content = data.get("content")
        # copy any provider-specific keys
        for k, v in data.items():
            if not hasattr(self, k):
                setattr(self, k, v)
