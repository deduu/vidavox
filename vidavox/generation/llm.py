import os
import requests
import base64
from dotenv import load_dotenv
from vidavox.utils.script_tracker import log_processing_time

# Load environment variables from .env file
load_dotenv()

# Mapping of provider names to their chat completion endpoints
PROVIDER_ENDPOINTS = {
    "openai":   "https://api.openai.com/v1/chat/completions",
    "deepseek": "https://api.deepseek.ai/v1/chat/completions",
    "sonnet":   "https://api.sonnet.ai/v1/chat/completions",
    "ollama":   "http://localhost:11434/api/generate",
}

# Mapping of provider names to their environment variable names
API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "sonnet": "SONNET_API_KEY"
}

class Client:
    def __init__(self, model, api_key=None):
        """
        Initialize the Client with a combined model string in the format "provider:model_name"
        e.g., "openai:gpt-4.0"
        
        :param model: A string in the format "provider:model_name"
        :param api_key: Optional API key for the specified provider. If not provided,
                       will attempt to load from environment variables.
        """
        try:
            provider, model_name = model.split(":", 1)
        except ValueError:
            raise ValueError("Model parameter must be in the format 'provider:model_name'")
        
        self.provider = provider.lower()
        self.default_model = model_name.strip()
        
        # If API key is not provided, try to get it from environment variables
        # Skip API key validation for Ollama
        if self.provider == "ollama":
            self.api_key = None
        else:
            if api_key is None:
                api_key = self._get_api_key_from_env()
            self.api_key = api_key
            if not self.api_key:
                raise ValueError(f"No API key provided for {self.provider} and none found in environment variables")
    
        
        # Initialize Chat with the provider, API key, and default model
        self.chat = Chat(self.provider, self.api_key, self.default_model)

    def _get_api_key_from_env(self):
        """
        Attempt to get the API key from environment variables.
        First checks for a provider-specific environment variable,
        then falls back to a generic API_KEY environment variable.
        """
        if self.provider == "ollama":
            return None  # Ollama does not require an API ke
        # Try provider-specific environment variable
        env_var_name = API_KEY_ENV_VARS.get(self.provider)
        if env_var_name:
            api_key = os.getenv(env_var_name)
            if api_key:
                return api_key
        
        # Try generic API_KEY environment variable
        generic_key = os.getenv('API_KEY')
        if generic_key:
            return generic_key
            
        return None


class Chat:
    def __init__(self, provider, api_key, default_model=None):
        self.provider = provider
        self.api_key = api_key
        self.default_model = default_model
        self.completions = Completions(self.provider, self.api_key, self.default_model)


class Completions:
    def __init__(self, provider, api_key, default_model=None):
        self.provider = provider
        self.api_key = api_key
        self.default_model = default_model
        
    @log_processing_time
    def create(self, messages, temperature=0.7, model=None, **kwargs):
        """
        Create a chat completion using the specified or default model.
        
        :param messages: A list of messages, e.g., [{"role": "user", "content": "Hello"}]
        :param temperature: Creativity parameter.
        :param model: Optional model override; if not provided, the default model is used.
        :param kwargs: Additional parameters like top_k, top_p, repeat_penalty, etc.
        :return: The JSON response from the provider's API.
        """
        # Use the provided model if given; otherwise, fall back to the default
        model_to_use = model if model is not None else self.default_model
        if model_to_use is None:
            raise ValueError("No model specified. Please provide a model in Client or override here.")

        # Lookup the API endpoint based on the provider
        url = PROVIDER_ENDPOINTS.get(self.provider)
        if not url:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Build the payload and headers
        # Handle Ollama's API payload structure
        if self.provider == "ollama":
            # Check if any message contains images
            has_images = False
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for content_item in msg["content"]:
                        if isinstance(content_item, dict) and content_item.get("type") == "image":
                            has_images = True
                            break
                    if has_images:
                        break
            
            if has_images:
                # VLM case: Process messages with images for Ollama
                processed_messages = self._process_ollama_vlm_messages(messages)
                
                payload = {
                    "model": model_to_use,
                    "messages": processed_messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        **kwargs
                    }
                }
            else:
                # Standard text case
                prompt = "\n".join([msg["content"] for msg in messages if msg["role"] == "user"])
                
                payload = {
                    "model": model_to_use,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        **kwargs
                    }
                }
            
            headers = {"Content-Type": "application/json"}
        else:
            payload = {
                "model": model_to_use,
                "messages": messages,
                "temperature": temperature,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        
        # Make the POST request
        response = requests.post(url, headers=headers, json=payload)

        try:
            response.raise_for_status()
            response_data = response.json()

            # If it's Ollama and we have a top-level "response" string, convert it
            if self.provider == "ollama" and "response" in response_data:
                # Wrap the "response" text in an OpenAI-style "choices" array
                ollama_text = response_data["response"]
                
                # Optionally handle finish reason, if present
                finish_reason = response_data["done_reason"] if "done_reason" in response_data else None
                
                response_data["choices"] = [{
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {
                        # Emulate an assistant role
                        "role": "assistant",
                        # Put the actual LLM text here
                        "content": ollama_text
                    }
                }]
                
                # (Optional) remove the original "response" to avoid confusion
                del response_data["response"]

            return Response(response_data)
        except requests.exceptions.HTTPError as e:
            print(f"ERROR: HTTP error occurred: {e}")
            if response.text:
                print(f"Response text: {response.text}")
            raise
        except requests.exceptions.JSONDecodeError as e:
            print(f"ERROR: Failed to decode JSON response: {e}")
            print(f"Response text: {response.text}")
            raise

    def _process_ollama_vlm_messages(self, messages):
        """
        Process message format for Ollama VLM models.
        
        :param messages: Original messages with possible image content
        :return: Processed messages in Ollama VLM format
        """
        processed_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # If content is a string, keep it as is
            if isinstance(content, str):
                processed_messages.append({
                    "role": role,
                    "content": content
                })
                continue
            
            # Handle multimodal content (list with text and images)
            if isinstance(content, list):
                processed_content = ""
                for item in content:
                    if item.get("type") == "text":
                        processed_content += item["text"]
                    elif item.get("type") == "image":
                        # Handle image: convert to base64 for Ollama
                        img_data = item.get("image_url", {}).get("url", "") 
                        
                        # If the image is a data URL
                        if img_data.startswith("data:image"):
                            # Extract the base64 part
                            img_base64 = img_data.split(",")[1]
                        elif img_data.startswith("http"):
                            # Download the image and convert to base64
                            img_response = requests.get(img_data)
                            img_response.raise_for_status()
                            img_base64 = base64.b64encode(img_response.content).decode("utf-8")
                        elif os.path.exists(img_data):
                            # Read from local file
                            with open(img_data, "rb") as img_file:
                                img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
                        else:
                            # Assume it's already base64
                            img_base64 = img_data
                        
                        # Ollama expects images in the format "![](data:image/jpeg;base64,<base64_data>)"
                        processed_content += f"\n![](data:image/jpeg;base64,{img_base64})\n"
                
                processed_messages.append({
                    "role": role,
                    "content": processed_content
                })
        
        return processed_messages


class Response:
    """Wrapper class for API responses to provide consistent attribute access"""
    def __init__(self, response_data):
        self.raw_response = response_data
        self.choices = []
        
        # Process choices and create Choice objects
        if 'choices' in response_data:
            self.choices = [Choice(choice) for choice in response_data['choices']]
        
        # Add other common response fields
        self.id = response_data.get('id')
        self.created = response_data.get('created')
        self.model = response_data.get('model')
        self.usage = response_data.get('usage', {})

    def __str__(self):
        if self.choices and len(self.choices) > 0:
            return self.choices[0].message.content
        return str(self.raw_response)


class Choice:
    """Wrapper class for individual choices in the response"""
    def __init__(self, choice_data):
        self.index = choice_data.get('index')
        self.finish_reason = choice_data.get('finish_reason')
        
        # Create Message object if message data exists
        message_data = choice_data.get('message', {})
        self.message = Message(message_data)


class Message:
    """Wrapper class for message content"""
    def __init__(self, message_data):
        self.role = message_data.get('role')
        self.content = message_data.get('content')
        # Add any additional message attributes that might be provider-specific
        for key, value in message_data.items():
            if not hasattr(self, key):
                setattr(self, key, value)