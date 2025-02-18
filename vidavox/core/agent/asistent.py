class AgentWorkFlow:
    def __init__(self, tools: List[callable], llm:Client, system_prompt: str):
        self.tools = tools
        self.llm = llm
        self.system_prompt = system_prompt

    @classmethod
    def from_tools_or_functions(cls, tools: List[callable], llm, system_prompt: str):
        return cls(tools, llm, system_prompt)

    def run(self, user_input: str) -> str:
        # Example of a basic workflow: call tools and combine their responses
        tool_results = [tool(user_input) for tool in self.tools]
        
        # Pass tool results and user input to the LLM for a final response
        combined_prompt = f"{self.system_prompt}\nUser input: {user_input}\nTool responses: {tool_results}"

        messages = [
        {"role": "system", "content":self.system_prompt},
        {"role": "user", "content": user_input},
        ]
        response = self.llm.chat.completions.create(messages=messages, temperature=0.75)
        
        class AgentWorkFlow:
    def __init__(self, tools: List[callable], llm:Client, system_prompt: str):
        self.tools = tools
        self.llm = llm
        self.system_prompt = system_prompt

    @classmethod
    def from_tools_or_functions(cls, tools: List[callable], llm, system_prompt: str):
        return cls(tools, llm, system_prompt)

    def run(self, user_input: str) -> str:
        # Example of a basic workflow: call tools and combine their responses
        tool_results = [tool(user_input) for tool in self.tools]
        
        # Pass tool results and user input to the LLM for a final response
        combined_prompt = f"{self.system_prompt}\nUser input: {user_input}\nTool responses: {tool_results}"

        messages = [
        {"role": "system", "content":self.system_prompt},
        {"role": "user", "content": user_input},
        ]
        response = self.llm.chat.completions.create(messages=messages, temperature=0.75)
        class AgentWorkFlow:
    def __init__(self, tools: List[callable], llm:Client, system_prompt: str):
        self.tools = tools
        self.llm = llm
        self.system_prompt = system_prompt

    @classmethod
    def from_tools_or_functions(cls, tools: List[callable], llm, system_prompt: str):
        return cls(tools, llm, system_prompt)

    def run(self, user_input: str) -> str:
        # Example of a basic workflow: call tools and combine their responses
        tool_results = [tool(user_input) for tool in self.tools]
        
        # Pass tool results and user input to the LLM for a final response
        combined_prompt = f"{self.system_prompt}\nUser input: {user_input}\nTool responses: {tool_results}"

        messages = [
        {"role": "system", "content":self.system_prompt},
        {"role": "user", "content": user_input},
        ]
        response = self.llm.chat.completions.create(messages=messages, temperature=0.75)
        return response