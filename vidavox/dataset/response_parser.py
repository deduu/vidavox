import json
import re
from typing import List, Dict

from .models import KeywordPair

class ResponseParser:
    """Parses LLM responses to extract structured data like keyword pairs."""
    
    @staticmethod
    def parse_keyword_pairs(content: str) -> List[KeywordPair]:
        """
        Parse the LLM response to extract keyword pairs from either sentences or questions.
        
        Args:
            content: The LLM response content
        
        Returns:
            List of KeywordPair objects (each containing a sentence/question and its corresponding keywords)
        """
        print(f"content: {content}")
        try:
            # First, try extracting a well-formatted JSON array
            matches = re.findall(r'\[\s*\{\s*"(sentence|question)"\s*:.*?\}\s*\]', content, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        data = json.loads(match)
                        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                            if all(("sentence" in item or "question" in item) and "keywords" in item for item in data):
                                return [
                                    KeywordPair(item.get("sentence") or item.get("question"), item["keywords"])
                                    for item in data
                                ]
                    except json.JSONDecodeError:
                        pass

            # Second, use a more lenient regex pattern
            match = re.search(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                        if all(("sentence" in item or "question" in item) and "keywords" in item for item in data):
                            return [
                                KeywordPair(item.get("sentence") or item.get("question"), item["keywords"])
                                for item in data
                            ]
                except json.JSONDecodeError:
                    pass
            
            # Third, clean the content and extract JSON manually
            cleaned_content = re.sub(r'```json\s*|\s*```', '', content)
            start_idx = cleaned_content.find('[')
            end_idx = cleaned_content.rfind(']')

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = cleaned_content[start_idx:end_idx+1]
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                        if all(("sentence" in item or "question" in item) and "keywords" in item for item in data):
                            return [
                                KeywordPair(item.get("sentence") or item.get("question"), item["keywords"])
                                for item in data
                            ]
                except json.JSONDecodeError:
                    pass

            # Fourth, manually extract sentences/questions and keywords
            text_pattern = r'"(sentence|question)"\s*:\s*"([^"]*)"'
            keywords_pattern = r'"keywords"\s*:\s*\[(.*?)\]'

            texts = re.findall(text_pattern, content)
            keyword_lists = re.findall(keywords_pattern, content)

            if texts and keyword_lists and len(texts) == len(keyword_lists):
                result = []
                for i in range(len(texts)):
                    _, text_value = texts[i]  # Extracting text value from the tuple
                    keywords_str = keyword_lists[i]
                    keywords = re.findall(r'"([^"]*)"', keywords_str)
                    result.append(KeywordPair(text_value, keywords))
                return result
                
            raise ValueError("Could not extract valid JSON with sentence/question-keyword pairs")

        except Exception as e:
            raise ValueError(f"Failed to parse response: {str(e)}\nResponse content: {content[:500]}...")
