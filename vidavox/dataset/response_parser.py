import json
import re
from typing import List

from .models import KeywordPair

class ResponseParser:
    """Parses LLM responses to extract structured data like keyword pairs."""
    
    @staticmethod
    def parse_keyword_pairs(content: str) -> List[KeywordPair]:
        """
        Parse the LLM response to extract keyword pairs.
        
        Args:
            content: The LLM response content
        
        Returns:
            List of KeywordPair objects
        """
        # Multiple approaches to extract JSON
        try:
            # First try: find JSON array pattern with improved regex
            matches = re.findall(r'\[\s*\{\s*"sentence"\s*:.*?\}\s*\]', content, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        data = json.loads(match)
                        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                            if all("sentence" in item and "keywords" in item for item in data):
                                return [KeywordPair(item["sentence"], item["keywords"]) for item in data]
                    except json.JSONDecodeError:
                        pass
            
            # Second try: look for a more relaxed pattern
            match = re.search(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                        if all("sentence" in item and "keywords" in item for item in data):
                            return [KeywordPair(item["sentence"], item["keywords"]) for item in data]
                except json.JSONDecodeError:
                    pass
            
            # Third try: use a much more lenient approach - find all JSON-like objects
            # Remove markdown code block syntax if present
            cleaned_content = re.sub(r'```json\s*|\s*```', '', content)
            
            # Try to find the start and end of a JSON array
            start_idx = cleaned_content.find('[')
            end_idx = cleaned_content.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = cleaned_content[start_idx:end_idx+1]
                try:
                    data = json.loads(json_str)
                    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                        if all("sentence" in item and "keywords" in item for item in data):
                            return [KeywordPair(item["sentence"], item["keywords"]) for item in data]
                except json.JSONDecodeError:
                    pass
            
            # Last resort: manual parsing
            # Look for sentence and keywords patterns directly
            sentence_pattern = r'"sentence"\s*:\s*"([^"]*)"'
            keywords_pattern = r'"keywords"\s*:\s*\[(.*?)\]'
            
            sentences = re.findall(sentence_pattern, content)
            keyword_lists = re.findall(keywords_pattern, content)
            
            if sentences and keyword_lists and len(sentences) == len(keyword_lists):
                result = []
                for i in range(len(sentences)):
                    keywords_str = keyword_lists[i]
                    # Extract keywords from the comma-separated string
                    keywords = re.findall(r'"([^"]*)"', keywords_str)
                    result.append(KeywordPair(sentences[i], keywords))
                return result
                
            raise ValueError("Could not extract valid JSON with sentence-keyword pairs")
        
        except Exception as e:
            raise ValueError(f"Failed to parse response: {str(e)}\nResponse content: {content[:500]}...")