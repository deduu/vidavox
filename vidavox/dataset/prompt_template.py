class PromptTemplates:
    """Collection of prompt templates for different tasks."""

    @staticmethod
    def get_default_bio_transformation_template() -> str:
        """
        Get the default prompt template for transforming question-keywords pairs to BIO annotations.
        
        Returns:
            Default prompt template string.
        """
        return """
        ### Input:
        Question: {question}
        Keywords: [{keywords}]
        
        ### Instructions:
        Tokenize the question into words and punctuation. For each token, assign a BIO label as follows:
        - Label a token as "B-KEY" if it is the first token of a keyword.
        - Label a token as "I-KEY" if it is a subsequent token of a multi-token keyword.
        - Otherwise, label the token as "O".
        
        Ensure that multi-token keywords are labeled correctly. Provide your answer in the following JSON format exactly:
        
        {{
            "question": "{question}",
            "tokens": ["token1", "token2", "..."],
            "labels": ["label1", "label2", "..."]
        }}
        
        For example, 
        {{
            "question": "The quick brown fox jumps over the lazy dog?",
            "tokens": ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "?"],
            "labels": ["O", "O", "O", "B-KEY", "I-KEY", "O", "O", "B-KEY", "I-KEY", "O"]
        }}

        VERY IMPORTANT: Your response must be valid JSON with no extra commentary.
        """

    

    
    @staticmethod
    def get_default_keyword_extraction_template() -> str:
        """
        Get the default prompt template for keyword extraction.
        
        Returns:
            Default prompt template string
        """
        return """
        ### Context (Retrieved Documents):
        {context}

        ### Instructions:
        You are an expert in natural language understanding and keyword extraction. Your task is to analyze the provided context and generate {n} number of structured list of **high-quality keywords** in {language}. Follow these steps carefully:

        #### **1. Identify the Most Relevant Information**
        - Extract **only the most essential sentences** that capture the core meaning of the text.
        - Ignore general statements, filler words, and redundant details.

        #### **2. Extract High-Quality Keywords**
        - Select **only the most specific and meaningful terms** that summarize key ideas.
        - Avoid **generic, stopwords, overly broad terms**, and unimportant words.
        - Focus on **proper nouns, technical terms, and critical concepts**.

        #### **3. Ensure Each Keyword is Meaningful**
        - **Keywords should be standalone and useful** in understanding the document.
        - **Avoid single letters, overly common words, or isolated verbs.**
        - If a keyword is ambiguous, **provide a more specific alternative**.

        #### **4. Structure the Output in JSON Format**
        Provide the extracted keywords in the following format:

        [
            {{"sentence": "Key sentence from the document.", "keywords": ["specific keyword 1", "specific keyword 2"]}},
            {{"sentence": "Another important sentence.", "keywords": ["critical concept 1", "critical concept 2"]}}
        ]

        VERY IMPORTANT: Your response must be a valid JSON array and nothing else. No explanatory text before or after the JSON.
        """
    
    @staticmethod
    def get_summary_template() -> str:
        """
        Get a template for summarizing text.
        
        Returns:
            Summary template string
        """
        return """
        ### Context:
        {context}
        
        ### Instructions:
        Summarize the key points from the provided context in {language}. 
        Create a concise summary with the following properties:
        - Include only the most important information
        - Maintain the original meaning and intent
        - Use clear, direct language
        - Limit to {max_length} words
        
        Format your response as valid JSON:
        
        {{
            "summary": "Your concise summary here",
            "key_points": ["Point 1", "Point 2", "Point 3"]
        }}
        """
    
    # Additional templates can be added as needed