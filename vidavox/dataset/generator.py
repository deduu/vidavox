# rag_evaluation_generator.py

import asyncio
import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
import random
from datasets import Dataset
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm.auto import tqdm
import re
from vidavox.document.doc_processor import process_doc_file
from vidavox.generation.llm import Client
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from typing import Optional, Callable, TypeVar, Generic

# Define a type variable for the data type the parser returns.
T = TypeVar("T")

# Default prompt template.
DEFAULT_PROMPT_TEMPLATE = (
    "Generate a factual question and answer based on this context:\n\n"
    "Context: {context}\n\n"
    "Output format:\n"
    "Question: (your question)\n"
    "Answer: (your answer)"
)

MULTI_TURN_PROMPT_TEMPLATE = """
    "Generate a multi-turn conversation based on this context:\n\n
    Context: {context}\n\n
    Output format:\n
    User: (User's initial message)
    Assistant: (Assistant's response)
    User: (User's follow-up message)
    Assistant: (Assistant's follow-up response)
    
    """

NMULTI_TURN_PROMPT_TEMPLATE = """
Generate a nuanced multi-turn conversation based on this context:

Context: {context}

Output format:\n
User: (User's initial message)
Assistant: (Assistant's response)
User: (User's follow-up message)
Assistant: (Assistant's follow-up response)
...
"""


@dataclass
class QAPair:
    question: str
    answer: str
    context: str
    source_doc: str
    metadata: Dict[str, Any]

class QAEvaluator:
    def __init__(self, model_pool: Client, min_score_threshold: int = 3):
        self.model_pool = model_pool
        self.min_score_threshold = min_score_threshold
        
    async def evaluate_qa_pair(self, qa_pair: QAPair) -> Dict[str, Any]:
        """Evaluate a QA pair using multiple criteria"""
        evaluation_prompts = {
            "groundedness": self._format_groundedness_prompt(qa_pair),
            "relevance": self._format_relevance_prompt(qa_pair),
            "standalone": self._format_standalone_prompt(qa_pair),
            "non_explicitness": self._format_non_explicitness_prompt(qa_pair),
            "factual_accuracy": self._format_factual_accuracy_prompt(qa_pair),
            "answer_completeness": self._format_completeness_prompt(qa_pair)
        }
        
        results = {}
        for criterion, prompt in evaluation_prompts.items():
            response = await self.model_pool.generate(
                messages=[{"role": "user", "content": prompt}]
            )
            logger.info(f"{criterion}: {response}")
            score, evaluation = self._extract_score_and_eval(response)
            results[f"{criterion}_score"] = score
            results[f"{criterion}_eval"] = evaluation
            
        return results

    def _format_groundedness_prompt(self, qa_pair: QAPair) -> str:
        return f"""Rate how well the context supports answering the question (1-5):
Question: {qa_pair.question}
Context: {qa_pair.context}
Answer: {qa_pair.answer}

Provide your evaluation in the format:
Evaluation: (detailed reasoning)
Total rating: (score 1-5)"""

    def _format_relevance_prompt(self, qa_pair: QAPair) -> str:
        return f"""Rate how relevant the question is to the context (1-5):
Question: {qa_pair.question}
Context: {qa_pair.context}

Provide your evaluation in the format:
Evaluation: (detailed reasoning)
Total rating: (score 1-5)"""

    def _format_standalone_prompt(self, qa_pair: QAPair) -> str:
        return f"""Rate how well the question can be understood without context (1-5):
Question: {qa_pair.question}

Provide your evaluation in the format:
Evaluation: (detailed reasoning)
Total rating: (score 1-5)"""

    def _format_non_explicitness_prompt(self, qa_pair: QAPair) -> str:
        return f"""Rate how much reasoning is required to answer the question (1-5):
Question: {qa_pair.question}
Answer: {qa_pair.answer}
Context: {qa_pair.context}

Provide your evaluation in the format:
Evaluation: (detailed reasoning)
Total rating: (score 1-5)"""

    def _format_factual_accuracy_prompt(self, qa_pair: QAPair) -> str:
        return f"""Rate the factual accuracy of the answer based on the context (1-5):
Question: {qa_pair.question}
Answer: {qa_pair.answer}
Context: {qa_pair.context}

Provide your evaluation in the format:
Evaluation: (detailed reasoning)
Total rating: (score 1-5)"""

    def _format_completeness_prompt(self, qa_pair: QAPair) -> str:
        return f"""Rate how completely the answer addresses the question (1-5):
Question: {qa_pair.question}
Answer: {qa_pair.answer}
Context: {qa_pair.context}

Provide your evaluation in the format:
Evaluation: (detailed reasoning)
Total rating: (score 1-5)"""

    @staticmethod
    def _extract_score_and_eval(response: str) -> tuple[Optional[int], Optional[str]]:
        """Extract numerical score and evaluation text from model response"""
        try:
            import re
            score_match = re.search(r"Total rating:\s*(\d+)", response)
            eval_match = re.search(r"Evaluation:\s*(.+?)(?=Total rating:|$)", response, re.DOTALL)
            
            score = int(score_match.group(1)) if score_match else None
            evaluation = eval_match.group(1).strip() if eval_match else None
            
            return score, evaluation
        except Exception as e:
            logger.error(f"Error extracting score and evaluation: {e}")
            return None, None
# A generic QA pair that can store any data type in the 'data' field.
class QAPair(Generic[T]):
    def __init__(self, data: T, context: str, source_doc: str, metadata: dict):
        self.data = data
        self.context = context
        self.source_doc = source_doc
        self.metadata = metadata
    
    @property
    def question(self):
        return self.data.get("question")
    
    @property
    def answer(self):
        return self.data.get("answer")

    def __repr__(self):
        return (f"QAPair(question={self.question}, answer={self.answer}, "
                f"context={self.context}, source_doc={self.source_doc})")
def multi_turn_parser(response: str, doc: "Document") -> Optional[QAPair[dict]]:
        """
        Parse a multi-turn conversation from the model's response.
        Expected format:
            User: (User's initial message)
            Assistant: (Assistant's response)
            User: (User's follow-up message)
            Assistant: (Assistant's follow-up response)
            ...
        """
        # Regex to find all conversation turns (User or Assistant) and their messages.
        # It looks for "User:" or "Assistant:" followed by any text until the next occurrence or the end of the string.
        turns = re.findall(r"(User|Assistant):\s*(.*?)(?=(?:User:|Assistant:|$))", response, re.DOTALL)
        if turns:
            conversation = [
                {"speaker": speaker, "message": message.strip()} 
                for speaker, message in turns if message.strip()
            ]
            data = {"conversation": conversation}
            return QAPair(
                data=data,
                context=doc.page_content,
                source_doc=doc.metadata.get("source", ""),
                metadata=doc.metadata
            )
        return None
class RAGEvaluationDatasetGenerator:
    def __init__(
        self,
        model_pool: Client,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        min_eval_score: int = 3,
        max_answer_length: int = 300,
        num_samples_per_chunk: int = 3
    ):
        self.model_pool = model_pool
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_eval_score = min_eval_score
        self.max_answer_length = max_answer_length
        self.num_samples_per_chunk = num_samples_per_chunk
        self.evaluator = QAEvaluator(model_pool, min_eval_score)

    async def generate_dataset(self, file_paths: List[str]) -> Dataset:
        """Generate and evaluate QA pairs from documents"""
        # Process documents
        docs = []
        for file_path in file_paths:
            chunks = process_doc_file(
                file_path,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            docs.extend(chunks)
        
        # Generate QA pairs
        qa_pairs = await self._generate_qa_pairs(docs)

        logger.info(f"Generated qa pairs: {qa_pairs}")
        
        # Evaluate and filter QA pairs
        # evaluated_pairs = await self._evaluate_qa_pairs(qa_pairs)

        # logger.info(f"Evaluated qa pairs: {evaluated_pairs}")
        
        # # Convert to dataset
        # return self._create_dataset(evaluated_pairs)




    # Default parser function which extracts a question and answer using regex.
    def default_qa_parser(response: str, doc: "Document") -> Optional[QAPair[dict]]:
        question_match = re.search(r"Question:\s*(.+?)(?=Answer:|$)", response, re.DOTALL)
        answer_match = re.search(r"Answer:\s*(.+?)$", response, re.DOTALL)
        if question_match and answer_match:
            data = {
                "question": question_match.group(1).strip(),
                "answer": answer_match.group(1).strip()
            }
            return QAPair(
                data=data,
                context=doc.page_content,
                source_doc=doc.metadata.get("source", ""),
                metadata=doc.metadata
            )
        return None


    
    # The generic async function that accepts optional custom prompt template and parser.
    async def _generate_single_qa_pair(
        self, 
        doc: Document,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        parser: Callable[[str, Document], Optional[T]] = default_qa_parser
    ) -> Optional[T]:
        """
        Generate a single QA pair (or any custom format) from a document chunk.

        Parameters:
            doc: The document chunk.
            prompt_template: A string template with placeholders (e.g. {context})
                            to be used to generate the prompt. Defaults to DEFAULT_PROMPT_TEMPLATE.
            parser: A callable that accepts the raw model response and the document,
                    and returns an object of type T. Defaults to default_qa_parser.
        
        Returns:
            An object of type T as returned by the parser, or None on failure.
        """
        prompt = prompt_template.format(context=doc.page_content)
        logger.info(f"Prompt: {prompt}")
        try:
            response = self.model_pool.chat.completions.create(
                messages=[{"role": "user", "content": prompt}]
            )
            logger.info(f"Response: {response.choices[0].message.content}")
            # Use the parser function to convert the raw response into your desired format.
            result = parser(response.choices[0].message.content, doc)
            logger.info(f"Result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error generating QA pair: {e}")
            return None

    async def _generate_qa_pairs(self, docs: List[Document]) -> List[QAPair]:
        """Generate multiple QA pairs per document chunk"""
        qa_pairs = []
        for doc in tqdm(docs[:1], desc="Generating QA pairs"):
            for _ in range(self.num_samples_per_chunk):
                try:
                    for i in [MULTI_TURN_PROMPT_TEMPLATE, NMULTI_TURN_PROMPT_TEMPLATE]:
                        qa_pair = await self._generate_single_qa_pair(doc, prompt_template=i, parser=multi_turn_parser)
                        print(f"qa_pair: {qa_pair}")
                        if qa_pair and len(qa_pair.answer) < self.max_answer_length:
                            qa_pairs.append(qa_pair)
                except Exception as e:
                    logger.error(f"Error generating QA pair: {e}")
        return qa_pairs

    async def _evaluate_qa_pairs(self, qa_pairs: List[QAPair]) -> List[Dict[str, Any]]:
        """Evaluate and filter QA pairs"""
        evaluated_pairs = []
        for qa_pair in tqdm(qa_pairs, desc="Evaluating QA pairs"):
            try:
                evaluation_results = await self.evaluator.evaluate_qa_pair(qa_pair)
                logger.info(f"evaluation_results: {evaluation_results}")
                if self._passes_quality_threshold(evaluation_results):
                    evaluated_pairs.append({
                        **qa_pair.__dict__,
                        **evaluation_results
                    })
            except Exception as e:
                logger.error(f"Error evaluating QA pair: {e}")
        return evaluated_pairs

    def _passes_quality_threshold(self, evaluation_results: Dict[str, Any]) -> bool:
        """Check if QA pair passes all quality thresholds"""
        required_scores = [
            "groundedness_score",
            "relevance_score",
            "standalone_score",
            "factual_accuracy_score",
            "answer_completeness_score"
        ]
        
        return all(
            evaluation_results.get(score, 0) >= self.min_eval_score
            for score in required_scores
        )

    @staticmethod
    def _create_dataset(evaluated_pairs: List[Dict[str, Any]]) -> Dataset:
        """Create and save the final dataset"""
        if not evaluated_pairs:
            raise ValueError("No QA pairs passed evaluation criteria")
            
        df = pd.DataFrame(evaluated_pairs)
        return Dataset.from_pandas(df)

async def main():
    # Load configuration
    
    # Initialize model pool
    model_pool = Client(model="ollama:llama3.2:1b")
    
    # Initialize generator
    generator = RAGEvaluationDatasetGenerator(
        model_pool=model_pool,
        chunk_size=2000,
        chunk_overlap=200,
        min_eval_score=3,
        max_answer_length=300,
        num_samples_per_chunk=3
    )
    
    # Generate dataset
    file_paths = [
        "./unrelated/Draft_POD.md",
        # "./eval/datasets/md/ksmi.md",
    ]
    
    dataset = await generator.generate_dataset(file_paths)
    
    # Save dataset
    dataset.to_csv("improved_eval_dataset.csv")
    logger.info("✅ Evaluation dataset saved to improved_eval_dataset.csv")

if __name__ == "__main__":
    asyncio.run(main())