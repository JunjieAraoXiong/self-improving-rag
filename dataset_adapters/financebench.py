"""FinanceBench dataset adapter."""

import json
from pathlib import Path
from typing import List, Optional
import pandas as pd

from dataset_adapters.base import BaseDatasetAdapter


class FinanceBenchAdapter(BaseDatasetAdapter):
    """Adapter for the FinanceBench financial QA dataset.

    FinanceBench contains questions about financial documents (10-K, 10-Q filings)
    from various public companies. Questions are categorized into:
    - metrics-generated: Numerical extraction from tables
    - domain-relevant: Domain knowledge questions
    - novel-generated: Reasoning questions

    Dataset format: JSONL with fields:
    - financebench_id: Unique identifier
    - company: Company name (e.g., "3M")
    - doc_name: Document name (e.g., "3M_2018_10K")
    - question_type: Question category
    - question: The question text
    - answer: Gold answer
    - justification: Explanation of the answer
    - evidence: List of evidence passages from the document
    """

    # Default paths relative to project root
    DEFAULT_QUESTIONS_PATH = "data/question_sets/financebench_open_source.jsonl"
    DEFAULT_DOCS_PATH = "data/question_sets/financebench_document_information.jsonl"

    def __init__(
        self,
        questions_path: Optional[str] = None,
        subset_csv: Optional[str] = None,
    ):
        """Initialize the FinanceBench adapter.

        Args:
            questions_path: Path to the questions JSONL file.
                           If None, uses default path.
            subset_csv: Optional path to CSV with subset of question IDs.
        """
        super().__init__(subset_csv=subset_csv)

        # Resolve paths relative to project root
        project_root = Path(__file__).parent.parent
        self.questions_path = questions_path or str(project_root / self.DEFAULT_QUESTIONS_PATH)

    def load_dataset(self) -> pd.DataFrame:
        """Load the FinanceBench dataset.

        Returns:
            DataFrame with questions and answers
        """
        if self._df is not None:
            return self._df

        # Load JSONL file
        records = []
        with open(self.questions_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        df = pd.DataFrame(records)

        # Apply subset filter if specified
        df = self._apply_subset_filter(df)

        self._df = df
        return df

    def get_question_column(self) -> str:
        """Return the question column name."""
        return "question"

    def get_answer_column(self) -> str:
        """Return the answer column name."""
        return "answer"

    def get_question_type_column(self) -> Optional[str]:
        """Return the question type column name."""
        return "question_type"

    def get_metadata_columns(self) -> List[str]:
        """Return additional metadata columns."""
        return [
            "financebench_id",
            "company",
            "doc_name",
            "question_reasoning",
            "justification",
        ]

    @property
    def name(self) -> str:
        """Return the dataset name."""
        return "financebench"

    def get_evidence_for_question(self, question_id: str) -> List[dict]:
        """Get the evidence passages for a specific question.

        Args:
            question_id: The financebench_id

        Returns:
            List of evidence dictionaries with 'evidence_text' and 'doc_name'
        """
        df = self.load_dataset()
        row = df[df['financebench_id'] == question_id]

        if row.empty:
            return []

        evidence = row.iloc[0].get('evidence', [])
        return evidence if isinstance(evidence, list) else []
