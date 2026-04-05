"""Base class for dataset adapters."""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class BaseDatasetAdapter(ABC):
    """Abstract base class for dataset adapters.

    Each adapter is responsible for loading a specific benchmark dataset
    and providing a unified interface for accessing questions and answers.
    """

    def __init__(self, subset_csv: Optional[str] = None):
        """Initialize the adapter.

        Args:
            subset_csv: Optional path to a CSV file containing a subset of question IDs
                       to evaluate. If None, use the full dataset.
        """
        self.subset_csv = subset_csv
        self._df = None

    @abstractmethod
    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset and return as a DataFrame.

        Returns:
            DataFrame with at least the columns returned by get_question_column()
            and get_answer_column()
        """
        pass

    @abstractmethod
    def get_question_column(self) -> str:
        """Return the name of the column containing questions.

        Returns:
            Column name string
        """
        pass

    @abstractmethod
    def get_answer_column(self) -> str:
        """Return the name of the column containing gold answers.

        Returns:
            Column name string
        """
        pass

    def get_question_type_column(self) -> Optional[str]:
        """Return the name of the column containing question types (if available).

        Returns:
            Column name string or None if not available
        """
        return None

    def get_metadata_columns(self) -> List[str]:
        """Return a list of additional metadata columns to preserve.

        Returns:
            List of column name strings
        """
        return []

    @property
    def name(self) -> str:
        """Return the dataset name."""
        return self.__class__.__name__.replace("Adapter", "").lower()

    def _apply_subset_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply subset filter if subset_csv is provided.

        Args:
            df: Full dataset DataFrame

        Returns:
            Filtered DataFrame
        """
        if self.subset_csv is None:
            return df

        try:
            subset_df = pd.read_csv(self.subset_csv)
            # Try to find a column that might be the ID column
            id_cols = ['id', 'question_id', 'financebench_id', 'idx']
            for col in id_cols:
                if col in subset_df.columns and col in df.columns:
                    return df[df[col].isin(subset_df[col])]

            # If no ID column found, use row indices
            if 'index' in subset_df.columns:
                return df.iloc[subset_df['index'].tolist()]

            print(f"Warning: Could not apply subset filter from {self.subset_csv}")
            return df

        except Exception as e:
            print(f"Warning: Failed to load subset file {self.subset_csv}: {e}")
            return df
