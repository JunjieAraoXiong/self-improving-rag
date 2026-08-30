"""Base class for dataset adapters."""

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        self._selection_metadata: Dict[str, Any] = {
            "subset_requested": subset_csv is not None,
            "subset_path": str(Path(subset_csv).resolve()) if subset_csv else None,
        }

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
            self._selection_metadata.update(
                {
                    "selection_mode": "full_dataset",
                    "source_row_count": int(len(df)),
                    "selected_row_count": int(len(df)),
                }
            )
            return df

        subset_path = Path(self.subset_csv).expanduser().resolve()
        subset_bytes = subset_path.read_bytes()
        subset_df = pd.read_csv(subset_path)
        if subset_df.empty:
            raise ValueError(f"Subset file is empty: {subset_path}")
        if df.empty:
            raise ValueError("Cannot apply a subset to an empty dataset")

        id_cols = ['id', 'question_id', 'financebench_id', 'idx']
        shared_id_cols = [
            column
            for column in id_cols
            if column in subset_df.columns and column in df.columns
        ]
        selection_mode: str
        selection_column: str
        canonical_ids: List[str]

        if shared_id_cols:
            selection_column = shared_id_cols[0]
            requested = subset_df[selection_column]
            available = df[selection_column]
            if requested.isna().any():
                raise ValueError(
                    f"Subset column {selection_column!r} contains null IDs"
                )
            if available.isna().any():
                raise ValueError(
                    f"Dataset column {selection_column!r} contains null IDs"
                )
            requested_ids = requested.astype(str)
            available_ids = available.astype(str)
            duplicate_requested = requested_ids[requested_ids.duplicated()].tolist()
            if duplicate_requested:
                raise ValueError(
                    "Subset contains duplicate IDs: "
                    f"{list(dict.fromkeys(duplicate_requested))}"
                )
            duplicate_available = available_ids[available_ids.duplicated()].tolist()
            if duplicate_available:
                raise ValueError(
                    f"Dataset column {selection_column!r} is not unique: "
                    f"{list(dict.fromkeys(duplicate_available))[:10]}"
                )
            available_set = set(available_ids)
            unknown = [value for value in requested_ids if value not in available_set]
            if unknown:
                raise ValueError(
                    "Subset contains IDs absent from the dataset: "
                    f"{unknown[:10]}"
                )
            indexed = df.assign(_selection_id=available_ids).set_index(
                "_selection_id", drop=True
            )
            filtered = indexed.loc[requested_ids.tolist()].reset_index(drop=True)
            canonical_ids = requested_ids.tolist()
            selection_mode = "id"
        elif 'index' in subset_df.columns:
            selection_column = 'index'
            raw_indices = pd.to_numeric(subset_df['index'], errors='raise')
            if raw_indices.isna().any() or any(float(value) % 1 for value in raw_indices):
                raise ValueError("Subset index values must be non-null integers")
            indices = [int(value) for value in raw_indices]
            if len(indices) != len(set(indices)):
                raise ValueError("Subset contains duplicate row indices")
            invalid = [value for value in indices if value < 0 or value >= len(df)]
            if invalid:
                raise ValueError(
                    f"Subset contains out-of-range row indices: {invalid[:10]}"
                )
            filtered = df.iloc[indices].reset_index(drop=True)
            canonical_ids = [str(value) for value in indices]
            selection_mode = "row_index"
        else:
            raise ValueError(
                "Subset file has no supported selection column shared with the "
                "dataset; expected one of id, question_id, financebench_id, idx, "
                "or index"
            )

        if filtered.empty:
            raise ValueError(f"Subset selected zero rows: {subset_path}")
        canonical_payload = json.dumps(
            canonical_ids, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        self._selection_metadata.update(
            {
                "selection_mode": selection_mode,
                "selection_column": selection_column,
                "source_row_count": int(len(df)),
                "requested_row_count": int(len(subset_df)),
                "selected_row_count": int(len(filtered)),
                "subset_file_sha256": hashlib.sha256(subset_bytes).hexdigest(),
                "selected_ids_sha256": hashlib.sha256(canonical_payload).hexdigest(),
            }
        )
        return filtered

    def get_selection_metadata(self) -> Dict[str, Any]:
        """Return a copy of the exact dataset/subset selection provenance."""

        return dict(self._selection_metadata)
