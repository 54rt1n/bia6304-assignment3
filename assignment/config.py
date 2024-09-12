# assignment/config.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """
    Configuration class for the assignment.
    """
    data_path: str = "assets/documents_40.tsv"
    chromadb_path: Optional[str] = None
    collection_name: str = "assignment3"
