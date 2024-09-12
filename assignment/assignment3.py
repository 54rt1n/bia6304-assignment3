# assignment/assignment3.py

"""
# Assignment 3

Name:
Date:
Tools Used:

## Assignment

The following functions are part of a package that is used to work with documents.

The assignment is to complete the implementations of the missing code.

## Submission

Complete this file and submit it to Canvas, as yourname-assignment3.py

The final implementation should pass all test cases, implement the logic, and have the questions
at the end of the file answered.

"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

from .config import Config
from .documents import DocumentQueryModel, HuggingFaceEmbedding

def get_model(config: Config) -> DocumentQueryModel:
    """
    Return a DocumentQueryModel with a HuggingFaceEmbedding.
    """

    raise ValueError("Not implemented")

def load_file(file_path: str) -> pd.DataFrame:
    """
    Load a file from a given path and return a pandas dataframe.

    Args:
        file_path (str): The path to the file.

    Returns:
        pd.DataFrame: The dataframe containing the data from the file.
    """

    raise ValueError("Not implemented")

def store_document(dqm: DocumentQueryModel, doc_id: str, doc_body: str) -> np.ndarray:
    """
    Using the DocumentQueryModel and the HuggingFaceEmbedding, this pipeline
    tokenizes the text, and stores the embeddings in a DocumentQueryModel.
    
    Args:
        dqm (DocumentQueryModel): The DocumentQueryModel to store the embeddings in.
        doc_id (str): The id of the document.
        doc_body (str): The input text to tokenize.
        
    Returns:
        The embedding of the text.
    """

    raise ValueError("Not implemented")

def document_indexer_pipeline(config: Config) -> int:
    """
    A simple document indexer pipeline that reads a file and stores the embeddings in a DocumentQueryModel.

    Args:
        config (Config): The configuration for the pipeline.

    Returns:
        int: The number of documents indexed.
    """

    raise ValueError("Not implemented")

def document_search_pipeline(config: Config, query_text: str, top_n: int = 1) -> List[Tuple[str, float, str]]:
    """
    A simple document search pipeline that returns the top n most relevant documents.

    It will use the DocumentQueryModel to query the embeddings and return the top n most relevant documents.

    Args:
        config (Config): The configuration for the pipeline.
        query_text (str): The text to search for.
        top_n (int): The number of documents to return.

    Returns:
        List[Tuple[str, float, str]]: The top n most relevant documents.
          The first element is the document id, the second is the similarity score, and the third is the document content.
    """
    
    raise ValueError("Not implemented")

"""
## Questions (2-4 paragraphs each)

### What difficulties did you have with this assignment?

Your answer here

### Did you notice any trends between document scores and relevance?

Your answer here

### Consider a usecase in your current job or academic career where a vector datastore would have been useful.

Your answer here

### How do you think using a different corpus would change the results?

Your answer here

"""