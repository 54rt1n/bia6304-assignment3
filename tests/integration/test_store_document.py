import numpy as np
import os
import pytest
import sys
from typing import List, Dict, Tuple

# Bootstrap the parent directory of 'assignment' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from assignment.assignment3 import store_document
from assignment.documents import DocumentQueryModel, HuggingFaceEmbedding

@pytest.fixture
def sample_texts() -> List[Tuple[str, str]]:
    return [
        ("doc_1", "The quick brown fox jumps over the lazy dog."),
        ("doc_2", "A journey of a thousand miles begins with a single step."),
        ("doc_3", "To be or not to be, that is the question."),
        ("doc_4", "All that glitters is not gold."),
        ("doc_5", "Where there's a will, there's a way."),
    ]

@pytest.fixture
def dqm():
    embedding = HuggingFaceEmbedding()
    dqm = DocumentQueryModel(lambda x: x.lower().split(), embedding)
    # Clear our collection
    dqm.clear()
    return dqm

def test_store_document(sample_texts: List[Tuple[str, str]], dqm: DocumentQueryModel):
    # Store embeddings
    stored_embeddings = []
    for i, (doc_id, doc_body) in enumerate(sample_texts):
        embedding = store_document(dqm, doc_id, doc_body)
        assert isinstance(embedding, np.ndarray), f"Expected numpy array, got {type(embedding)}"
        assert embedding.shape[0] > 0, "Embedding should not be empty"
        stored_embeddings.append((doc_id, embedding))
    
    # Verify that all documents are stored
    assert dqm.document_count == len(sample_texts), "Not all documents were stored"
    
    # Query using stored embeddings
    for i, (doc_id, doc_body) in enumerate(sample_texts):
        results = dqm.query(doc_body, top_n=1)
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        retrieved_doc_id, score = results[0]
        assert retrieved_doc_id == doc_id, f"Expected {doc_id}, got {retrieved_doc_id}"
        assert 0 <= score <= 1, f"Score should be between 0 and 1, got {score}"

    # Test with a non-exact match
    mixed_embedding = (stored_embeddings[0][1] + stored_embeddings[1][1]) / 2
    results = dqm.query_embedding(mixed_embedding, top_n=2)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0][0] in ["doc_1", "doc_2"], f"Expected doc_1 or doc_2, got {results[0][0]}"
    assert results[1][0] in ["doc_1", "doc_2"], f"Expected doc_1 or doc_2, got {results[1][0]}"
    assert results[0][0] != results[1][0], "Expected different documents for top 2 results"