# assignment3/tests/unit/test_assignment.py

import numpy as np
import os
import pandas as pd
import pytest
import sys

# Bootstrap the parent directory of 'assignment' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from assignment.assignment3 import Config, get_model, load_file, store_document, document_indexer_pipeline, document_search_pipeline
from assignment.documents import DocumentQueryModel, HuggingFaceEmbedding

@pytest.fixture
def mock_config():
    return Config(data_path="test_data.tsv", chromadb_path=None, collection_name="test_collection")

@pytest.fixture
def dqm():
    embedding = HuggingFaceEmbedding()
    dqm = DocumentQueryModel(lambda x: x.lower().split(), embedding)
    # Clear our collection
    dqm.clear()
    return dqm

def test_get_model(mock_config):
    model = get_model(mock_config)
    assert isinstance(model, DocumentQueryModel)
    assert model.collection_name == "test_collection"

def test_load_file(tmp_path):
    # Create a temporary TSV file
    test_file = tmp_path / "test_data.tsv"
    test_file.write_text("id\tdocument\n1\tTest document")
    
    df = load_file(str(test_file))
    assert len(df) == 1
    assert df.iloc[0]["id"] == 1
    assert df.iloc[0]["document"] == "Test document"

def test_store_document(dqm):
    doc_id = "test_doc"
    doc_body = "This is a test document"
    embedding = store_document(dqm, doc_id, doc_body)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] > 0
    assert dqm.document_count == 1

def test_document_indexer_pipeline(mock_config, mocker):
    # Mock the load_file and store_document functions
    mocker.patch('assignment.assignment3.load_file', return_value=pd.DataFrame({'id': ['1', '2'], 'document': ['Doc 1', 'Doc 2']}))
    mocker.patch('assignment.assignment3.store_document', return_value=np.array([0.1, 0.2, 0.3]))
    
    count = document_indexer_pipeline(mock_config)
    assert count == 2

def test_document_search_pipeline(mock_config, mocker):
    # Mock the get_model function and the query method of DocumentQueryModel
    mock_model = mocker.Mock()
    mock_model.document_count = 10
    mock_model.query.return_value = [("doc1", 0.9), ("doc2", 0.8)]
    mock_model.get_document.side_effect = lambda x: f"Content of {x}"
    mocker.patch('assignment.assignment3.get_model', return_value=mock_model)
    
    results = document_search_pipeline(mock_config, "test query", top_n=2)
    assert len(results) == 2
    assert results[0] == ("doc1", 0.9, "Content of doc1")
    assert results[1] == ("doc2", 0.8, "Content of doc2")

# Additional edge case tests

def test_store_document_empty(dqm):
    doc_id = "empty_doc"
    doc_body = ""
    embedding = store_document(dqm, doc_id, doc_body)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] > 0

def test_document_search_pipeline_no_results(mock_config, mocker):
    mock_model = mocker.Mock()
    mock_model.document_count = 10
    mock_model.query.return_value = []
    mocker.patch('assignment.assignment3.get_model', return_value=mock_model)
    
    results = document_search_pipeline(mock_config, "nonexistent query")
    assert len(results) == 0

def test_load_file_empty(tmp_path):
    empty_file = tmp_path / "empty.tsv"
    empty_file.write_text("id\tdocument\n")
    
    df = load_file(str(empty_file))
    assert len(df) == 0

def test_document_indexer_pipeline_empty_file(mock_config, mocker):
    mocker.patch('assignment.assignment3.load_file', return_value=pd.DataFrame(columns=['id', 'document']))
    mocker.patch('assignment.assignment3.store_document', return_value=np.array([]))
    
    count = document_indexer_pipeline(mock_config)
    assert count == 0

# Performance test (this might take longer to run)
def test_document_indexer_pipeline_large_dataset(mock_config, mocker):
    large_df = pd.DataFrame({
        'id': [f'doc{i}' for i in range(1000)],
        'document': [f'Document content {i}' for i in range(1000)]
    })
    mocker.patch('assignment.assignment3.load_file', return_value=large_df)
    mocker.patch('assignment.assignment3.store_document', return_value=np.random.rand(384))  # Assuming 384-dimensional embeddings
    
    count = document_indexer_pipeline(mock_config)
    assert count == 1000