# assignment/documents/dqm.py

import chromadb
import numpy as np
from typing import List, Callable, Tuple, Optional


class DocumentQueryModel:
    """
    A simplified Document Query Model that uses a single collection with a flexible pipeline strategy 
    for document indexing and query preprocessing.
    
    Attributes:
        client: ChromaDB client to manage document storage and querying.
        collection_name: Name of the ChromaDB collection.
        collection: ChromaDB collection for storing documents and their embeddings.
        ef: The embedding function to use for indexing documents.
        preprocess: A callable strategy for preprocessing and indexing documents and queries.

    Usage:
        embedding_function = lambda x: np.random.rand(1, 100)  # Example embedding function that returns random embeddings
        preprocess = lambda x: x.split()  # Example preprocessing function
        dqm = DocumentQueryModel(preprocess, embedding_function, collection_name="my_collection")

        dqm = DocumentQueryModel(preprocess, HuggingFaceEmbedding(), ...)
        # Index a document
        dqm.insert("doc1", "This is a sample document.")

        print(f"Document count: {dqm.document_count}")

        # Query the document
        query_result = dqm.query("This is a sample query.")

        # Delete all documents
        dqm.clear()

    """

    def __init__(self, preprocess: Callable[[str], List[str]], embedding_function: Callable[[str], np.ndarray],
                 chromadb_path: Optional[str] = None, collection_name: str = "document_collection"):
        """
        Initializes the DocumentQueryModel with a ChromaDB client and collection.

        Args:
            preprocess: A callable strategy (function) that preprocesses a document for indexing and querying.
            embedding_function: A callable strategy (function) that calculates the embedding for a document.
            chromadb_path: Optional path to a persistent ChromaDB client. If provided, the client will be initialized with this path.
            collection_name: Name of the ChromaDB collection.
        """
        # Initialize the ChromaDB client and the collection
        if chromadb_path:
            self.client = chromadb.PersistentClient(path=chromadb_path)
        else:
            self.client = chromadb.Client()

        self.collection_name = collection_name

        # Initialize the embedding function
        self.ef = embedding_function

        # Set the provided pipeline strategy for processing documents and queries
        self.preprocess = preprocess

        # Initialize the collection if it doesn't exist
        self._init_collection()

    def _init_collection(self):
        """
        Initializes the collection if it doesn't exist.
        """
        try:
            self.collection = self.client.get_collection(self.collection_name) 
        except ValueError:
            self.collection = self.client.create_collection(self.collection_name)

    def _calculate_embedding(self, text: str) -> np.ndarray:
        """
        Calculates the embedding for a given text using the specified embedding function
        and the current pipeline strategy.

        Args:
            text: The text for which to calculate the embedding.
        Returns:
            List[float]: The calculated embedding.
        """

        # Use the pipeline strategy to preprocess the document and convert it to a vector
        tokens = self.preprocess(text)

        # Convert the preprocessed document to a vector
        embedding = self.ef(' '.join(tokens))

        return embedding

    @property
    def document_count(self) -> int:
        """
        Returns the total number of documents in the collection.

        Returns:
            int: The count of documents in the collection.

        Usage:
            document_count = dqm.document_count
            print(f"Total documents in the collection: {document_count}")
        """
        return self.collection.count()

    def insert(self, doc_id: str, doc: str) -> np.ndarray:
        """
        Inserts a document into the collection after preprocessing with the pipeline strategy.

        Args:
            doc_id: A unique identifier for the document.
            doc: The document to insert.
        """

        # Calculate the embedding for the document
        embedding = self._calculate_embedding(doc)

        # Add the document and its embedding to the ChromaDB collection
        self.collection.add(
            documents=[doc],
            embeddings=[embedding.tolist()],
            ids=[doc_id],
        )

        return embedding

    def query(self, query_text: str, top_n: int = 5, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Queries the indexed documents using the DQM preprocessing/embedding strategy.

        Args:
            query_text: The query as a string.
            top_n: Number of top results to return (default: 5).
            threshold: The similarity threshold for considering a document relevant.

        Returns:
            A list of document IDs of the top-k results based on similarity, and their distances.
        """

        # Calculate the embedding for the query
        embedding = self._calculate_embedding(query_text)

        return self.query_embedding(embedding, top_n, threshold)

    def query_embedding(self, embedding: np.ndarray, top_n: int = 5, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Queries the indexed documents using the provided embedding.

        Args:
            embedding: The word embedding to match against.
            threshold: The similarity threshold for considering a document relevant.
            top_n: Number of top results to return (default: 5).

        Returns:
            A list of document IDs of the top-k results based on similarity, and their distances.
        """

        # Perform the query in ChromaDB
        results = self.collection.query(query_embeddings=embedding.tolist(), n_results=top_n)

        # Return the matched document IDs
        return [
            # The results are the IDs and distances
            (i, d)
            # Use zip to pair the IDs and distances
            for i, d in zip(list(results['ids'][0]), list(results['distances'][0]))
            # Filter out documents that don't meet the threshold
            if threshold is None or d < threshold
        ]

    def get_document(self, doc_id: str) -> Optional[str]:
        """
        Retrieves a document from the collection.

        Args:
            doc_id: The ID of the document to retrieve.

        Returns:
            The document content as a string, or None if the document is not found.
        """
        try:
            result = self.collection.get(ids=[doc_id])
            return result['documents'][0]
        except IndexError:
            return None
    
    def clear(self):
        """
        Clears the collection.
        """
        results = self.collection.get()
        if len(results['ids']) > 0:
            self.collection.delete(results['ids'])