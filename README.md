# Assignment 3: Document Indexing and Search System

## Overview

This project implements a document indexing and search system using vector embeddings. It's designed to demonstrate proficiency in natural language processing, information retrieval, and software engineering concepts. The system allows users to index a collection of documents and perform semantic searches based on vector similarity.

## Key Features

- Document indexing using HuggingFace embeddings
- Efficient document storage and retrieval with ChromaDB
- Semantic search functionality
- Command-line interface for easy interaction
- Comprehensive test suite

## Technology Stack

- Python 3.9+
- Poetry for dependency management
- pandas for data handling
- NumPy for numerical operations
- ChromaDB for vector storage and retrieval
- PyTorch and Transformers for generating embeddings
- Click for building the CLI
- Pytest for testing

## Project Structure

```
assignment3/
├── assignment/
│   ├── __init__.py
│   ├── __main__.py
│   ├── assignment3.py
│   ├── config.py
│   └── documents/
│       ├── __init__.py
│       ├── dqm.py
│       └── embedding.py
├── tests/
│   ├── integration/
│   │   └── test_store_document.py
│   └── unit/
│       └── test_assignment.py
├── assets/
│   └── documents_40.tsv
├── pyproject.toml
└── README.md
```

## Setup

1. Ensure you have Python 3.9 or higher installed.
   - If you are on windows, see `Windows Installation` below.
2. Install Poetry if you haven't already:
   ```
   pip install poetry
   ```
3. Clone the repository:
   ```
   git clone https://github.com/54rt1n/assignment3.git
   cd assignment3
   ```
4. Install dependencies:
   ```
   poetry install
   ```

## Running Tests

To run the test suite:

```
poetry run pytest
```

This will run all unit and integration tests. To see a coverage report:

```
poetry run pytest --cov=assignment --cov-report=term-missing
```

## Using the CLI

The project includes a command-line interface for indexing documents and performing searches.

### Indexing Documents

To index documents from a TSV file:

```
poetry run assignment --chromadb-path local/db.dat index /path/to/your/documents.tsv
```

By default, it will use the `assets/documents_40.tsv` file if no path is provided.

### Searching Documents

To search for documents:

```
poetry run assignment --chromadb-path local/db.dat search "your search query"
```

You can specify the number of results to return:

```
poetry run assignment --chromadb-path local/db.dat search --num 5 "your search query"
```

## Core Components

### DocumentQueryModel (DQM)

The `DocumentQueryModel` class in `documents/dqm.py` is responsible for document storage and retrieval. It uses ChromaDB as the backend for efficient vector storage and similarity search.

### HuggingFaceEmbedding

The `HuggingFaceEmbedding` class in `documents/embedding.py` generates embeddings for documents and queries using a pre-trained model from HuggingFace.

### Main Pipeline Functions

The core functionality is implemented in `assignment3.py`:

- `get_model()`: Creates a `DocumentQueryModel` instance.
- `load_file()`: Loads documents from a TSV file.
- `store_document()`: Processes and stores a single document.
- `document_indexer_pipeline()`: Indexes all documents from a file.
- `document_search_pipeline()`: Performs a search query and returns relevant documents.

## Configuration

The `Config` class in `config.py` manages configuration settings for the project. You can modify these settings to customize the behavior of the system.

## Windows Installation

Chromadb requires a C++ compiler to build. If you're on Windows, you might need to install a C++ compiler for some parts of the chromadb install.

You can download the Visual C++ Build Tools from here:

https://visualstudio.microsoft.com/visual-cpp-build-tools/

You will need to download the installer, then select Individual Components and select the MSVC v143 build tools - VS 2022 C++ x64/x86 build tools (Latest).

You will also need to install the Windows 11 SDK, whatever the latest version is.

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are correctly installed (`poetry install`).
2. Check that you're using a compatible Python version (3.9+).
3. Verify that the input data is in the correct TSV format.
4. If you're having issues with embeddings, ensure you have a stable internet connection to download the model.

## License

MIT

## Acknowledgments

- Claude Sonnet 3.5 and GPT-4o for code assistance and documentation
- HuggingFace for providing pre-trained models
- ChromaDB for efficient vector storage and retrieval
- All other open-source libraries used in this project