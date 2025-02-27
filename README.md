# Dictionary API Server

A simple OpenAPI-compliant server for Lingvo DSL dictionary lookups

## Features

- Fast dictionary lookups (<500ms response time)
- Complete preservation of linguistic data from DSL dictionaries
- Support for stress marks, conjugations, declensions, and other linguistic features
- Multi-word lookup capabilities
- Inverted indexing for efficient searches
- Automatic caching of parsed dictionaries for faster startup
- OpenAPI documentation with Swagger UI

## Requirements

- Python 3.8 or higher
- FastAPI and dependencies (see requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd dictionary-api-server
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Server

Start the server with:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

The first time you start the server, it will parse all DSL dictionaries in the `Dicts` directory. This may take some time depending on the number and size of dictionaries. Subsequent starts will be faster as it uses the cached parsed data.

### API Endpoints

Once the server is running, you can access the following endpoints:

- **GET /dictionaries**: List all available dictionaries with their statistics
- **GET /lookup/{word}**: Look up a specific word or phrase
- **GET /multi-lookup?query=word1 word2**: Look up multiple words at once
- **GET /search?query=text**: Search within dictionary entries
- **GET /health**: Health check endpoint

### API Documentation

The OpenAPI documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Dictionary Format

The server reads dictionaries in the DSL (Dictionary Specification Language) format, which is used by programs like ABBYY Lingvo and GoldenDict. 

The DSL files should be placed in the `Dicts` directory. The server supports the following DSL file properties:
- Dictionary metadata in .ann files
- All DSL markup for formatting
- Stress marks, part of speech tags, etc.

## Performance Optimizations

This server is optimized for performance:
- In-memory storage for fast lookups
- Inverted indexing for efficient word searches
- Normalization of headwords for case-insensitive matching
- Caching of parsed dictionaries to avoid re-parsing on restart
- Asynchronous endpoints for concurrent requests

## License

[MIT License](LICENSE)
