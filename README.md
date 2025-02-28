# Lingvo DSL Dictionary API Server

A simple OpenAPI-compliant server for Lingvo DSL dictionary lookups.
> Has no official connection whatsoever to either ABBYY or Lingvo.


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

#### Main Dictionary Endpoints

- **GET /dictionaries**: List all available dictionaries with their statistics
- **GET /lookup/{word}**: Look up specific words or phrases in dictionaries
- **GET /search?query=text**: Search within dictionary entries
- **GET /health**: Health check endpoint

#### Understanding the Difference Between Lookup and Search

The API offers two distinct ways to find dictionary information:

1. **Lookup** (`/lookup/{word}`):
   - Purpose: Direct dictionary lookup for known words
   - Behavior: 
     - Returns complete dictionary entries for exact matches
     - Can handle multiple words (spaces in the URL path)
     - By default uses exact matching (can be changed with the `exact_match` parameter)
     - Returns all matching entries

2. **Search** (`/search?query=text`):
   - Purpose: Exploratory search for relevant information
   - Behavior:
     - By default searches within definitions (content search)
     - Can search only in headwords with `exact_match=true`
     - Results are limited to 10 entries by default (configurable with `limit`)
     - Better for when you don't know the exact word or want to find related information

Choose the endpoint that better fits your use case - lookup for known words, search for exploration.


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
