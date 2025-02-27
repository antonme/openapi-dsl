# Task: Build a High-Performance Dictionary API Server in Python

## Objective
Create an OpenAPI-compliant server in Python that provides dictionary services with these key requirements:

- **Source**: Convert DSL dictionaries (documentation: https://anatoly314.github.io/dsl-manual/) to a format optimized for performance
- **Response Time**: All queries must complete in under 500ms
- **Data Completeness**: Preserve 100% of the original dictionary data

## Core Requirements

1. **API Endpoints**:
   - List all available dictionaries with their statistics
   - Look up words with complete linguistic information
   - Support filtering by dictionary or word properties
   - **Support multi-word queries returning full articles for each word**

2. **Linguistic Data Support**:
   - Word stresses/accents
   - Conjugations and declensions
   - Grammatical cases
   - Syllable breakdowns
   - Root word identification
   - Inflection patterns
   - Parts of speech classification
   - Phrases and idiomatic expressions
   - Spelling variants and rules
   - All other metadata present in source dictionaries

3. **Implementation Considerations**:
   - Use standard data formats (JSON/Protocol Buffers)
   - Implement proper error handling with informative messages
   - Design for horizontal scalability
   - Include comprehensive API documentation
   - Add request rate limiting
   - **Optimize for efficient multi-word lookups**

## Validation Criteria
Before submission, verify:

1. The server starts and responds to API requests
2. Sample word lookup (e.g., "абажур") returns complete information
3. Response time stays below 500ms under load
4. Output matches expected schema and contains all source data
5. API documentation is complete and accurate
6. Server handles edge cases (not found words, malformed requests)
7. Phrase lookups return expected results with proper context
8. Spelling variations are properly handled and returned
9. **Multi-word queries (e.g., "стол стул") return full articles for each word**
10. **Batch query performance scales efficiently with the number of words**

## Deliverables
1. Source code with build instructions
2. OpenAPI specification document
3. Performance benchmarks
4. Sample requests and expected responses
5. Deployment instructions