# ELMS Test Suite

## Natural Language Edge Case Testing

The test suite validates that the ELMS system can handle real-world natural language inputs dynamically, without hardcoding or fallbacks.

### Running Tests

```bash
# Start the server first
python3 serv_vectionary.py

# Run natural language tests
python3 tests/test_natural_language_edge_cases.py
```

### Test Coverage

Tests cover:
- Basic copula patterns ("X is a Y")
- Verb-based queries with collective nouns ("Who makes decisions?")
- Possessive relationships ("Who are Mary's children?")
- Conjunctions ("Alice and Bob are students")
- Family relationships
- Multiple attributes

### Philosophy

All tests use natural language exactly as users would type it. The system must:
- Handle all inputs dynamically (no hardcoding)
- Leverage Vectionary for parsing
- Document Vectionary limitations (not system failures)
- Provide no fallbacks - if it works, it's dynamic; if it doesn't, it's documented

