# Natural Language to Propositional Logic API

A functional live API that converts natural English into propositional logic statements, built as part of the ELMSLAB project for advancing true reasoning models.

## Project Overview

This project addresses the challenge of converting natural language text into formal propositional logic formulas. Unlike current "reasoning models" that merely mimic reasoning through text generation, this system provides a foundation for true automated reasoning using human-readable language.

## Features

- **Natural Language Parsing**: Convert English sentences to propositional logic formulas
- **Multiple Logical Connectives**: Support for AND (∧), OR (∨), NOT (¬), IMPLIES (→), and IFF (↔)
- **Normal Form Conversion**: Convert formulas to Conjunctive Normal Form (CNF) and Disjunctive Normal Form (DNF)
- **Truth Table Generation**: Generate complete truth tables for any formula
- **Formula Validation**: Validate formula syntax and properties
- **REST API**: Full REST API with comprehensive endpoints
- **Batch Processing**: Process multiple texts simultaneously

## Architecture

The system is built with a clean separation of concerns:

```
src/
├── api/           # FastAPI REST endpoints
├── core/          # Natural language processing engine
├── models/        # Propositional logic data structures
└── utils/         # Utility functions and helpers
```

### Core Components

1. **Propositional Logic Models** (`src/models/propositional_logic.py`)
   - Atomic formulas, connectives, and compound formulas
   - CNF/DNF conversion algorithms
   - Formula evaluation and manipulation

2. **Natural Language Parser** (`src/core/nlp_parser.py`)
   - spaCy-based text processing
   - Logical operator detection
   - Proposition extraction and structuring

3. **API Endpoints** (`src/api/main.py`)
   - RESTful API with FastAPI
   - Request/response validation with Pydantic
   - Comprehensive error handling

4. **Utility Functions** (`src/utils/formula_utils.py`)
   - Truth table generation
   - Formula validation and analysis
   - Logical equivalence checking

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ELMSLAB
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install spaCy language model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Running the Demo

```bash
python demo.py
```

This will demonstrate:
- Basic propositional logic formula creation
- Natural language to logic conversion
- Truth table generation
- CNF/DNF conversion
- Formula validation

### Running the API Server

```bash
python -m uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### Convert Text
```http
POST /convert
Content-Type: application/json

{
  "text": "It is raining and the ground is wet",
  "include_cnf": true,
  "include_dnf": true,
  "include_truth_table": true
}
```

**Response**:
```json
{
  "original_text": "It is raining and the ground is wet",
  "propositional_formula": "(raining ∧ ground_wet)",
  "confidence": 0.85,
  "atoms": ["raining", "ground_wet"],
  "cnf_formula": "(raining ∧ ground_wet)",
  "dnf_formula": "(raining ∧ ground_wet)",
  "truth_table": {
    "atoms": ["ground_wet", "raining"],
    "rows": [
      {"values": [false, false], "result": false},
      {"values": [false, true], "result": false},
      {"values": [true, false], "result": false},
      {"values": [true, true], "result": true}
    ],
    "is_tautology": false,
    "is_contradiction": false,
    "is_satisfiable": true
  }
}
```

### Batch Convert
```http
POST /batch_convert
Content-Type: application/json

{
  "texts": [
    "It is raining",
    "The ground is wet",
    "If it rains, then the ground is wet"
  ],
  "include_cnf": true
}
```

### Validate Formula
```http
POST /validate
Content-Type: application/json

{
  "formula": "(p ∧ q) → r"
}
```

### Get Examples
```http
GET /examples
```

## Examples

### Basic Conversions

| Natural Language | Propositional Logic |
|------------------|-------------------|
| "It is raining and the ground is wet" | `(raining ∧ ground_wet)` |
| "If it rains, then the ground will be wet" | `(rain → ground_wet)` |
| "Either it is sunny or it is cloudy" | `(sunny ∨ cloudy)` |
| "It is not raining" | `¬rain` |
| "The sky is blue if and only if it is daytime" | `(sky_blue ↔ daytime)` |

### Complex Examples

| Natural Language | Propositional Logic |
|------------------|-------------------|
| "If it's not raining and the sun is shining, then it's a good day" | `((¬rain ∧ sun_shining) → good_day)` |
| "Either the meeting is cancelled or we need to prepare" | `(meeting_cancelled ∨ need_prepare)` |
| "The project succeeds if and only if the team works hard and the budget is sufficient" | `(project_succeeds ↔ (team_works_hard ∧ budget_sufficient))` |

## Testing

Run the test suite:

```bash
pytest tests/
```

## Technical Details

### Propositional Logic Foundation

The system implements a complete propositional logic framework with:

- **Atomic Formulas**: Basic propositional variables
- **Logical Connectives**: ¬, ∧, ∨, →, ↔
- **Normal Forms**: CNF and DNF conversion
- **Truth Tables**: Complete semantic analysis
- **Formula Validation**: Syntax and semantic checking

### Natural Language Processing

The NLP component uses:

- **spaCy**: For advanced text processing and linguistic analysis
- **Pattern Matching**: For logical operator detection
- **Proposition Extraction**: For identifying atomic statements
- **Confidence Scoring**: For conversion quality assessment

### API Design

The REST API follows best practices:

- **FastAPI**: Modern, fast web framework
- **Pydantic**: Data validation and serialization
- **OpenAPI**: Automatic API documentation
- **Error Handling**: Comprehensive error responses
- **CORS Support**: Cross-origin resource sharing

## Future Enhancements

1. **First-Order Logic**: Extend to predicate logic with quantifiers
2. **Advanced NLP**: Integration with transformer models
3. **Logical Reasoning**: Automated theorem proving
4. **Knowledge Graphs**: Integration with semantic networks
5. **Multi-language Support**: Support for multiple languages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is part of the ELMSLAB research initiative.

## Acknowledgments

- Based on research in formal logic and natural language processing
- Inspired by advances in neurosymbolic AI
- Built with modern Python web technologies

---

**ELMSLAB CPSC26-08** - Advancing True Reasoning Models
