# Natural Language to Propositional Logic API

A functional live API that converts natural English into propositional logic statements, built as part of the ELMSLAB project for advancing true reasoning models.

## Project Overview

This project addresses the challenge of converting natural language text into formal propositional logic formulas. Unlike current "reasoning models" that merely mimic reasoning through text generation, this system provides a foundation for true automated reasoning using human-readable language.

## Features

- **Advanced spaCy Integration**: Real linguistic intelligence with dependency parsing
- **Proper Negation Handling**: Correctly processes negation using spaCy's `neg` dependency
- **Multiple Logical Connectives**: Support for AND (∧), OR (∨), NOT (¬), IMPLIES (→), and IFF (↔)
- **Truth Table Generation**: Generate complete truth tables with accurate semantics
- **Logical Inference**: Check if conclusions follow from premises
- **REST API**: Full REST API with comprehensive endpoints
- **Production Ready**: Clean, focused codebase with proper error handling

## Current Project Structure

```
ELMSLAB/
├── fixed_api.py          # Main working API server
├── test_negation.py      # Negation testing script
├── standalone_test.py    # Core functionality test
├── src/                  # Source code modules
│   ├── models/          # Propositional logic data structures
│   │   └── propositional_logic.py
│   └── utils/           # Utility functions
│       └── formula_utils.py
├── tests/                # Unit tests
│   └── test_basic.py
├── requirements.txt      # Dependencies
├── README.md            # This file
├── QUICK_START.md       # Quick start guide
└── venv/                # Virtual environment
```

## Installation

1. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies** (if needed):
   ```bash
   pip install -r requirements.txt
   ```

3. **Install spaCy language model** (if needed):
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Quick Start

### 1. Start the API Server
```bash
source venv/bin/activate
python fixed_api.py
```
The API will be available at `http://localhost:8000`

### 2. Test the API
```bash
curl -X POST "http://127.0.0.1:8000/convert" \
     -H "Content-Type: application/json" \
     -d '{"text": "It is not raining", "include_truth_table": true}' | python3 -m json.tool
```
## API Endpoints

### Convert Text to Logic
```http
POST /convert
Content-Type: application/json

{
  "text": "It is not raining and it is cold",
  "include_cnf": true,
  "include_truth_table": true
}
```

**Response**:
```json
{
  "original_text": "It is not raining and it is cold",
  "propositional_formula": "(¬it_rain ∧ cold)",
  "confidence": 0.8,
  "atoms": ["cold", "it_rain"],
  "cnf_formula": "(¬it_rain ∧ cold)",
  "truth_table": {
    "atoms": ["cold", "it_rain"],
    "rows": [
      {"values": [false, false], "result": false},
      {"values": [false, true], "result": false},
      {"values": [true, false], "result": true},
      {"values": [true, true], "result": false}
    ],
    "is_tautology": false,
    "is_contradiction": false,
    "is_satisfiable": true
  }
}
```

### Logical Inference
```http
POST /infer
Content-Type: application/json

{
  "premises": ["It is not raining and it is cold"],
  "conclusion": "It is not raining"
}
```

**Response**:
```json
{
  "valid": true,
  "premises": ["(¬it_rain ∧ cold)"],
  "conclusion": "¬it_rain",
  "implication": "((¬it_rain ∧ cold) → ¬it_rain)",
  "truth_table_summary": {
    "is_tautology": true,
    "is_contradiction": false,
    "is_satisfiable": true
  }
}
```


## Working Examples

### Negation (Fixed!)
| Natural Language | Propositional Logic | Truth Table |
|------------------|-------------------|-------------|
| "It is not raining" | `¬it_rain` | ¬false = true, ¬true = false ✓ |
| "The dog is not barking" | `¬dog_bark` | Correct negation semantics ✓ |

### Conjunctions
| Natural Language | Propositional Logic |
|------------------|-------------------|
| "It is raining and the ground is wet" | `(it_rain ∧ ground_wet)` |
| "It is not raining and it is cold" | `(¬it_rain ∧ cold)` |

### Disjunctions
| Natural Language | Propositional Logic |
|------------------|-------------------|
| "Either it is sunny or it is cloudy" | `(it_sunny ∨ it_cloudy)` |
| "We can go to the beach or stay home" | `(we_go_beach ∨ stay_home)` |

### Conditionals
| Natural Language | Propositional Logic |
|------------------|-------------------|
| "If it is not raining then we can go outside" | `(¬it_rain → we_go_outside)` |
| "If it rains, then the ground will be wet" | `(it_rains → ground_wet)` |

## Testing

### Run Negation Tests
```bash
python test_negation.py
```

### Run Core Functionality Tests
```bash
python standalone_test.py
```

### Run Unit Tests
```bash
python -m pytest tests/
```

## Technical Details

### spaCy Integration
- **Dependency Parsing**: Uses spaCy's `neg` dependency for negation detection
- **Linguistic Intelligence**: Real understanding of sentence structure
- **Proposition Extraction**: Identifies atomic propositions from natural language

### Propositional Logic Engine
- **Complete Formula System**: All logical connectives (¬, ∧, ∨, →, ↔)
- **Accurate Truth Tables**: Proper semantic evaluation
- **Logical Inference**: Valid reasoning detection
- **Normal Form Conversion**: CNF and DNF transformations

### API Features
- **FastAPI**: Modern, fast web framework
- **Pydantic**: Data validation and serialization
- **Interactive Documentation**: Built-in API docs
- **Error Handling**: Comprehensive error management

## Achievements so far

✅ **Fixed Negation Parsing**: Properly handles negation using spaCy's dependency parsing  
✅ **Accurate Truth Tables**: Correct semantic analysis for all formula types  
✅ **Logical Inference**: Valid reasoning detection with counterexample finding  
✅ **Production Ready**: Clean, focused codebase with comprehensive testing  
✅ **spaCy Integration**: Real linguistic intelligence for natural language processing  

## Troubleshooting

### Port Already in Use
```bash
lsof -i :8000
kill <PID>
```

### Virtual Environment Issues
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### spaCy Model Missing
```bash
python -m spacy download en_core_web_sm
```

## Future Enhancements

1. **First-Order Logic**: Extend to predicate logic with quantifiers
2. **Advanced NLP**: Integration with transformer models
3. **Logical Reasoning**: Automated theorem proving
4. **Knowledge Graphs**: Integration with semantic networks
5. **Multi-language Support**: Support for multiple languages


This project is part of the ELMS LAB.


**ELMSLAB CPSC26-08** - Advancing True Reasoning Models