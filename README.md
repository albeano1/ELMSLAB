# ELMS - Enhanced Logic Modeling System


### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt


### CLI Usage

```bash
# Basic example
python3 ELMS.py "Bob walks. Does Bob walk?" --env prod

# Complex reasoning
python3 ELMS.py "A lighthouse shines whenever night falls. Tonight, the sky is full of stars. Is the lighthouse shining?" --env prod

# JSON output
python3 ELMS.py "Jack gave Jill a book. Does Jill have the book?" --env prod --json

# Query knowledge base (no premises required)
python3 ELMS.py "Is John a doctor?" --env prod
```

### Web UI

```bash
# Start the API server
source venv/bin/activate
python3 serv_vectionary.py &

# Open browser to webdemo.html
open webdemo.html
```

## Architecture


```
User Input
    ↓
Vectionary API → Semantic Trees (lemmas, roles, marks, definitions)
    ↓
Formula Generation → predicate(args) from tree data
    ↓
Tree-Based Reasoning → Match using tree lemmas, roles, marks
    ↓
Formal Theorem → With pronoun resolution & temporal markers
```

### Core Files

**CLI (Independent):**
- `ELMS.py` - **Fully independent CLI** with built-in parsing, reasoning, and API client
  - Contains complete Vectionary integration
  - No dependencies on other project files
  - Can be used standalone

**Web (Shared Backend):**
- `vectionary_98_api.py` - FastAPI web server
- `vectionary_98_percent_solution.py` - Backend reasoning engine (used by web API)
- `logic_ui_final.html` - Web user interface

**Knowledge Base:**
- `vectionary_knowledge_base.py` - Knowledge base management
- `vectionary_knowledge_base.json` - Knowledge base data

**Reference:**
- `vectionary.py` - API reference implementation

**Testing:**
- `tests/` - Test suite
- `test_edge_cases.sh` - Edge case testing script

## Examples

### Simple Direct Matching
```bash
Input: "Bob walks. Does Bob walk?"
Output: Valid (tree lemma matching)
Theorem: walk(Bob) → walk(Bob)
```

### Universal Quantifiers
```bash
Input: "All birds can fly. Tweety is a bird. Can Tweety fly?"
Output: Valid (universal instantiation)
Theorem: ∀x(bird(x) → fly(x)) ∧ bird(Tweety) → fly(Tweety)
```

### Conditional Reasoning
```bash
Input: "A lighthouse shines whenever night falls. Tonight, the sky is full of stars. Is the lighthouse shining?"
Output: Valid (conditional reasoning with temporal markers)
Theorem: ∀x(fall(night) → shine(lighthouse)) ∧ be(sky, stars) → shine(lighthouse)
```

### Temporal & Pronouns
```bash
Input: "John opened the door. Then he entered the room. Did John enter the room?"
Output: Valid (pronoun resolution + temporal markers)
Theorem: (open(John, door) ∧ [then] enter(John, room)) → enter(John, room)
```

## API Environments

- `--env prod` - Production Vectionary API (default)
- `--env dev` - Development API endpoint
- `--env local` - Local Vectionary server

## Reasoning Strategies

The system uses **3 tree-based strategies** (no hardcoding):

1. **Universal Reasoning** - Matches tree lemmas and semantic roles
2. **Conditional Reasoning** - Uses tree marks for temporal/conditional logic
3. **Direct Matching** - Compares premise and conclusion trees


**Zero text pattern matching** (`if 'word' in text`)

## Web Features

- **Result Caching** - Instant mode switching between analysis types
- **Formal Theorems** - Displays actual theorems, not placeholders
- **Parse Trees** - Shows full Vectionary semantic trees
- **Semantic Analysis** - Displays definitions and roles
- **Claude Integration** - Compare with Claude's reasoning (if configured)



## Requirements

- Python 3.8+ (3.11 or 3.12 recommended for best compatibility)
- Core: `requests`, `python-dotenv`
- Optional: `fastapi`, `uvicorn`, `pytest` (for web server and testing)
- See `requirements.txt` for full list

## Configuration

Create a `.env.local` file (optional):
```bash

# Edit .env.local and add your actual API keys
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

## Troubleshooting

### API Rate Limiting
If you encounter rate limits:
- Wait 1-5 minutes between requests
- Use a local Vectionary server (`--env local`)
- Use simpler sentences

## Development

### Project Structure
```
ELMSLAB/
├── ELMS.py                              # Standalone CLI (fully independent)
├── vectionary_98_api.py                 # Web API server
├── vectionary_98_percent_solution.py    # Backend for web (not used by CLI)
├── logic_ui_final.html                  # Web UI
├── vectionary_knowledge_base.py         # Knowledge base
├── vectionary_knowledge_base.json       # KB data
├── vectionary.py                        # API reference
├── requirements.txt                     # Dependencies
├── env.example                          # Config template
├── tests/                               # Test suite
│   ├── __init__.py
│   ├── test_edge_cases.py
│   └── README.md
└── venv/                                # Virtual environment
```

## License



## Credits

---


