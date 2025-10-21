<div align="center">

![ELMS Logo](ELMS.svg)

# ELMS - Enhanced Logic Modeling System

A hybrid logical reasoning system powered by Vectionary semantic parsing and Prolog inference.

</div>


### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp env.example .env.local
# Edit .env.local and add your actual API keys

### CLI Usage

```bash
# Basic example
python3 ELMS.py "Bob walks. Does Bob walk?" --env prod

# Open-ended question (draw conclusions)
python3 ELMS.py "Maria is a student. John is a student. Maria studies regularly. Who are students?" --env prod

# Debug mode (see behind-the-scenes)
python3 ELMS.py "All cats are mammals. Fluffy is a cat. What mammals do we have?" --env prod --debug

# JSON output
python3 ELMS.py "Jack gave Jill a book. Does Jill have the book?" --env prod --json
```

### Web Demo

```bash
# Start the API server
source venv/bin/activate
uvicorn serv_vectionary:app --host 0.0.0.0 --port 8002 --reload &

# Start the web server
python3 -m http.server 8000 &

# Open browser
open http://localhost:8000/webdemo.html
```



## Architecture

```
User Input
    ↓
Vectionary API → Semantic Trees (lemmas, roles, marks, definitions)
    ↓
Dynamic Conversion → Convert NL to Prolog facts and rules
    ↓
Prolog Inference → Query Prolog knowledge base
    ↓
Results → Formatted answer with explanation
```

### Core Files

**CLI:**
- `ELMS.py` - Main CLI with Vectionary parsing and Prolog inference
  - Dynamic NL to Prolog conversion
  - Open-ended question detection
  - Debug mode with tree visualization

**Hybrid Reasoning:**
- `hybrid_reasoner.py` - Prolog + Vectionary integration
- `prolog_reasoner.py` - Prolog inference engine (pytholog wrapper)

**Web Demo:**
- `serv_vectionary.py` - FastAPI server for web demo
- `webdemo.html` - Interactive web interface

**Integration:**
- `claude_integration.py` - Claude API integration (optional)
- `vectionary_knowledge_base.py` - Knowledge base management
- `vectionary_knowledge_base.json` - Knowledge base data

**Reference:**
- `vectionaryref.py` - Vectionary API reference implementation

## Examples

### Open-ended Questions
```bash
Input: "Maria is a student. John is a student. Maria studies regularly. Who are students?"
Output: John and Maria
Prolog: student(X)
```

### Universal Quantification
```bash
Input: "All cats are mammals. Fluffy is a cat. Whiskers is a cat. What mammals do we have?"
Output: Fluffy, Whiskers
Prolog: mammal(X) :- cat(X), cat(fluffy), cat(whiskers)
```

### Compound Predicates
```bash
Input: "Alice teaches mathematics. Bob teaches science. Alice has many students. Who are teachers with many students?"
Output: Alice
Prolog: teacher(X), has_many_students(X)
```

## API Environments

- `--env prod` - Production Vectionary API (default)
- `--env dev` - Development API endpoint
- `--env local` - Local Vectionary server

## Dynamic Conversion

The system uses **Vectionary semantic parsing** to dynamically convert natural language to Prolog:

1. **Premises** → Prolog facts and rules
   - "X is a Y" → `y(x)`
   - "X is Y of Z" → `y(x, z)`
   - "All X are Y" → `y(Z) :- x(Z)`
   - "X does Y" → `do_y(x)`
   - "X does Y Z-ly" → `do_y_z(x)`

2. **Queries** → Prolog queries
   - "Who are X?" → `x(X)`
   - "Who are X who Y?" → `x(X), y(X)`
   - "What X do we have?" → `x(X)`

3. **Open-ended Detection** → Uses POS tags and dependency labels
   - Question pronouns (who, what, which)
   - Relative clauses
   - No hardcoded word lists

## Debug Mode

Use `--debug` flag to see all behind-the-scenes steps:

```bash
python3 ELMS.py "Maria is a student. Who are students?" --env prod --debug
```




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
├── ELMS.py                              # Main CLI
├── hybrid_reasoner.py                   # Prolog + Vectionary integration
├── prolog_reasoner.py                   # Prolog inference engine
├── serv_vectionary.py                   # FastAPI server
├── webdemo.html                         # Web UI
├── claude_integration.py                # Claude API integration
├── vectionary_knowledge_base.py         # Knowledge base
├── vectionary_knowledge_base.json       # KB data
├── vectionaryref.py                     # Vectionary API reference
├── requirements.txt                     # Dependencies
├── env.example                          # Config template
├── .env.local                           # Local config (gitignored)
├── tests/                               # Test suite
│   └── test_edge_cases.py
└── venv/                                # Virtual environment
```

## License



## Credits

---


