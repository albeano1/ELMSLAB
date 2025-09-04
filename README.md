# Enhanced Natural Language to Logic API with First-Order Logic Support

A production-ready API that converts natural English into both propositional and first-order logic statements with advanced reasoning capabilities, built as part of the ELMSLAB project for advancing true reasoning models.

## Project Overview

This project addresses the challenge of converting natural language text into formal logic formulas and performing automated logical reasoning. Unlike current "reasoning models" that merely mimic reasoning through text generation, this system provides a foundation for true automated reasoning using human-readable language.

## Features

### 🧠 **Dual Logic Support**
- **Propositional Logic**: Complete support for AND (∧), OR (∨), NOT (¬), IMPLIES (→), and IFF (↔)
- **First-Order Logic**: Universal (∀) and existential (∃) quantifiers with predicates and individual constants
- **Automatic Detection**: Smart logic type detection based on natural language patterns

### 🔍 **Advanced Natural Language Processing**
- **spaCy Integration**: Real linguistic intelligence with dependency parsing
- **Proper Negation Handling**: Correctly processes negation using spaCy's `neg` dependency
- **Quantifier Recognition**: Detects "All", "Some", "Every", "There exists" patterns
- **Individual Constants**: Handles proper names like "Socrates", "John", etc.

### ⚡ **Powerful Inference Engine**
- **Propositional Inference**: 100% accurate truth table-based reasoning
- **First-Order Detection**: Identifies when advanced theorem proving is needed
- **Pattern Recognition**: Detects common inference patterns (Modus Ponens, Modus Tollens, etc.)
- **Fallacy Detection**: Identifies invalid arguments with counterexamples
- **Explanation Generation**: Human-readable explanations of reasoning results

### 🌐 **User-Friendly Interface**
- **Interactive Web UI**: Modern HTML interface with example buttons
- **REST API**: Full REST API with comprehensive endpoints
- **CORS Support**: Works seamlessly in web browsers
- **Real-time Testing**: Instant feedback on logical reasoning

## Current Project Structure

```
ELMSLAB/
├── enhanced_fol_api.py          # 🚀 MAIN API: Enhanced First-Order Logic API
├── logic_ui.html                # 🌐 Interactive web interface
├── src/                         # Source code modules
│   ├── models/                  # Logic data structures
│   │   ├── propositional_logic.py    # Propositional logic models
│   │   └── first_order_logic.py      # First-order logic models
│   ├── core/                    # Core processing modules
│   │   └── first_order_parser.py     # First-order logic parser
│   └── utils/                   # Utility functions
│       └── formula_utils.py
├── requirements.txt             # Dependencies
├── README.md                   # This file
├── QUICK_START.md              # Quick start guide
└── venv/                       # Virtual environment
```

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- Virtual environment (already set up in this project)

### 1. Setup (One-time)
```bash
# Navigate to the project directory
cd /path/to/ELMSLAB

# Activate the virtual environment
source venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt

# Install spaCy language model (if needed)
python -m spacy download en_core_web_sm
```

### 2. Start the Enhanced API Server
```bash
# Make sure you're in the project directory and venv is activated
source venv/bin/activate
python enhanced_fol_api.py
```

The API will start at `http://127.0.0.1:8000`

### 3. Test the System

#### Option A: Use the Web Interface (Recommended)
1. Open `logic_ui.html` in your browser
2. Try the example buttons:
   - **Socrates Example**: Classic first-order logic
   - **Business Logic**: Complex propositional reasoning
   - **Birds Example**: Universal quantifiers
   - **Propositional Logic**: Simple conditionals

#### Option B: Test with curl commands

**Test Propositional Logic:**
```bash
curl -X POST "http://127.0.0.1:8000/infer" \
     -H "Content-Type: application/json" \
     -d '{
       "premises": ["If it rains then the ground is wet", "It is raining"],
       "conclusion": "The ground is wet",
       "logic_type": "propositional"
     }' | python3 -m json.tool
```

**Test First-Order Logic:**
```bash
curl -X POST "http://127.0.0.1:8000/infer" \
     -H "Content-Type: application/json" \
     -d '{
       "premises": ["All humans are mortal", "Socrates is human"],
       "conclusion": "Socrates is mortal",
       "logic_type": "auto"
     }' | python3 -m json.tool
```

**Test Natural Language Conversion:**
```bash
curl -X POST "http://127.0.0.1:8000/convert" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "All humans are mortal",
       "logic_type": "auto"
     }' | python3 -m json.tool
```

### 4. View API Documentation
Visit `http://127.0.0.1:8000/docs` for interactive API documentation.
## 📚 API Endpoints

### Convert Text to Logic
```http
POST /convert
Content-Type: application/json

{
  "text": "All humans are mortal",
  "logic_type": "auto",
  "include_truth_table": true
}
```

**Response (First-Order Logic)**:
```json
{
  "original_text": "All humans are mortal",
  "first_order_formula": "∀h((humans(h) → mortal(h)))",
  "formula_type": "QuantifiedFormula",
  "confidence": 0.8,
  "variables": ["h"],
  "constants": [],
  "predicates": ["humans(h)", "mortal(h)"],
  "logic_type": "first_order",
  "detected_logic_type": "first_order"
}
```

**Response (Propositional Logic)**:
```json
{
  "original_text": "It is not raining and it is cold",
  "propositional_formula": "(¬it_rain ∧ cold)",
  "confidence": 0.8,
  "atoms": ["cold", "it_rain"],
  "logic_type": "propositional",
  "detected_logic_type": "propositional"
}
```

### Logical Inference
```http
POST /infer
Content-Type: application/json

{
  "premises": [
    "If it rains then the ground is wet",
    "It is raining"
  ],
  "conclusion": "The ground is wet",
  "logic_type": "propositional"
}
```

**Response (Valid Propositional Inference)**:
```json
{
  "valid": true,
  "premises": ["(it_rain → ground_wet)", "it_rain"],
  "conclusion": "ground_wet",
  "implication": "((it_rain → ground_wet) ∧ it_rain) → ground_wet",
  "logic_type": "propositional",
  "explanation": "✓ Valid inference: The conclusion logically follows from the premises.",
  "counterexample": null,
  "truth_table_summary": {
    "is_tautology": true,
    "is_contradiction": false,
    "is_satisfiable": true
  }
}
```

**Response (First-Order Logic)**:
```json
{
  "valid": "unknown",
  "premises": ["∀h((humans(h) → mortal(h)))", "human(Socrates)"],
  "conclusion": "mortal(Socrates)",
  "logic_type": "first_order",
  "explanation": "First-order logic inference requires sophisticated theorem proving. This is a simplified representation.",
  "note": "Full first-order inference engine not yet implemented"
}
```



## 💡 Working Examples

### First-Order Logic Examples
| Natural Language | First-Order Logic | Type |
|------------------|-------------------|------|
| "All humans are mortal" | `∀h((humans(h) → mortal(h)))` | Universal Quantifier |
| "Some birds cannot fly" | `∃b((birds(b) ∧ ¬fly(b)))` | Existential Quantifier |
| "Socrates is human" | `human(Socrates)` | Individual Statement |
| "Every student studies" | `∀s((student(s) → studies(s)))` | Universal Quantifier |

### Propositional Logic Examples
| Natural Language | Propositional Logic | Type |
|------------------|-------------------|------|
| "It is not raining" | `¬it_rain` | Negation |
| "It is raining and the ground is wet" | `(it_rain ∧ ground_wet)` | Conjunction |
| "Either it is sunny or it is cloudy" | `(it_sunny ∨ it_cloudy)` | Disjunction |
| "If it rains, then the ground will be wet" | `(it_rain → ground_wet)` | Implication |

### Inference Examples
| Premises | Conclusion | Valid? | Logic Type |
|----------|------------|--------|------------|
| "If it rains then the ground is wet", "It is raining" | "The ground is wet" | ✅ Valid | Propositional |
| "All humans are mortal", "Socrates is human" | "Socrates is mortal" | 🔍 First-order | First-Order |
| "If it rains then the ground is wet", "The ground is wet" | "It is raining" | ❌ Invalid | Propositional |

## 🧪 Testing the System

### Quick Test (Recommended)
1. **Start the API**: `python enhanced_fol_api.py`
2. **Open the Web Interface**: Open `logic_ui.html` in your browser
3. **Try the examples**: Click the example buttons to test different scenarios

### Manual Testing with curl

**Test Propositional Logic (Modus Ponens):**
```bash
curl -X POST "http://127.0.0.1:8000/infer" \
     -H "Content-Type: application/json" \
     -d '{
       "premises": ["If it rains then the ground is wet", "It is raining"],
       "conclusion": "The ground is wet",
       "logic_type": "propositional"
     }'
```
**Expected**: `"valid": true`

**Test First-Order Logic:**
```bash
curl -X POST "http://127.0.0.1:8000/infer" \
     -H "Content-Type: application/json" \
     -d '{
       "premises": ["All humans are mortal", "Socrates is human"],
       "conclusion": "Socrates is mortal",
       "logic_type": "auto"
     }'
```
**Expected**: `"valid": "unknown"` (first-order logic detected)

**Test Natural Language Conversion:**
```bash
curl -X POST "http://127.0.0.1:8000/convert" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Some birds cannot fly",
       "logic_type": "auto"
     }'
```
**Expected**: `"first_order_formula": "∃b((birds(b) ∧ ¬fly(b)))"`

## 🔧 Technical Details

### Dual Logic Architecture
- **Propositional Logic**: Complete support for all logical connectives (¬, ∧, ∨, →, ↔)
- **First-Order Logic**: Universal (∀) and existential (∃) quantifiers with predicates
- **Smart Detection**: Automatic logic type detection based on natural language patterns
- **Formula Parsing**: Robust parser with proper operator precedence and parentheses handling

### Advanced Natural Language Processing
- **spaCy Integration**: Real linguistic intelligence with dependency parsing
- **Negation Detection**: Uses spaCy's `neg` dependency for accurate negation handling
- **Quantifier Recognition**: Detects "All", "Some", "Every", "There exists" patterns
- **Individual Constants**: Handles proper names and specific entities

### Inference Engine
- **Propositional Inference**: 100% accurate truth table-based reasoning
- **First-Order Detection**: Identifies when advanced theorem proving is needed
- **Pattern Recognition**: Detects common inference patterns (Modus Ponens, Modus Tollens, etc.)
- **Counterexample Generation**: Provides scenarios where invalid inferences fail

### API Features
- **FastAPI**: Modern, fast web framework with automatic documentation
- **CORS Support**: Works seamlessly in web browsers
- **Pydantic**: Data validation and serialization
- **Error Handling**: Comprehensive error management with helpful messages

## 🎯 Current Capabilities

### ✅ **Fully Working**
- **Propositional Logic Conversion**: Natural language → propositional formulas
- **Propositional Inference**: 100% accurate reasoning with truth tables
- **First-Order Logic Conversion**: Natural language → first-order formulas
- **Logic Type Detection**: Automatic detection of propositional vs first-order
- **Web Interface**: Interactive HTML interface with example buttons
- **CORS Support**: Browser-compatible API requests
- **Negation Handling**: Proper negation parsing and representation

### 🔍 **Partially Working**
- **First-Order Inference**: Detects first-order logic but requires advanced theorem proving
- **Complex Quantifiers**: Basic universal and existential quantifier support

### 🚧 **Future Enhancements**
- **Full First-Order Theorem Proving**: Complete inference engine for quantified logic
- **Advanced Pattern Recognition**: More sophisticated inference pattern detection
- **Multi-language Support**: Support for multiple natural languages

## 🛠️ Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
lsof -i :8000
kill <PID>
```

**Virtual Environment Issues:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**spaCy Model Missing:**
```bash
python -m spacy download en_core_web_sm
```

**API Not Responding:**
1. Check if the server is running: `ps aux | grep python`
2. Restart the API: `pkill -f "python.*api" && python enhanced_fol_api.py`

**Web Interface "Load Failed" Error:**
- Make sure the API server is running on `http://127.0.0.1:8000`
- Check browser console for CORS errors
- Try refreshing the page (Cmd+R or F5)

## 🎯 Quick Reference

### Start the System
```bash
cd /path/to/ELMSLAB
source venv/bin/activate
python enhanced_fol_api.py
```

### Test Propositional Logic
```bash
curl -X POST "http://127.0.0.1:8000/infer" \
     -H "Content-Type: application/json" \
     -d '{"premises": ["If it rains then the ground is wet", "It is raining"], "conclusion": "The ground is wet", "logic_type": "propositional"}'
```

### Test First-Order Logic
```bash
curl -X POST "http://127.0.0.1:8000/infer" \
     -H "Content-Type: application/json" \
     -d '{"premises": ["All humans are mortal", "Socrates is human"], "conclusion": "Socrates is mortal", "logic_type": "auto"}'
```

### Use Web Interface
Open `logic_ui.html` in your browser and click the example buttons.

---

## 🚀 **Ready to Use!**

This enhanced natural language to logic API is now ready for use with:
- ✅ **Working propositional logic inference**
- ✅ **First-order logic conversion and detection**
- ✅ **Interactive web interface**
- ✅ **CORS support for browser compatibility**
- ✅ **Comprehensive error handling**

**Start the system and begin testing logical reasoning with natural language!**

---

*This project is part of the ELMS LAB - Advancing True Reasoning Models*