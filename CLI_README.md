# ELMS Command Line Interface

The Enhanced Logic Modeling System (ELMS) provides powerful command-line tools for natural language logical reasoning with Vectionary integration.

## System Architecture

- **Main Deliverable**: `ELMS.py` - Standalone command-line interface (completely self-contained)
- **Demo Interface**: HTML webpage at `http://localhost:8002` - Calls the API for demonstration purposes
- **API Server**: `vectionary_98_api.py` - Powers the web interface and provides API endpoints

## Quick Start

```bash
# Simple usage with the wrapper script
./elms "Jack gave Jill a book. Then they walked home together. Does Jill feel grateful?" --env prod --json

# Or use the full path
python ELMS.py "All birds can fly. Tweety is a bird. Can Tweety fly?" --env prod --json
```

## Available Scripts

### 1. `./elms` (Recommended)
Simple wrapper script that uses the standalone system:
- **Standalone Mode**: Uses `ELMS.py` - completely self-contained
- No external dependencies or API servers required
- Full logical reasoning capabilities

### 2. `ELMS.py`
Standalone version that works completely offline:
- Pure Python implementation with no external dependencies
- Built-in semantic parsing and logical reasoning
- Full feature parity with web version
- Rich semantic analysis and Vectionary-style parse trees

### 3. Web Interface (Demo)
Interactive web interface for demonstration:
- **Access**: `http://localhost:8002` (when API server is running)
- **Purpose**: Demo the system capabilities with a user-friendly interface
- **Backend**: Calls the API server (`vectionary_98_api.py`) for processing
- **Features**: Interactive premise/conclusion input, knowledge base management, truth tables

## Starting the Web Demo

To run the web interface for demonstration:

```bash
# Start the API server
cd /Users/seannickerson/ELMSLAB
source venv/bin/activate
python vectionary_98_api.py

# Access the web interface at:
# http://localhost:8002
```

The web interface provides:
- Interactive premise/conclusion input
- Knowledge base management
- Truth table visualization
- Temporal logic demonstrations
- Preset scenarios for testing

## Usage Examples

### Basic Usage
```bash
# JSON output (recommended for programmatic use)
./elms "Jack gave Jill a book. Then they walked home together. Does Jill feel grateful?" --env prod --json

# Human-readable output
./elms "All birds can fly. Tweety is a bird. Can Tweety fly?" --env prod

# Verbose output with detailed reasoning
./elms "The family shared a meal. Everyone who shares meals feels connected. Does the family feel connected?" --env prod --verbose
```

### Advanced Usage
```bash
# Custom API URL
python ELMS_API.py "Your text here" --api-url http://localhost:8002 --json

# Different environments
./elms "Your text here" --env dev --json
./elms "Your text here" --env test --json
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--env` | Environment setting (prod/dev/test) | `prod` |
| `--json` | Output results in JSON format | Human-readable |
| `--verbose` | Enable detailed reasoning steps | Basic output |
| `--api-url` | API server URL (ELMS_API.py only) | `http://localhost:8002` |

## Input Format

The system automatically splits your input into premises and conclusion:

**Premises**: Statements that provide information
**Conclusion**: Question that asks about the logical consequence

### Examples:
```bash
# Universal instantiation
"All birds can fly. Tweety is a bird. Can Tweety fly?"

# Temporal reasoning
"John opened the door. Then he entered the room. Did John enter the room?"

# Gift-gratitude reasoning
"Jack gave Jill a book. People who receive gifts feel grateful. Does Jill feel grateful?"

# Family meal sharing
"The family gathered around the table. The family shared a meal. Everyone who shares meals together feels connected. Does the family feel connected?"
```

## Output Formats

### JSON Output (`--json`)
```json
{
  "input": {
    "text": "Your input text",
    "environment": "prod"
  },
  "analysis": {
    "valid": true,
    "confidence": "HIGH CONFIDENCE",
    "explanation": "Detailed explanation...",
    "reasoning_steps": ["Step 1", "Step 2", ...],
    "parsed_premises": ["premise1", "premise2", ...],
    "parsed_conclusion": "conclusion",
    "vectionary_enhanced": true
  }
}
```

### Human-Readable Output
```
================================================================================
ELMS - Enhanced Logic Modeling System
================================================================================

📝 Input Text:
   Your input text here

🎯 Analysis Result:
   Valid: Yes
   Confidence: HIGH CONFIDENCE

📋 Explanation:
   Detailed explanation with Vectionary parse trees and semantic analysis
```

## Exit Codes

- `0`: Valid conclusion (success)
- `1`: Invalid conclusion or error
- `130`: Operation cancelled by user (Ctrl+C)

## Requirements

- Python 3.7+
- Virtual environment activated (automatically handled by `./elms`)
- No external dependencies or services required

## Troubleshooting

### No External Dependencies Required
The standalone system requires no external services:
- No API servers needed
- No network connections required
- Completely self-contained

### Virtual Environment Issues
Make sure the virtual environment is activated:
```bash
source venv/bin/activate
```

### Permission Issues
Make sure the scripts are executable:
```bash
chmod +x elms ELMS.py ELMS_API.py
```

## Advanced Features

### Rich Vectionary Integration
When using API mode, the system provides:
- **Semantic Roles**: agent, beneficiary, patient relationships
- **Word Definitions**: Complete definitions for disambiguation
- **Temporal Markers**: "then", "after", "before" with semantic analysis
- **Spatial Markers**: "home", "together", "away" with context
- **Linguistic Features**: tense, mood, number, person awareness

### Comprehensive Edge Case Prevention
The system handles:
- Universal instantiation patterns
- Temporal-spatial reasoning
- Gift-gratitude scenarios
- Family meal sharing
- Pronoun resolution
- Medical treatment reasoning
- And many more complex scenarios

## Examples by Category

### Universal Instantiation
```bash
./elms "All birds can fly. Tweety is a bird. Can Tweety fly?" --json
./elms "All cars have engines. This car is a car. Does this car have an engine?" --json
./elms "All rainy days are wet. Today is a rainy day. Is today wet?" --json
```

### Temporal Reasoning
```bash
./elms "John opened the door. Then he entered the room. Did John enter the room?" --json
./elms "Sarah finished her homework before she watched a movie. Then she went to bed. Did Sarah go to bed after finishing her homework and watching a movie?" --json
```

### Social Reasoning
```bash
./elms "Jack gave Jill a book. People who receive gifts feel grateful. Does Jill feel grateful?" --json
./elms "Alice and Bob are friends. Alice told Bob a secret. All people who share secrets are close. Are Alice and Bob close?" --json
```

### Family Scenarios
```bash
./elms "The family gathered around the table. The family shared a meal. Everyone who shares meals together feels connected. Does the family feel connected?" --json
```

This command-line interface makes ELMS accessible for both interactive use and integration into larger systems!
