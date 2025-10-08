# LLM Integration Guide for Enhanced Logic Reasoning System

This guide explains how to integrate and use the Claude API with your Enhanced Logic Modeling System (ELMS) for advanced logical reasoning and validation.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with virtual environment
2. **Anthropic API Key** for Claude integration
3. **All existing ELMS dependencies** (see requirements.txt)

### 1. Setup Claude API Key

```bash
# Copy the example environment file
cp env.example .env

# Edit .env file and add your actual API key
# The .env file is already in .gitignore for security
nano .env  # or use your preferred editor
```

**Or set as environment variable:**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 2. Install Additional Dependencies

```bash
# Activate your virtual environment
source venv/bin/activate

# Install the new dependency
pip install aiohttp==3.9.1

# Or reinstall all requirements
pip install -r requirements.txt
```

### 3. Start the Enhanced API Server

```bash
# Start the server with LLM integration
python vectionary_98_api.py
```

The server will start at `http://localhost:8002` with LLM integration enabled.

## 🧠 Features Overview

### Core LLM Integration Features

1. **Claude API Integration** - Direct communication with Anthropic's Claude
2. **LLM Response Validation** - Validate LLM responses against formal logic
3. **Comparative Analysis** - Compare Vectionary vs Claude reasoning
4. **Truth Table Generation** - Generate truth tables for validation
5. **Knowledge Base Integration** - Use existing knowledge for context

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/llm/status` | GET | Check LLM integration status |
| `/llm/validate` | POST | Validate LLM response against logic |
| `/llm/reason` | POST | Get Claude's reasoning on premises/conclusion |
| `/llm/convert` | POST | Convert text using both Vectionary and Claude |
| `/llm/test` | POST | Test LLM integration with examples |

## 📖 Usage Examples

### 1. Basic LLM Reasoning

```bash
curl -X POST "http://localhost:8002/llm/reason" \
     -H "Content-Type: application/json" \
     -d '{
       "premises": ["All humans are mortal", "Socrates is human"],
       "conclusion": "Socrates is mortal",
       "use_knowledge_base": true
     }'
```

**Response:**
```json
{
  "success": true,
  "reasoning_result": {
    "claude_response": {
      "content": "Yes, Socrates is mortal. This follows from universal instantiation...",
      "confidence": 0.95,
      "reasoning_steps": ["1. All humans are mortal (universal rule)", "2. Socrates is human (instance)", "3. Therefore, Socrates is mortal (universal instantiation)"]
    },
    "validation_result": {
      "is_valid": true,
      "validation_confidence": 0.98,
      "reasoning": "✅ LLM response is VALID. Consistent with logical premises."
    }
  }
}
```

### 2. LLM Response Validation

```bash
curl -X POST "http://localhost:8002/llm/validate" \
     -H "Content-Type: application/json" \
     -d '{
       "llm_response": "Yes, the conclusion follows logically from the premises.",
       "premises": ["All birds can fly", "Tweety is a bird"],
       "conclusion": "Tweety can fly"
     }'
```

### 3. Comparative Text Conversion

```bash
curl -X POST "http://localhost:8002/llm/convert" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "All birds can fly",
       "target_logic_type": "auto"
     }'
```

## 🖥️ Web Interface

### Enhanced HTML Interface

Open `logic_ui_llm_enhanced.html` in your browser for a full-featured interface with:

- **Basic Logic Tab**: Traditional logic analysis
- **LLM Integration Tab**: Claude reasoning and validation
- **Advanced Tab**: Complex analysis and truth tables

### Key Features:

1. **Real-time LLM Status** - Shows if Claude is available
2. **Interactive Examples** - Pre-loaded examples for testing
3. **Validation Results** - Visual feedback on LLM response validity
4. **Truth Table Display** - Generated truth tables for validation
5. **Comparative Analysis** - Side-by-side Vectionary vs Claude results

## 🧪 Testing

### Run Comprehensive Tests

```bash
# Run the full test suite
python test_llm_integration.py
```

**Test Coverage:**
- Claude API connectivity
- LLM response validation
- Truth table generation
- Knowledge base integration
- API endpoint functionality
- Comprehensive workflow testing

### Test Results

The test suite will generate:
- Console output with pass/fail status
- Detailed JSON report: `llm_integration_test_results.json`
- Success rate calculation
- Component availability status

## 🔧 Configuration

### Environment Variables

**Using .env file (recommended):**
```bash
# Copy the example file
cp env.example .env

# Edit .env file with your actual values
ANTHROPIC_API_KEY=your-actual-api-key-here
CLAUDE_MODEL=claude-3-5-sonnet-20241022
CLAUDE_TIMEOUT=30
```

**Or set as environment variables:**
```bash
export ANTHROPIC_API_KEY=your-api-key-here
export CLAUDE_MODEL=claude-3-5-sonnet-20241022
export CLAUDE_TIMEOUT=30
```

**Note:** The `.env` file is automatically ignored by git for security.

### API Configuration

The system automatically detects Claude availability and gracefully degrades if the API key is not available.

## 📊 Workflow Examples

### Complete LLM Validation Workflow

```python
# Step 1: Parse with Vectionary
from vectionary_98_percent_solution import Vectionary98PercentSolution
vectionary = Vectionary98PercentSolution()
parsed = vectionary.parse_with_vectionary_98("All birds can fly")

# Step 2: Get Claude reasoning
from claude_integration import ClaudeIntegration
claude = ClaudeIntegration()
result = await claude.reason_with_claude(
    ["All birds can fly", "Tweety is a bird"],
    "Can Tweety fly?"
)

# Step 3: Validate Claude's response
validation = claude.validate_llm_response(
    result["claude_response"].content,
    ["All birds can fly", "Tweety is a bird"],
    "Can Tweety fly?"
)

# Step 4: Generate truth table
truth_table = validation["truth_table"]
```

### Integration with Existing System

```python
# Use with existing knowledge base
from vectionary_knowledge_base import VectionaryKnowledgeBase
kb = VectionaryKnowledgeBase()

# Add facts
kb.add_fact("All birds can fly", confidence=0.95)

# Query with LLM context
result = await claude.reason_with_claude(
    ["All birds can fly", "Tweety is a bird"],
    "Can Tweety fly?",
    use_knowledge_base=True
)
```

## 🚨 Troubleshooting

### Common Issues

1. **Claude API Key Not Set**
   ```
   Error: Anthropic API key is required
   Solution: Set ANTHROPIC_API_KEY environment variable
   ```

2. **API Connection Timeout**
   ```
   Error: Claude API error 408: Request timeout
   Solution: Check internet connection and API key validity
   ```

3. **Rate Limiting**
   ```
   Error: Claude API error 429: Rate limit exceeded
   Solution: Wait and retry, or upgrade API plan
   ```

4. **Invalid Response Format**
   ```
   Error: LLM validation failed: Invalid response format
   Solution: Check Claude response format and parsing logic
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

```bash
# Check API status
curl http://localhost:8002/llm/status

# Test LLM integration
curl -X POST http://localhost:8002/llm/test
```

## 📈 Performance Considerations

### Optimization Tips

1. **Batch Requests**: Group multiple validations together
2. **Cache Results**: Store frequently used logic conversions
3. **Async Processing**: Use async/await for concurrent operations
4. **Rate Limiting**: Implement client-side rate limiting

### Resource Usage

- **Memory**: ~50MB additional for Claude integration
- **Network**: ~1-5KB per API request
- **Processing**: ~100-500ms per validation (depending on complexity)

## 🔮 Future Enhancements

### Planned Features

1. **Multi-LLM Support**: Integration with GPT-4, Gemini, etc.
2. **Advanced Validation**: More sophisticated consistency checking
3. **Learning System**: Improve validation based on feedback
4. **Batch Processing**: Handle multiple validations simultaneously
5. **Custom Models**: Fine-tuned models for specific domains

### Contributing

To contribute to the LLM integration:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📚 API Reference

### ClaudeIntegration Class

```python
class ClaudeIntegration:
    def __init__(self, api_key: Optional[str] = None)
    async def query_claude(self, prompt: str, system_prompt: Optional[str] = None) -> ClaudeResponse
    def validate_llm_response(self, llm_response: str, premises: List[str], conclusion: str) -> Dict[str, Any]
    async def reason_with_claude(self, premises: List[str], conclusion: str, use_knowledge_base: bool = True) -> Dict[str, Any]
```

### Request/Response Models

```python
class LLMValidateRequest(BaseModel):
    llm_response: str
    premises: List[str]
    conclusion: str

class LLMReasonRequest(BaseModel):
    premises: List[str]
    conclusion: str
    use_knowledge_base: bool = True
```

## 🎯 Use Cases

### 1. Educational Applications
- Validate student logic reasoning
- Provide detailed explanations
- Generate practice problems

### 2. Research Applications
- Compare different reasoning approaches
- Validate research hypotheses
- Generate formal logic from natural language

### 3. Business Applications
- Validate business logic
- Check policy consistency
- Generate formal specifications

### 4. Development Applications
- Validate code logic
- Generate test cases
- Check specification compliance

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Run the test suite to identify problems
3. Check API logs for detailed error messages
4. Verify Claude API key and permissions

---

**Ready to enhance your logic reasoning with LLM integration!** 🚀

Start with the quick start guide and explore the web interface to see the full capabilities of your enhanced system.
