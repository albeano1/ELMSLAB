# LLM Integration Testing Guide

## 📊 Current Status

### ✅ What's Working
- API server is running successfully on `http://localhost:8002`
- Claude integration is initialized (shows "✅ Claude integration initialized successfully")
- Environment variables are loading correctly from `.env` file
- LLM status endpoint works: `claude_available": true`
- Basic logic reasoning system is working

### ⚠️ Issue Found
The Claude model name `claude-3-5-sonnet-20241022` is not valid and returns a 404 error.

### 🔧 Fix Required
Update the model name in `claude_integration.py` to a valid Claude model. Common valid models include:
- `claude-3-5-sonnet-20241022`
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

## 🧪 Testing Steps

### 1. **Test Basic Logic (Without LLM)**
```bash
cd /Users/seannickerson/ELMSLAB
source venv/bin/activate

# Test basic inference
curl -X POST "http://localhost:8002/infer" \
     -H "Content-Type: application/json" \
     -d '{
       "premises": ["All birds can fly", "Tweety is a bird"],
       "conclusion": "Can Tweety fly?",
       "logic_type": "auto"
     }' | python3 -m json.tool
```

**Expected:** Valid logic response with reasoning steps

### 2. **Test LLM Status**
```bash
curl -s http://localhost:8002/llm/status | python3 -m json.tool
```

**Expected:**
```json
{
    "claude_available": true,
    "vectionary_available": true,
    "knowledge_base_available": true
}
```

### 3. **Fix Model Name and Test LLM**

**Option A: Update model name manually**
```bash
# Edit claude_integration.py, line 69
# Change model name to a valid one
nano claude_integration.py
```

**Option B: Set model via environment variable**
```bash
# Edit .env file
echo "CLAUDE_MODEL=claude-3-5-sonnet-20241022" >> .env
```

**Then restart the server:**
```bash
pkill -f vectionary_98_api
source venv/bin/activate
python3 vectionary_98_api.py &
```

### 4. **Test LLM Reasoning**
```bash
curl -X POST "http://localhost:8002/llm/reason" \
     -H "Content-Type: application/json" \
     -d '{
       "premises": ["All humans are mortal", "Socrates is human"],
       "conclusion": "Socrates is mortal",
       "use_knowledge_base": true
     }' | python3 -m json.tool
```

**Expected:** Claude's reasoning response with validation

### 5. **Test LLM Validation**
```bash
curl -X POST "http://localhost:8002/llm/validate" \
     -H "Content-Type: application/json" \
     -d '{
       "llm_response": "Yes, Socrates is mortal because all humans are mortal and Socrates is human.",
       "premises": ["All humans are mortal", "Socrates is human"],
       "conclusion": "Socrates is mortal"
     }' | python3 -m json.tool
```

**Expected:** Validation result with truth table and consistency analysis

### 6. **Test LLM Conversion**
```bash
curl -X POST "http://localhost:8002/llm/convert" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "All birds can fly",
       "target_logic_type": "auto"
     }' | python3 -m json.tool
```

**Expected:** Both Vectionary and Claude conversion results with comparison

### 7. **Run Comprehensive Test Suite**
```bash
source venv/bin/activate
python3 test_llm_integration.py
```

**Expected:** Test results with success rate report

## 🌐 Web Interface Testing

### Open the Enhanced Web Interface
```bash
open logic_ui_llm_enhanced.html
# Or manually open in your browser
```

### Test Features:
1. **Basic Logic Tab** - Test traditional logic reasoning
2. **LLM Integration Tab** - Test Claude reasoning and validation
3. **Advanced Tab** - Test truth tables and advanced analysis

## 📝 Current Test Results

### ✅ Passed Tests:
- Environment setup
- API server startup
- Virtual environment activation
- Dependency installation
- .env file loading
- Vectionary parsing
- Knowledge base operations
- Basic logic reasoning
- LLM status endpoint

### ❌ Failed Tests:
- Claude API calls (model name issue)
- LLM reasoning endpoint
- LLM validation endpoint
- LLM conversion endpoint

## 🔧 Quick Fixes

### If Claude API fails:
1. Check API key: `grep ANTHROPIC_API_KEY .env`
2. Update model name in `claude_integration.py` or `.env`
3. Restart server: `pkill -f vectionary_98_api && python3 vectionary_98_api.py &`

### If server won't start:
1. Kill existing instances: `pkill -f vectionary_98_api`
2. Check port availability: `lsof -i :8002`
3. Check logs: `tail -f server.log`

### If dependencies are missing:
```bash
source venv/bin/activate
pip install -r requirements.txt
pip install aiohttp httpx pytest python-dotenv
```

## 📊 Success Metrics

- ✅ Basic logic reasoning: **Working**
- ✅ Vectionary parsing: **Working**
- ✅ Knowledge base: **Working**
- ✅ API endpoints: **Working**
- ⚠️ Claude integration: **Needs model name fix**
- ⚠️ LLM reasoning: **Pending Claude fix**
- ⚠️ LLM validation: **Pending Claude fix**

## 🎯 Next Steps

1. **Fix model name** in `claude_integration.py` or set via `.env`
2. **Restart server** to apply changes
3. **Run tests** to verify everything works
4. **Use web interface** for interactive testing

---

**Note:** Once the model name is fixed, all LLM integration features should work correctly. The infrastructure is in place and working - just needs the correct Claude model identifier.
