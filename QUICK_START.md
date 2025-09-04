# 🚀 **Quick Start Guide - Natural Language to Propositional Logic API**

## 📋 **Prerequisites**
- Python 3.8+ installed
- Terminal/Command line access

## ⚡ **Quick Setup (2 minutes)**

### **1. Activate Virtual Environment**
```bash
source venv/bin/activate
```
*You should see `(venv)` in your terminal prompt*

### **2. Start the API Server**
```bash
python fixed_api.py
```
*Server will start on http://127.0.0.1:8000*

### **3. Test the API**
Open a new terminal and run:
```bash
curl -X POST "http://127.0.0.1:8000/convert" \
     -H "Content-Type: application/json" \
     -d '{"text": "It is not raining", "include_truth_table": true}' | python3 -m json.tool
```

## 🧪 **Testing Examples**

### **Test Negation**
```bash
curl -X POST "http://127.0.0.1:8000/convert" \
     -H "Content-Type: application/json" \
     -d '{"text": "The dog is not barking", "include_truth_table": true}' | python3 -m json.tool
```

### **Test Conjunction**
```bash
curl -X POST "http://127.0.0.1:8000/convert" \
     -H "Content-Type: application/json" \
     -d '{"text": "It is raining and it is cold", "include_truth_table": true}' | python3 -m json.tool
```

### **Test Disjunction**
```bash
curl -X POST "http://127.0.0.1:8000/convert" \
     -H "Content-Type: application/json" \
     -d '{"text": "Either it is sunny or it is cloudy", "include_truth_table": true}' | python3 -m json.tool
```

### **Test Conditional**
```bash
curl -X POST "http://127.0.0.1:8000/convert" \
     -H "Content-Type: application/json" \
     -d '{"text": "If it is not raining then we can go outside", "include_truth_table": true}' | python3 -m json.tool
```

### **Test Logical Inference**
```bash
curl -X POST "http://127.0.0.1:8000/infer" \
     -H "Content-Type: application/json" \
     -d '{
       "premises": ["It is not raining and it is cold"],
       "conclusion": "It is not raining"
     }' | python3 -m json.tool
```

## 🔧 **Alternative Testing Methods**

### **Run Negation Test Script**
```bash
python test_negation.py
```

### **Run Core Functionality Test**
```bash
python standalone_test.py
```

### **Run Unit Tests**
```bash
python -m pytest tests/
```

## 🌐 **Web Interface**

Once the server is running, visit:
- **API Documentation**: http://127.0.0.1:8000/docs
- **Alternative Docs**: http://127.0.0.1:8000/redoc
- **Health Check**: http://127.0.0.1:8000/health

## 📊 **Expected Results**

### **Negation Example**
**Input**: "It is not raining"
**Output**: `¬it_rain`
**Truth Table**: 
- When `it_rain` is `false` → result is `true` ✓
- When `it_rain` is `true` → result is `false` ✓

### **Conjunction Example**
**Input**: "It is raining and it is cold"
**Output**: `(it_rain ∧ cold)`
**Truth Table**: Only true when both are true ✓

## 🛠️ **Troubleshooting**

### **Port Already in Use**
```bash
lsof -i :8000
kill <PID>
```

### **Virtual Environment Issues**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **spaCy Model Missing**
```bash
python -m spacy download en_core_web_sm
```

## 📁 **Project Structure**
```
ELMSLAB/
├── fixed_api.py          # Main API server
├── test_negation.py      # Negation testing
├── standalone_test.py    # Core functionality test
├── src/                  # Source code modules
├── tests/                # Unit tests
├── requirements.txt      # Dependencies
├── README.md            # Full documentation
└── venv/                # Virtual environment
```

## 🎯 **What This API Does**

Converts natural English sentences into formal propositional logic:
- ✅ **Negation**: "It is not raining" → `¬it_rain`
- ✅ **Conjunction**: "A and B" → `(A ∧ B)`
- ✅ **Disjunction**: "A or B" → `(A ∨ B)`
- ✅ **Implication**: "If A then B" → `(A → B)`
- ✅ **Truth Tables**: Complete semantic analysis
- ✅ **Logical Inference**: Valid reasoning detection

## 🚀 **Ready to Go!**


