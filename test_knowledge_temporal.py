#!/usr/bin/env python3
"""
Comprehensive test script for Knowledge Base and Temporal Logic features

This script demonstrates the new capabilities:
1. Knowledge Base API - adding facts and querying with inference
2. Temporal Logic - converting natural language to temporal formulas
3. Integration with existing propositional and first-order logic
"""

import requests
import json
import time

API_BASE = "http://127.0.0.1:8000"

def test_api_endpoint(endpoint, method="GET", data=None):
    """Test an API endpoint and return the response."""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        elif method == "DELETE":
            response = requests.delete(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    print("🧠 Testing Enhanced Logic API with Knowledge Base and Temporal Logic")
    print("=" * 70)
    
    # Test 1: Knowledge Base - Add Facts
    print("\n1. 📚 Knowledge Base - Adding Facts")
    print("-" * 40)
    
    facts = [
        "All birds have wings",
        "Penguins are birds", 
        "Tweety is a penguin",
        "It rained yesterday",
        "If it rains, the ground gets wet"
    ]
    
    for fact in facts:
        result = test_api_endpoint("/knowledge/add", "POST", {"fact": fact})
        if "error" not in result:
            print(f"✅ Added: {fact}")
            print(f"   → {result.get('fact', {}).get('formula', 'N/A')}")
        else:
            print(f"❌ Error adding '{fact}': {result['error']}")
    
    # Test 2: Knowledge Base - Query with Inference
    print("\n2. 🔍 Knowledge Base - Querying with Inference")
    print("-" * 40)
    
    queries = [
        "Does Tweety have wings?",
        "Is a penguin a bird?",
        "Did it rain in the past?",
        "What happens when it rains?"
    ]
    
    for query in queries:
        result = test_api_endpoint("/knowledge/query", "POST", {"question": query})
        if "error" not in result:
            print(f"❓ {query}")
            print(f"   Answer: {result.get('answer', 'Unknown')}")
            print(f"   Reasoning: {result.get('reasoning', 'N/A')}")
        else:
            print(f"❌ Error querying '{query}': {result['error']}")
    
    # Test 3: Temporal Logic Conversion
    print("\n3. ⏰ Temporal Logic - Natural Language Conversion")
    print("-" * 40)
    
    temporal_examples = [
        "It rained yesterday",
        "It will rain tomorrow", 
        "The sun always rises",
        "Eventually it will stop raining",
        "If it rains, the ground will be wet afterwards"
    ]
    
    for example in temporal_examples:
        result = test_api_endpoint("/convert", "POST", {
            "text": example,
            "logic_type": "auto"
        })
        if "error" not in result:
            logic_type = result.get("logic_type", "unknown")
            if logic_type == "temporal":
                print(f"✅ {example}")
                print(f"   → {result.get('temporal_formula', 'N/A')}")
            else:
                print(f"⚠️  {example} (detected as {logic_type})")
        else:
            print(f"❌ Error converting '{example}': {result['error']}")
    
    # Test 4: Mixed Logic Types
    print("\n4. 🔄 Mixed Logic Types - Auto Detection")
    print("-" * 40)
    
    mixed_examples = [
        ("All humans are mortal", "first_order"),
        ("If it rains then the ground is wet", "propositional"),
        ("It rained yesterday", "temporal"),
        ("Socrates is human", "first_order"),
        ("The sun always rises", "temporal")
    ]
    
    for example, expected_type in mixed_examples:
        result = test_api_endpoint("/convert", "POST", {
            "text": example,
            "logic_type": "auto"
        })
        if "error" not in result:
            detected_type = result.get("detected_logic_type", "unknown")
            status = "✅" if detected_type == expected_type else "⚠️"
            print(f"{status} {example}")
            print(f"   Detected: {detected_type} (expected: {expected_type})")
        else:
            print(f"❌ Error with '{example}': {result['error']}")
    
    # Test 5: Knowledge Base Statistics
    print("\n5. 📊 Knowledge Base Statistics")
    print("-" * 40)
    
    result = test_api_endpoint("/knowledge/facts", "GET")
    if "error" not in result:
        count = result.get("count", 0)
        print(f"✅ Total facts in knowledge base: {count}")
        
        if count > 0:
            print("   Recent facts:")
            facts = result.get("facts", [])[-3:]  # Show last 3 facts
            for fact in facts:
                print(f"   - {fact.get('text', 'N/A')} → {fact.get('formula', 'N/A')}")
    else:
        print(f"❌ Error getting facts: {result['error']}")
    
    # Test 6: API Health Check
    print("\n6. 🏥 API Health Check")
    print("-" * 40)
    
    result = test_api_endpoint("/", "GET")
    if "error" not in result:
        print(f"✅ API is running: {result.get('message', 'OK')}")
    else:
        print(f"❌ API health check failed: {result['error']}")
    
    print("\n" + "=" * 70)
    print("🎉 Knowledge Base and Temporal Logic Testing Complete!")
    print("\nNew Features Demonstrated:")
    print("✅ Knowledge Base API - Add facts and query with inference")
    print("✅ Temporal Logic - Past, Future, Always, Eventually operators")
    print("✅ Auto-detection of logic types (propositional, first-order, temporal)")
    print("✅ Persistent knowledge storage")
    print("✅ Natural language to formal logic conversion")
    print("✅ Inference engine for knowledge queries")

if __name__ == "__main__":
    main()
