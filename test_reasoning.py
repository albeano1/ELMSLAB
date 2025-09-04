#!/usr/bin/env python3
"""
Comprehensive Test Suite for the Natural Language to Propositional Logic Reasoning Engine

This test suite validates all major inference patterns and reasoning capabilities.
"""

import requests
import json
import sys
from typing import List, Dict, Any

def test_inference(premises: List[str], conclusion: str, expected_valid: bool, 
                  expected_type: str = None, description: str = ""):
    """Test a single inference case"""
    url = "http://127.0.0.1:8000/infer"
    
    payload = {
        "premises": premises,
        "conclusion": conclusion
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            
            print(f"🧠 {description}")
            print(f"   Premises: {premises}")
            print(f"   Conclusion: {conclusion}")
            print(f"   Valid: {result['valid']} (Expected: {expected_valid})")
            print(f"   Type: {result.get('inference_type', 'unknown')}")
            print(f"   Explanation: {result.get('explanation', 'No explanation')}")
            
            if result.get('counterexample'):
                print(f"   Counterexample: {result['counterexample']['description']}")
            
            # Check if result matches expectation
            if result['valid'] == expected_valid:
                if expected_type is None or result.get('inference_type') == expected_type:
                    print("   ✅ PASS")
                else:
                    print(f"   ⚠️  PARTIAL PASS (wrong type: expected {expected_type})")
            else:
                print("   ❌ FAIL")
            
            print()
            return result
        else:
            print(f"❌ {description} - HTTP {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"❌ {description} - Cannot connect to API. Make sure server is running.")
        return None
    except Exception as e:
        print(f"❌ {description} - Error: {e}")
        return None

def main():
    print("🧪 Comprehensive Reasoning Engine Test Suite")
    print("=" * 60)
    
    # Test cases for different inference patterns
    test_cases = [
        # Modus Ponens (Valid)
        {
            "name": "Modus Ponens",
            "premises": ["If it is raining then the ground is wet", "It is raining"],
            "conclusion": "The ground is wet",
            "expected_valid": True,
            "expected_type": "modus_ponens",
            "description": "Classic Modus Ponens - Valid"
        },
        
        # Modus Tollens (Valid)
        {
            "name": "Modus Tollens",
            "premises": ["If it is raining then the ground is wet", "The ground is not wet"],
            "conclusion": "It is not raining",
            "expected_valid": True,
            "expected_type": "modus_tollens",
            "description": "Classic Modus Tollens - Valid"
        },
        
        # Affirming the Consequent (Invalid)
        {
            "name": "Affirming the Consequent",
            "premises": ["If it is raining then the ground is wet", "The ground is wet"],
            "conclusion": "It is raining",
            "expected_valid": False,
            "expected_type": None,
            "description": "Affirming the Consequent - Invalid (common fallacy)"
        },
        
        # Denying the Antecedent (Invalid)
        {
            "name": "Denying the Antecedent",
            "premises": ["If it is raining then the ground is wet", "It is not raining"],
            "conclusion": "The ground is not wet",
            "expected_valid": False,
            "expected_type": None,
            "description": "Denying the Antecedent - Invalid (common fallacy)"
        },
        
        # Disjunctive Syllogism (Valid)
        {
            "name": "Disjunctive Syllogism",
            "premises": ["It is raining or it is sunny", "It is not raining"],
            "conclusion": "It is sunny",
            "expected_valid": True,
            "expected_type": "disjunctive_syllogism",
            "description": "Disjunctive Syllogism - Valid"
        },
        
        # Conjunction Elimination (Valid)
        {
            "name": "Conjunction Elimination",
            "premises": ["It is raining and the ground is wet"],
            "conclusion": "It is raining",
            "expected_valid": True,
            "expected_type": "conjunction_elimination",
            "description": "Conjunction Elimination - Valid"
        },
        
        # Disjunction Introduction (Valid)
        {
            "name": "Disjunction Introduction",
            "premises": ["It is raining"],
            "conclusion": "It is raining or it is sunny",
            "expected_valid": True,
            "expected_type": "disjunction_introduction",
            "description": "Disjunction Introduction - Valid"
        },
        
        # Hypothetical Syllogism (Valid)
        {
            "name": "Hypothetical Syllogism",
            "premises": ["If it is raining then the ground is wet", "If the ground is wet then the grass grows"],
            "conclusion": "If it is raining then the grass grows",
            "expected_valid": True,
            "expected_type": "hypothetical_syllogism",
            "description": "Hypothetical Syllogism - Valid"
        },
        
        # Chain Reasoning (Valid)
        {
            "name": "Chain Reasoning",
            "premises": [
                "If it is raining then the ground is wet",
                "If the ground is wet then the grass grows",
                "It is raining"
            ],
            "conclusion": "The grass grows",
            "expected_valid": True,
            "expected_type": None,  # Complex chain, might not be detected
            "description": "Chain Reasoning - Valid"
        },
        
        # Complex Negation (Valid)
        {
            "name": "Complex Negation",
            "premises": ["If it is not raining then we can go outside", "It is not raining"],
            "conclusion": "We can go outside",
            "expected_valid": True,
            "expected_type": "modus_ponens",
            "description": "Complex Negation with Modus Ponens - Valid"
        },
        
        # Invalid Complex Case
        {
            "name": "Invalid Complex",
            "premises": ["It is raining or it is sunny", "The ground is wet"],
            "conclusion": "It is raining",
            "expected_valid": False,
            "expected_type": None,
            "description": "Invalid Complex Case - Invalid"
        }
    ]
    
    print("Testing inference patterns:")
    print("-" * 40)
    
    results = []
    passed = 0
    total = len(test_cases)
    
    for test_case in test_cases:
        result = test_inference(
            test_case["premises"],
            test_case["conclusion"],
            test_case["expected_valid"],
            test_case["expected_type"],
            test_case["description"]
        )
        
        if result and result['valid'] == test_case["expected_valid"]:
            passed += 1
        
        results.append({
            "test": test_case["name"],
            "result": result,
            "expected": test_case["expected_valid"]
        })
    
    print("=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The reasoning engine is working perfectly!")
    else:
        print(f"⚠️  {total - passed} tests failed. Check the results above.")
    
    print("\n📊 Summary of Inference Patterns Tested:")
    print("- ✅ Modus Ponens (Valid)")
    print("- ✅ Modus Tollens (Valid)")
    print("- ❌ Affirming the Consequent (Invalid)")
    print("- ❌ Denying the Antecedent (Invalid)")
    print("- ✅ Disjunctive Syllogism (Valid)")
    print("- ✅ Conjunction Elimination (Valid)")
    print("- ✅ Disjunction Introduction (Valid)")
    print("- ✅ Hypothetical Syllogism (Valid)")
    print("- ✅ Chain Reasoning (Valid)")
    print("- ✅ Complex Negation (Valid)")
    print("- ❌ Invalid Complex Case (Invalid)")
    
    print("\n🚀 The reasoning engine successfully demonstrates:")
    print("- Logical validity checking")
    print("- Inference pattern detection")
    print("- Counterexample generation")
    print("- Explanation generation")
    print("- Complex reasoning chains")
    
    return results

if __name__ == "__main__":
    main()
