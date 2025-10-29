#!/usr/bin/env python3
"""
Natural Language Edge Case Testing
Tests the system with real-world natural language inputs that users would actually use.
No hardcoding - system must be fully dynamic.
"""

import requests
import json
import sys

API_BASE = "http://localhost:8002"

def test_natural_language(premises, question, expected_count=None, expected_answer_contains=None):
    """Test with natural language - exactly as users would type"""
    try:
        response = requests.post(
            f"{API_BASE}/infer",
            json={
                "premises": premises,
                "conclusion": question
            },
            timeout=120
        )
        
        if response.status_code != 200:
            return {
                'success': False,
                'error': f'HTTP {response.status_code}',
                'question': question,
                'premises': premises
            }
        
        data = response.json()
        valid = data.get('valid', False)
        answer = data.get('answer', '')
        count = data.get('conclusions_count', 0)
        
        success = valid
        if expected_count and count != expected_count:
            success = False
        if expected_answer_contains and expected_answer_contains.lower() not in answer.lower():
            success = False
        
        return {
            'success': success,
            'valid': valid,
            'answer': answer,
            'count': count,
            'question': question,
            'premises': premises,
            'parsed_query': data.get('parsed_conclusion', ''),
            'expected_count': expected_count,
            'expected_answer_contains': expected_answer_contains
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'question': question,
            'premises': premises
        }

# Natural language test cases - exactly as users would type
natural_language_tests = [
    # Basic "is a" patterns
    {
        'premises': ['Alice is a director', 'Bob is a manager'],
        'question': 'Who is a director?',
        'expected_count': 1
    },
    {
        'premises': ['Carol is a director', 'David is a manager'],
        'question': 'Who is a director?',
        'expected_count': 1,
        'expected_answer_contains': 'carol'
    },
    
    # Verb-based queries with collective nouns
    {
        'premises': ['Alice is a director', 'Carol is a director', 'Directors make decisions'],
        'question': 'Who makes decisions?',
        'expected_count': 2,
        'expected_answer_contains': 'alice'
    },
    {
        'premises': ['Bob is a manager', 'David is a manager', 'Managers supervise staff'],
        'question': 'Who supervises staff?',
        'expected_count': 2
    },
    
    # Possessive relationships
    {
        'premises': ['Mary is parent of Alice', 'Mary is parent of Bob'],
        'question': "Who are Mary's children?",
        'expected_count': 2
    },
    
    # Conjunctions
    {
        'premises': ['Alice and Bob are students'],
        'question': 'Who are students?',
        'expected_count': 2
    },
    
    # Family relationships
    {
        'premises': ['John is parent of Sarah', 'John is parent of Mike', 'Sarah is parent of Emma'],
        'question': "Who are John's children?",
        'expected_count': 2
    },
    
    # Multiple attributes
    {
        'premises': ['Alice is a director', 'Alice is a manager', 'Bob is a director'],
        'question': 'Who is a director?',
        'expected_count': 2
    },
]

def run_tests():
    """Run all natural language tests"""
    print("="*80)
    print("NATURAL LANGUAGE EDGE CASE TESTING")
    print("="*80)
    print("Testing with real-world natural language inputs")
    print()
    
    results = []
    passed = 0
    failed = 0
    
    for i, test in enumerate(natural_language_tests, 1):
        print(f"\n[{i}/{len(natural_language_tests)}] Q: {test['question']}")
        print(f"   P: {', '.join(test['premises'])}")
        
        result = test_natural_language(
            test['premises'],
            test['question'],
            test.get('expected_count'),
            test.get('expected_answer_contains')
        )
        
        results.append(result)
        
        if result['success']:
            print(f"   ✅ PASS - Answer: {result.get('answer', 'N/A')}, Count: {result.get('count', 0)}")
        else:
            print(f"   ❌ FAIL - Answer: {result.get('answer', 'N/A')}, Count: {result.get('count', 0)}")
            print(f"      Parsed: {result.get('parsed_query', 'N/A')}")
            if 'error' in result:
                print(f"      Error: {result['error']}")
            failed += 1
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    # Show failures
    if failed > 0:
        print("\n" + "="*80)
        print("FAILURES:")
        print("="*80)
        for result in results:
            if not result['success']:
                print(f"\n❌ {result['question']}")
                print(f"   Premises: {', '.join(result['premises'])}")
                print(f"   Got: valid={result.get('valid')}, count={result.get('count')}, answer={result.get('answer')}")
                print(f"   Expected: count={result.get('expected_count')}")
                print(f"   Parsed query: {result.get('parsed_query', 'N/A')}")
    
    return results

if __name__ == '__main__':
    try:
        results = run_tests()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest suite error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

