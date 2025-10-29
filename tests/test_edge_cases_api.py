#!/usr/bin/env python3
"""
Comprehensive Edge Case Testing via API
Tests all possible edge cases to ensure the system is truly dynamic
"""

import requests
import json
import sys

API_BASE = "http://localhost:8002"

def test_query_api(premises, conclusion, expected_answer=None, expected_count=None):
    """Test a single query via API"""
    try:
        response = requests.post(
            f"{API_BASE}/infer",
            json={
                "premises": premises,
                "conclusion": conclusion
            },
            timeout=120
        )
        
        if response.status_code != 200:
            return {
                'success': False,
                'error': f'HTTP {response.status_code}: {response.text[:200]}',
                'premises': premises,
                'conclusion': conclusion
            }
        
        data = response.json()
        valid = data.get('valid', False)
        answer = data.get('answer', '')
        count = data.get('conclusions_count', 0)
        
        # Check expectations
        success = valid
        if expected_answer and expected_answer.lower() not in answer.lower():
            success = False
            if not valid:
                success = False
        if expected_count and count != expected_count:
            success = False
        
        return {
            'success': success,
            'valid': valid,
            'answer': answer,
            'count': count,
            'premises': premises,
            'conclusion': conclusion,
            'expected_answer': expected_answer,
            'expected_count': expected_count,
            'parsed_conclusion': data.get('parsed_conclusion', ''),
            'parsed_premises': data.get('parsed_premises', [])
        }
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'error': 'Cannot connect to API server',
            'premises': premises,
            'conclusion': conclusion
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'premises': premises,
            'conclusion': conclusion
        }

# Comprehensive edge case tests
edge_cases = [
    # Basic copula patterns
    {
        'name': 'Basic copula - Who is a director?',
        'premises': ['Alice is a director', 'Bob is a manager'],
        'conclusion': 'Who is a director?',
        'expected_count': 1
    },
    {
        'name': 'Copula with modifier role (Carol issue)',
        'premises': ['Carol is a director', 'David is a manager'],
        'conclusion': 'Who is a director?',
        'expected_count': 1
    },
    
    # Verb-based queries with collective nouns
    {
        'name': 'Who makes decisions?',
        'premises': ['Alice is a director', 'Carol is a director', 'Directors make decisions'],
        'conclusion': 'Who makes decisions?',
        'expected_count': 2
    },
    {
        'name': 'Who supervises staff?',
        'premises': ['Bob is a manager', 'David is a manager', 'Managers supervise staff'],
        'conclusion': 'Who supervises staff?',
        'expected_count': 2
    },
    
    # Possessive relationships
    {
        'name': 'Possessive - Marys children',
        'premises': ['Mary is parent of Alice', 'Mary is parent of Bob'],
        'conclusion': 'Who are Mary\'s children?',
        'expected_count': 2
    },
    
    # Conjunctions
    {
        'name': 'Conjunction - Alice and Bob are students',
        'premises': ['Alice and Bob are students'],
        'conclusion': 'Who are students?',
        'expected_count': 2
    },
]

def run_comprehensive_tests():
    """Run all edge case tests"""
    print("="*80)
    print("COMPREHENSIVE EDGE CASE TESTING (VIA API)")
    print("="*80)
    print()
    
    results = []
    passed = 0
    failed = 0
    
    for i, test in enumerate(edge_cases, 1):
        print(f"\n[{i}/{len(edge_cases)}] Testing: {test['name']}")
        print(f"  Premises: {', '.join(test['premises'])}")
        print(f"  Conclusion: {test['conclusion']}")
        
        result = test_query_api(
            test['premises'],
            test['conclusion'],
            test.get('expected_answer'),
            test.get('expected_count')
        )
        
        results.append(result)
        
        if result['success']:
            print(f"  ✅ PASS - Answer: {result.get('answer', 'N/A')}, Count: {result.get('count', 0)}")
            passed += 1
        else:
            error = result.get('error', 'Unknown error')
            print(f"  ❌ FAIL: {error}")
            if 'parsed_conclusion' in result:
                print(f"     Parsed query: {result.get('parsed_conclusion', 'N/A')}")
            failed += 1
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {len(edge_cases)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success Rate: {(passed/len(edge_cases)*100):.1f}%")
    
    # List failures
    if failed > 0:
        print("\n" + "="*80)
        print("FAILURES:")
        print("="*80)
        for i, result in enumerate(results, 1):
            if not result['success']:
                print(f"\n[{i}] {result['conclusion']}")
                print(f"   Error: {result.get('error', 'Unknown')}")
                print(f"   Premises: {', '.join(result['premises'])}")
                print(f"   Parsed query: {result.get('parsed_conclusion', 'N/A')}")
                print(f"   Got: valid={result.get('valid')}, count={result.get('count')}, answer={result.get('answer')}")
    
    return results

if __name__ == '__main__':
    results = run_comprehensive_tests()

