#!/usr/bin/env python3
"""
Comprehensive Edge Case Testing for ELMS System
Tests all possible edge cases to ensure the system is truly dynamic
"""

import subprocess
import json
import sys
import os

def test_query(premises, conclusion, expected_answer=None, expected_count=None):
    """Test a single query and return results"""
    # Convert premises to a single string
    premises_str = ". ".join(premises) + "." if premises else ""
    full_input = f"{premises_str} {conclusion}"
    
    try:
        # Run ELMS.py with JSON output
        result = subprocess.run(
            ["python3", "ELMS.py", full_input, "--json", "--env", "prod"],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': f'Command failed: {result.stderr}',
                'premises': premises,
                'conclusion': conclusion
            }
        
        # Try to parse JSON from output
        try:
            output_lines = result.stdout.strip().split('\n')
            json_start = None
            for i, line in enumerate(output_lines):
                if line.strip().startswith('{'):
                    json_start = i
                    break
            
            if json_start is not None:
                json_str = '\n'.join(output_lines[json_start:])
                data = json.loads(json_str)
                
                valid = data.get('valid', False)
                answer = data.get('answer', '')
                count = data.get('conclusions_count', 0)
                
                # Check expectations
                success = valid
                if expected_answer and expected_answer.lower() not in answer.lower():
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
                    'expected_count': expected_count
                }
        except json.JSONDecodeError:
            pass
        
        return {
            'success': False,
            'error': 'Could not parse JSON output',
            'premises': premises,
            'conclusion': conclusion,
            'stdout': result.stdout[:500]
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Timeout',
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
        'name': 'Basic copula - is a',
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
        'expected_count': 2,
        'expected_answer': 'alice'
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
    
    # Universal quantification
    {
        'name': 'Universal - All cats are mammals',
        'premises': ['All cats are mammals', 'Fluffy is a cat'],
        'conclusion': 'Is Fluffy a mammal?',
        'expected_answer': 'yes'
    },
    {
        'name': 'Universal - What mammals do we have?',
        'premises': ['All cats are mammals', 'Fluffy is a cat', 'Whiskers is a cat'],
        'conclusion': 'What mammals do we have?',
        'expected_count': 2
    },
    
    # Verb patterns
    {
        'name': 'Transitive verb - give',
        'premises': ['John gives Mary a book'],
        'conclusion': 'What does John give Mary?',
        'expected_answer': 'book'
    },
    {
        'name': 'Verb with adverb - studies regularly',
        'premises': ['Maria studies regularly', 'John works hard'],
        'conclusion': 'Who studies regularly?',
        'expected_count': 1
    },
    
    # Negation
    {
        'name': 'Negation - No birds can fly underwater',
        'premises': ['No birds can fly underwater', 'Penguins are birds'],
        'conclusion': 'Can penguins fly underwater?',
        'expected_answer': 'no'
    },
    
    # Complex queries
    {
        'name': 'Complex - Who are students who study regularly?',
        'premises': ['Maria is a student', 'John is a student', 'Maria studies regularly'],
        'conclusion': 'Who are students who study regularly?',
        'expected_count': 1
    },
    
    # Plural handling
    {
        'name': 'Plural handling - Directors make decisions',
        'premises': ['Alice is a director', 'Carol is a director', 'Directors make decisions'],
        'conclusion': 'Who makes decisions?',
        'expected_count': 2
    },
]

def run_comprehensive_tests():
    """Run all edge case tests"""
    print("="*80)
    print("COMPREHENSIVE EDGE CASE TESTING")
    print("="*80)
    print()
    
    results = []
    passed = 0
    failed = 0
    vectionary_limitations = 0
    
    for i, test in enumerate(edge_cases, 1):
        print(f"\n[{i}/{len(edge_cases)}] Testing: {test['name']}")
        print(f"  Premises: {', '.join(test['premises'])}")
        print(f"  Conclusion: {test['conclusion']}")
        
        result = test_query(
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
            # Check if it's a Vectionary limitation
            if 'could not convert' in error.lower() or 'no pattern matched' in error.lower():
                print(f"  ⚠️  VECTIONARY LIMITATION: {error[:100]}")
                vectionary_limitations += 1
            else:
                print(f"  ❌ FAIL: {error[:100]}")
                failed += 1
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {len(edge_cases)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Vectionary Limitations: {vectionary_limitations} ⚠️")
    print(f"Success Rate: {(passed/len(edge_cases)*100):.1f}%")
    
    # List failures that are NOT Vectionary limitations
    if failed > 0:
        print("\n" + "="*80)
        print("SYSTEM ISSUES (NOT VECTIONARY LIMITATIONS):")
        print("="*80)
        for result in results:
            if not result['success']:
                error = result.get('error', 'Unknown error')
                if 'could not convert' not in error.lower() and 'no pattern matched' not in error.lower():
                    print(f"\n❌ {result['conclusion']}")
                    print(f"   Error: {error[:200]}")
                    print(f"   Premises: {', '.join(result['premises'])}")
    
    return results

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    results = run_comprehensive_tests()

