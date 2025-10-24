#!/usr/bin/env python3
"""
Natural Language Tests with Debug Output Handling

Test runner that handles debug output and extracts JSON from the full output.
"""

import subprocess
import json
import sys
import os
import time
import re

def extract_json_from_output(output):
    """Extract JSON from mixed debug output"""
    # Look for JSON in the output
    lines = output.split('\n')
    json_lines = []
    in_json = False
    
    for line in lines:
        if line.strip().startswith('{'):
            in_json = True
        if in_json:
            json_lines.append(line)
            if line.strip().endswith('}'):
                break
    
    if json_lines:
        json_str = '\n'.join(json_lines)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON anywhere in the output
    json_match = re.search(r'\{.*\}', output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    return None

def run_natural_language_test(premises, conclusion, test_name, timeout=120):
    """Run a natural language test with debug output handling"""
    print(f"\n🧪 {test_name}")
    print(f"📝 Input: {premises}")
    print(f"❓ Question: {conclusion}")
    
    full_input = f"{premises} {conclusion}"
    cmd = ["python3", "ELMS.py", full_input, "--env", "prod", "--json"]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        end_time = time.time()
        
        if result.returncode == 0:
            # Try to extract JSON from the output
            json_data = extract_json_from_output(result.stdout)
            
            if json_data:
                success = json_data.get('valid', False)
                confidence = json_data.get('confidence', 0.0)
                reasoning_time = json_data.get('reasoning_time', 0.0)
                conclusions = json_data.get('conclusions', [])
                conclusions_count = json_data.get('conclusions_count', 0)
                
                print(f"✅ Result: {'Valid' if success else 'Invalid'}")
                print(f"📊 Confidence: {confidence:.2f}")
                print(f"⏱️ Time: {end_time - start_time:.2f}s")
                print(f"🧠 Reasoning Time: {reasoning_time:.2f}s")
                print(f"💡 Conclusions: {conclusions_count}")
                if conclusions:
                    print(f"📝 Results: {conclusions}")
                
                return {
                    'success': True,
                    'valid': success,
                    'confidence': confidence,
                    'total_time': end_time - start_time,
                    'reasoning_time': reasoning_time,
                    'conclusions_count': conclusions_count,
                    'conclusions': conclusions
                }
            else:
                print(f"❌ No JSON found in output")
                print(f"Raw output: {result.stdout[:300]}...")
                return {'success': False, 'error': 'No JSON found in output'}
        else:
            print(f"❌ Command failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return {'success': False, 'error': f'Command failed: {result.stderr}'}
    except subprocess.TimeoutExpired:
        print(f"⏰ Test timed out after {timeout} seconds")
        return {'success': False, 'error': f'Timeout after {timeout} seconds'}
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """Run natural language tests with debug output handling"""
    print("🚀 NATURAL LANGUAGE TESTS WITH DEBUG OUTPUT")
    print("="*60)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Natural language tests
    tests = [
        {
            'premises': "All cats are mammals. Fluffy is a cat. Whiskers is a cat.",
            'conclusion': "What mammals do we have?",
            'name': "Universal Quantification",
            'timeout': 60
        },
        {
            'premises': "Alice is a doctor. Bob is an engineer. Carol is a teacher.",
            'conclusion': "Who are the professionals?",
            'name': "Copula Patterns",
            'timeout': 60
        },
        {
            'premises': "Maria is a student. John is a student. Maria studies regularly.",
            'conclusion': "Who are students?",
            'name': "Simple Open-ended",
            'timeout': 60
        },
        {
            'premises': "All dogs are animals. Fido is a dog.",
            'conclusion': "Is Fido an animal?",
            'name': "Yes/No Question",
            'timeout': 60
        }
    ]
    
    results = []
    passed = 0
    total = len(tests)
    
    for test in tests:
        result = run_natural_language_test(
            test['premises'], 
            test['conclusion'], 
            test['name'],
            test['timeout']
        )
        results.append(result)
        
        if result['success']:
            passed += 1
    
    # Print summary
    print(f"\n📊 RESULTS: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed!")
    elif passed >= total * 0.8:
        print("✅ Most tests passed - system working well")
    else:
        print("⚠️ Some tests failed - check system status")
    
    # Performance summary
    successful_tests = [r for r in results if r['success']]
    if successful_tests:
        avg_confidence = sum(r.get('confidence', 0) for r in successful_tests) / len(successful_tests)
        avg_time = sum(r.get('total_time', 0) for r in successful_tests) / len(successful_tests)
        avg_reasoning_time = sum(r.get('reasoning_time', 0) for r in successful_tests) / len(successful_tests)
        print(f"\n📈 PERFORMANCE:")
        print(f"Average Confidence: {avg_confidence:.2f}")
        print(f"Average Total Time: {avg_time:.2f}s")
        print(f"Average Reasoning Time: {avg_reasoning_time:.2f}s")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
