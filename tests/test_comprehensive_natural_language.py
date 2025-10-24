#!/usr/bin/env python3
"""
Comprehensive Natural Language System Tests

Complete test suite for the ELMS system using natural language inputs.
Tests the full pipeline from natural language to logical conclusions.
"""

import subprocess
import json
import sys
import os
import time
import re
from typing import List, Dict, Any

class ComprehensiveNaturalLanguageTester:
    """Comprehensive tester for the ELMS natural language system"""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.total = 0
    
    def extract_json_from_output(self, output: str) -> Dict[str, Any]:
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
    
    def run_test(self, premises: str, conclusion: str, test_name: str, timeout: int = 180) -> Dict[str, Any]:
        """Run a comprehensive natural language test"""
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
                json_data = self.extract_json_from_output(result.stdout)
                
                if json_data:
                    success = json_data.get('valid', False)
                    confidence = json_data.get('confidence', 0.0)
                    reasoning_time = json_data.get('reasoning_time', 0.0)
                    conclusions = json_data.get('conclusions', [])
                    conclusions_count = json_data.get('conclusions_count', 0)
                    logic_type = json_data.get('logic_type', 'UNKNOWN')
                    
                    print(f"✅ Result: {'Valid' if success else 'Invalid'}")
                    print(f"📊 Confidence: {confidence:.2f}")
                    print(f"⏱️ Time: {end_time - start_time:.2f}s")
                    print(f"🧠 Reasoning Time: {reasoning_time:.2f}s")
                    print(f"💡 Conclusions: {conclusions_count}")
                    print(f"🔬 Logic Type: {logic_type}")
                    if conclusions:
                        print(f"📝 Results: {conclusions}")
                    
                    return {
                        'test_name': test_name,
                        'success': True,
                        'valid': success,
                        'confidence': confidence,
                        'total_time': end_time - start_time,
                        'reasoning_time': reasoning_time,
                        'conclusions_count': conclusions_count,
                        'conclusions': conclusions,
                        'logic_type': logic_type
                    }
                else:
                    print(f"❌ No JSON found in output")
                    print(f"Raw output: {result.stdout[:300]}...")
                    return {
                        'test_name': test_name,
                        'success': False,
                        'error': 'No JSON found in output'
                    }
            else:
                print(f"❌ Command failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
                return {
                    'test_name': test_name,
                    'success': False,
                    'error': f'Command failed: {result.stderr}',
                    'returncode': result.returncode
                }
        except subprocess.TimeoutExpired:
            print(f"⏰ Test timed out after {timeout} seconds")
            return {
                'test_name': test_name,
                'success': False,
                'error': f'Timeout after {timeout} seconds'
            }
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return {
                'test_name': test_name,
                'success': False,
                'error': str(e)
            }
    
    def test_universal_quantification(self):
        """Test universal quantification patterns"""
        print("\n" + "="*60)
        print("🔬 TESTING UNIVERSAL QUANTIFICATION")
        print("="*60)
        
        tests = [
            {
                'premises': "All cats are mammals. Fluffy is a cat. Whiskers is a cat.",
                'conclusion': "What mammals do we have?",
                'name': "Basic Universal Quantification"
            },
            {
                'premises': "Every student studies hard. All teachers are educators.",
                'conclusion': "What students do we have?",
                'name': "Mixed Quantifiers"
            },
            {
                'premises': "All dogs are animals. Fido is a dog.",
                'conclusion': "Is Fido an animal?",
                'name': "Universal to Yes/No"
            }
        ]
        
        for test in tests:
            result = self.run_test(test['premises'], test['conclusion'], test['name'])
            self.results.append(result)
            self.total += 1
            if result['success']:
                self.passed += 1
            else:
                self.failed += 1
    
    def test_copula_patterns(self):
        """Test copula verb patterns"""
        print("\n" + "="*60)
        print("🔬 TESTING COPULA PATTERNS")
        print("="*60)
        
        tests = [
            {
                'premises': "Alice is a doctor. Bob is an engineer. Carol is a teacher.",
                'conclusion': "Who are the professionals?",
                'name': "Basic Copula Patterns"
            },
            {
                'premises': "John is a student. Mary is a student.",
                'conclusion': "Who are students?",
                'name': "Simple Copula Open-ended"
            },
            {
                'premises': "The sky is blue.",
                'conclusion': "Is the sky blue?",
                'name': "Simple Yes/No Copula"
            }
        ]
        
        for test in tests:
            result = self.run_test(test['premises'], test['conclusion'], test['name'])
            self.results.append(result)
            self.total += 1
            if result['success']:
                self.passed += 1
            else:
                self.failed += 1
    
    def test_transitive_verbs(self):
        """Test transitive verb patterns"""
        print("\n" + "="*60)
        print("🔬 TESTING TRANSITIVE VERBS")
        print("="*60)
        
        tests = [
            {
                'premises': "John gives Mary a book.",
                'conclusion': "What does John give Mary?",
                'name': "3-Argument Transitive Verb"
            },
            {
                'premises': "Maria studies regularly. John works hard.",
                'conclusion': "Who studies regularly?",
                'name': "Transitive Verbs with Adverbs"
            },
            {
                'premises': "Tom helps his friend.",
                'conclusion': "Does Tom help his friend?",
                'name': "Simple Transitive Yes/No"
            }
        ]
        
        for test in tests:
            result = self.run_test(test['premises'], test['conclusion'], test['name'])
            self.results.append(result)
            self.total += 1
            if result['success']:
                self.passed += 1
            else:
                self.failed += 1
    
    def test_edge_cases(self):
        """Test edge cases and complex scenarios"""
        print("\n" + "="*60)
        print("🔬 TESTING EDGE CASES")
        print("="*60)
        
        tests = [
            {
                'premises': "Mary is parent of Alice. Mary is parent of Bob.",
                'conclusion': "Who are Mary's children?",
                'name': "Possessive Relationships"
            },
            {
                'premises': "Alice and Bob are students.",
                'conclusion': "Who are students?",
                'name': "Complex Conjunctions"
            },
            {
                'premises': "No birds can fly underwater. Penguins are birds.",
                'conclusion': "Can penguins fly underwater?",
                'name': "Negation Patterns"
            }
        ]
        
        for test in tests:
            result = self.run_test(test['premises'], test['conclusion'], test['name'])
            self.results.append(result)
            self.total += 1
            if result['success']:
                self.passed += 1
            else:
                self.failed += 1
    
    def run_all_tests(self):
        """Run all comprehensive natural language tests"""
        print("🚀 COMPREHENSIVE NATURAL LANGUAGE SYSTEM TESTS")
        print("="*80)
        
        # Change to project directory
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Run all test categories
        self.test_universal_quantification()
        self.test_copula_patterns()
        self.test_transitive_verbs()
        self.test_edge_cases()
        
        # Print comprehensive summary
        self.print_comprehensive_summary()
    
    def print_comprehensive_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE NATURAL LANGUAGE TEST SUMMARY")
        print("="*80)
        
        print(f"Total Tests: {self.total}")
        print(f"Passed: {self.passed} ✅")
        print(f"Failed: {self.failed} ❌")
        print(f"Success Rate: {(self.passed/self.total)*100:.1f}%")
        
        # Detailed results by category
        print(f"\n📋 DETAILED RESULTS:")
        for result in self.results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{status} {result['test_name']}")
            if result['success']:
                print(f"    Confidence: {result.get('confidence', 0):.2f}")
                print(f"    Time: {result.get('total_time', 0):.2f}s")
                print(f"    Conclusions: {result.get('conclusions_count', 0)}")
            else:
                print(f"    Error: {result.get('error', 'Unknown error')}")
        
        # Performance analysis
        successful_tests = [r for r in self.results if r['success']]
        if successful_tests:
            avg_confidence = sum(r.get('confidence', 0) for r in successful_tests) / len(successful_tests)
            avg_time = sum(r.get('total_time', 0) for r in successful_tests) / len(successful_tests)
            avg_reasoning_time = sum(r.get('reasoning_time', 0) for r in successful_tests) / len(successful_tests)
            
            print(f"\n📈 PERFORMANCE METRICS:")
            print(f"Average Confidence: {avg_confidence:.2f}")
            print(f"Average Total Time: {avg_time:.2f}s")
            print(f"Average Reasoning Time: {avg_reasoning_time:.2f}s")
            
            # Logic type analysis
            logic_types = {}
            for result in successful_tests:
                logic_type = result.get('logic_type', 'UNKNOWN')
                logic_types[logic_type] = logic_types.get(logic_type, 0) + 1
            
            print(f"\n🔬 LOGIC TYPE DISTRIBUTION:")
            for logic_type, count in logic_types.items():
                print(f"  {logic_type}: {count} tests")
        
        # System assessment
        print(f"\n🎯 SYSTEM ASSESSMENT:")
        if self.passed / self.total >= 0.8:
            print("✅ EXCELLENT - System performing well")
        elif self.passed / self.total >= 0.6:
            print("⚠️ GOOD - Some issues to address")
        else:
            print("❌ NEEDS IMPROVEMENT - Significant issues detected")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if self.failed > 0:
            print("• Address failing tests to improve system reliability")
        if avg_time > 30:
            print("• Consider performance optimizations for faster response times")
        if avg_confidence < 0.8:
            print("• Review conversion logic to improve confidence scores")
        
        return self.passed == self.total


def main():
    """Main test runner"""
    print("🧪 ELMS Comprehensive Natural Language System Tester")
    print("Testing the complete system with natural language inputs")
    
    tester = ComprehensiveNaturalLanguageTester()
    success = tester.run_all_tests()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
