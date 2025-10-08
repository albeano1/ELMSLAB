#!/usr/bin/env python3
"""
Comprehensive Edge Case Test Suite
Runs automatically to ensure no edge cases exist in the system
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ELMS import ELMSStandalone
import json
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class TestCase:
    """Test case with expected result"""
    name: str
    premises: List[str]
    conclusion: str
    expected_valid: bool
    category: str
    description: str

class EdgeCaseValidator:
    """Validates that no edge cases exist in the system"""
    
    def __init__(self):
        self.elms = ELMSStandalone()
        self.test_cases = self._create_comprehensive_test_suite()
        
    def _create_comprehensive_test_suite(self) -> List[TestCase]:
        """Create comprehensive test suite covering all edge cases"""
        return [
            # Valid Universal Instantiation Cases
            TestCase(
                name="Bird Flying Valid",
                premises=["All birds can fly", "Tweety is a bird"],
                conclusion="Can Tweety fly?",
                expected_valid=True,
                category="universal_instantiation",
                description="Standard universal instantiation - birds can fly"
            ),
            TestCase(
                name="Gift Gratitude Valid",
                premises=["Jack gave Jill a book", "Everyone who receives a gift feels grateful"],
                conclusion="Does Jill feel grateful?",
                expected_valid=True,
                category="semantic_roles",
                description="Gift-giving with beneficiary semantic role"
            ),
            TestCase(
                name="Doctor Patient Valid",
                premises=["John is a doctor", "Mary is a patient", "John examined Mary", "All doctors help patients"],
                conclusion="Did John help Mary?",
                expected_valid=True,
                category="universal_instantiation",
                description="Doctor-patient universal instantiation"
            ),
            TestCase(
                name="Family Meal Valid",
                premises=["The family shared a meal", "Everyone who shares meals feels connected"],
                conclusion="Does the family feel connected?",
                expected_valid=True,
                category="universal_instantiation",
                description="Family meal sharing universal instantiation"
            ),
            
            # Valid Temporal Reasoning Cases
            TestCase(
                name="Temporal Sequence Valid",
                premises=["Sarah finished her homework before she watched a movie", "Then she went to bed"],
                conclusion="Did Sarah go to bed after finishing her homework and watching a movie?",
                expected_valid=True,
                category="temporal_reasoning",
                description="Temporal sequence with before/then markers"
            ),
            TestCase(
                name="Door Enter Valid",
                premises=["John opened the door", "Then he entered the room"],
                conclusion="Did John enter the room?",
                expected_valid=True,
                category="temporal_reasoning",
                description="Simple temporal sequence"
            ),
            
            # Invalid Cases - Type Mismatch
            TestCase(
                name="Fish Flying Invalid",
                premises=["All birds can fly", "Tweety is a fish"],
                conclusion="Can Tweety fly?",
                expected_valid=False,
                category="universal_instantiation",
                description="Invalid universal instantiation - fish cannot fly"
            ),
            
            # Invalid Cases - Universal Rule Mismatch
            TestCase(
                name="Mismatched Restaurant Invalid",
                premises=[
                    "John and Mary went to their favorite restaurant",
                    "They ordered wine with their meal",
                    "All customers who try new dishes have memorable experiences"
                ],
                conclusion="Did John and Mary have a memorable experience?",
                expected_valid=False,
                category="universal_rule_mismatch",
                description="Mismatched universal rule - ordered wine vs try new dishes"
            ),
            
            # Invalid Cases - Entity Mismatch
            TestCase(
                name="Alice Tom Invalid",
                premises=[
                    "Alice and Bob are friends",
                    "Alice told Bob a secret",
                    "All people who share secrets are close"
                ],
                conclusion="Are Alice and Tom close?",
                expected_valid=False,
                category="entity_mismatch",
                description="Entity mismatch - Alice/Bob vs Alice/Tom"
            ),
            
            # Additional Edge Cases
            TestCase(
                name="Computer Processor Valid",
                premises=["All computers have processors", "My laptop is a computer"],
                conclusion="Does my laptop have a processor?",
                expected_valid=True,
                category="universal_instantiation",
                description="Computer-processor universal instantiation"
            ),
            TestCase(
                name="Rainy Day Valid",
                premises=["All rainy days are wet", "Today is a rainy day"],
                conclusion="Is today wet?",
                expected_valid=True,
                category="universal_instantiation",
                description="Weather universal instantiation"
            ),
            TestCase(
                name="Pronoun Resolution Valid",
                premises=["Jack loves reading", "He reads every day"],
                conclusion="Does Jack read every day?",
                expected_valid=True,
                category="pronoun_resolution",
                description="Pronoun resolution - he → Jack"
            ),
            
            # Negative entity cases
            TestCase(
                name="Wrong Person Invalid",
                premises=["John gave Mary a gift", "Everyone who receives gifts feels happy"],
                conclusion="Does Tom feel happy?",
                expected_valid=False,
                category="entity_mismatch",
                description="Entity mismatch - Mary vs Tom"
            ),
            
            # Temporal edge cases
            TestCase(
                name="Temporal Chain Valid",
                premises=[
                    "Alice woke up at 7am",
                    "Then she had breakfast",
                    "After that she went to work"
                ],
                conclusion="Did Alice go to work after waking up?",
                expected_valid=True,
                category="temporal_reasoning",
                description="Multi-step temporal chain"
            ),
        ]
    
    def run_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Run a single test case"""
        try:
            result = self.elms.prove_theorem(test_case.premises, test_case.conclusion)
            
            actual_valid = result.get('valid', False)
            confidence = result.get('confidence', 0)
            matches = actual_valid == test_case.expected_valid
            
            return {
                'name': test_case.name,
                'category': test_case.category,
                'expected_valid': test_case.expected_valid,
                'actual_valid': actual_valid,
                'confidence': confidence,
                'matches': matches,
                'explanation': result.get('explanation', ''),
                'reasoning_steps': result.get('reasoning_steps', [])
            }
        except Exception as e:
            return {
                'name': test_case.name,
                'category': test_case.category,
                'expected_valid': test_case.expected_valid,
                'actual_valid': False,
                'confidence': 0,
                'matches': False,
                'error': str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all edge case tests"""
        print("=" * 70)
        print("COMPREHENSIVE EDGE CASE VALIDATION")
        print("=" * 70)
        
        results = {
            'total_tests': len(self.test_cases),
            'passed': 0,
            'failed': 0,
            'test_results': [],
            'failed_tests': []
        }
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[{i}/{len(self.test_cases)}] Testing: {test_case.name}")
            print(f"   Category: {test_case.category}")
            print(f"   Expected: {'VALID' if test_case.expected_valid else 'INVALID'}")
            
            test_result = self.run_test(test_case)
            results['test_results'].append(test_result)
            
            if test_result['matches']:
                results['passed'] += 1
                print(f"   Result: PASS (Got {test_result['actual_valid']}, Confidence: {test_result['confidence']})")
            else:
                results['failed'] += 1
                results['failed_tests'].append({
                    'name': test_case.name,
                    'expected': test_case.expected_valid,
                    'actual': test_result['actual_valid'],
                    'category': test_case.category
                })
                print(f"   Result: FAIL (Expected {test_case.expected_valid}, Got {test_result['actual_valid']})")
                if 'error' in test_result:
                    print(f"   Error: {test_result['error']}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed']} ({results['passed']/results['total_tests']*100:.1f}%)")
        print(f"Failed: {results['failed']} ({results['failed']/results['total_tests']*100:.1f}%)")
        
        if results['failed'] == 0:
            print("\n SUCCESS: ALL EDGE CASES PASS!")
            print("=" * 70)
        else:
            print(f"\n WARNING: {results['failed']} EDGE CASES FAILED!")
            print("\nFailed Tests:")
            for failed in results['failed_tests']:
                print(f"  - {failed['name']} ({failed['category']})")
                print(f"    Expected: {failed['expected']}, Got: {failed['actual']}")
            print("=" * 70)
        
        return results
    
    def assert_no_edge_cases(self):
        """Assert that no edge cases exist - raises exception if any test fails"""
        results = self.run_all_tests()
        
        if results['failed'] > 0:
            raise AssertionError(
                f"{results['failed']} edge case(s) detected! "
                f"Failed tests: {', '.join([t['name'] for t in results['failed_tests']])}"
            )
        
        return True

def main():
    """Main test runner"""
    validator = EdgeCaseValidator()
    
    try:
        validator.assert_no_edge_cases()
        print("\nValidation successful - no edge cases detected!")
        return 0
    except AssertionError as e:
        print(f"\nValidation failed: {e}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error during validation: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())

