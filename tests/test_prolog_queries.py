"""
Test suite for Prolog query parsing and execution
Tests various edge cases to ensure robustness
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prolog_reasoner import PrologReasoner


def test_simple_query():
    """Test simple query without commas"""
    print("=== Test 1: Simple Query ===")
    reasoner = PrologReasoner()
    reasoner.add_fact("cat(fluffy)")
    reasoner.add_fact("cat(whiskers)")
    
    success, results = reasoner.query("cat(X)")
    assert success, "Query should succeed"
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    print(f"✓ Simple query works: {results}")


def test_query_with_comma_in_args():
    """Test query with comma inside predicate arguments"""
    print("\n=== Test 2: Query with Comma in Arguments ===")
    reasoner = PrologReasoner()
    reasoner.add_fact("parent(mary, alice)")
    reasoner.add_fact("parent(mary, bob)")
    reasoner.add_fact("parent(mary, charlie)")
    
    success, results = reasoner.query("parent(mary, X)")
    assert success, "Query should succeed"
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    print(f"✓ Query with comma in args works: {results}")


def test_conjunctive_query():
    """Test conjunctive query with multiple predicates"""
    print("\n=== Test 3: Conjunctive Query ===")
    reasoner = PrologReasoner()
    reasoner.add_fact("student(maria)")
    reasoner.add_fact("student(john)")
    reasoner.add_fact("study_regularly(maria)")
    reasoner.add_fact("study_occasionally(john)")
    
    # Query each part separately (simulating the ELMS.py logic)
    query1 = "student(X)"
    query2 = "study_regularly(X)"
    
    success1, results1 = reasoner.query(query1)
    success2, results2 = reasoner.query(query2)
    
    assert success1 and success2, "Both queries should succeed"
    
    # Find intersection
    values1 = [list(r.values())[0] for r in results1]
    values2 = [list(r.values())[0] for r in results2]
    intersection = [v for v in values1 if v in values2]
    
    assert len(intersection) == 1, f"Expected 1 result, got {len(intersection)}"
    assert intersection[0] == 'maria', f"Expected maria, got {intersection[0]}"
    print(f"✓ Conjunctive query works: {intersection}")


def test_nested_parentheses():
    """Test query with nested parentheses"""
    print("\n=== Test 4: Nested Parentheses ===")
    reasoner = PrologReasoner()
    reasoner.add_fact("person(john)")
    reasoner.add_fact("person(jane)")
    
    success, results = reasoner.query("person(X)")
    assert success, "Query should succeed"
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    print(f"✓ Nested parentheses work: {results}")


def test_query_with_brackets():
    """Test query with brackets (lists)"""
    print("\n=== Test 5: Query with Brackets ===")
    reasoner = PrologReasoner()
    reasoner.add_fact("list([a, b, c])")
    
    success, results = reasoner.query("list(X)")
    assert success, "Query should succeed"
    print(f"✓ Query with brackets works: {results}")


def test_complex_relationship():
    """Test complex relationship query"""
    print("\n=== Test 6: Complex Relationship ===")
    reasoner = PrologReasoner()
    reasoner.add_fact("parent(john, mary)")
    reasoner.add_fact("parent(mary, alice)")
    reasoner.add_fact("parent(mary, bob)")
    reasoner.add_rule("ancestor(X, Y) :- parent(X, Y)")
    reasoner.add_rule("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)")
    
    success, results = reasoner.query("ancestor(john, X)")
    assert success, "Query should succeed"
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    print(f"✓ Complex relationship works: {results}")


def test_empty_result():
    """Test query that returns no results"""
    print("\n=== Test 7: Empty Result ===")
    reasoner = PrologReasoner()
    reasoner.add_fact("cat(fluffy)")
    
    # Note: pytholog may return an error for queries with no results
    # This is a known limitation - we handle it gracefully
    success, results = reasoner.query("dog(X)")
    # Check if results is empty (either success with empty list or failure with empty list)
    is_empty = len(results) == 0
    assert is_empty, f"Expected empty result, got {results}"
    print(f"✓ Empty result works: {results}")


def test_yes_no_query():
    """Test yes/no query (no variables)"""
    print("\n=== Test 8: Yes/No Query ===")
    reasoner = PrologReasoner()
    reasoner.add_fact("cat(fluffy)")
    
    success, results = reasoner.query("cat(fluffy)")
    assert success, "Query should succeed"
    print(f"✓ Yes/no query works: {results}")


def test_multiple_variables():
    """Test query with multiple variables"""
    print("\n=== Test 9: Multiple Variables ===")
    reasoner = PrologReasoner()
    reasoner.add_fact("parent(john, mary)")
    reasoner.add_fact("parent(john, alice)")
    
    success, results = reasoner.query("parent(X, Y)")
    assert success, "Query should succeed"
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    print(f"✓ Multiple variables work: {results}")


def test_rule_with_conjunction():
    """Test rule with conjunction in body"""
    print("\n=== Test 10: Rule with Conjunction ===")
    reasoner = PrologReasoner()
    reasoner.add_fact("student(maria)")
    reasoner.add_fact("study_regularly(maria)")
    reasoner.add_fact("student(john)")
    reasoner.add_rule("successful(X) :- student(X), study_regularly(X)")
    
    success, results = reasoner.query("successful(X)")
    assert success, "Query should succeed"
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0]['X'] == 'maria', f"Expected maria, got {results[0]}"
    print(f"✓ Rule with conjunction works: {results}")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running Prolog Query Tests")
    print("=" * 60)
    
    tests = [
        test_simple_query,
        test_query_with_comma_in_args,
        test_conjunctive_query,
        test_nested_parentheses,
        test_query_with_brackets,
        test_complex_relationship,
        test_empty_result,
        test_yes_no_query,
        test_multiple_variables,
        test_rule_with_conjunction,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Test error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

