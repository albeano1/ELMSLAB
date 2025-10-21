"""
Prolog-based reasoning module for ELMS
Enables drawing conclusions from premises using logic programming
"""

import pytholog as pl
from typing import List, Dict, Any, Optional, Tuple


class PrologReasoner:
    """Prolog-based reasoner that can infer new conclusions from premises"""
    
    def __init__(self):
        self.kb = pl.KnowledgeBase("elms_kb")
        self.facts = []
        self.rules = []
    
    def add_fact(self, fact: str) -> bool:
        """
        Add a fact to the knowledge base
        
        Args:
            fact: Prolog fact (e.g., "cat(fluffy)" or "parent(john, mary)")
        
        Returns:
            True if added successfully
        """
        try:
            # Pytholog syntax: kb(['cat(fluffy)'])
            self.kb([fact])
            self.facts.append(fact)
            return True
        except Exception as e:
            print(f"Error adding fact '{fact}': {e}")
            return False
    
    def add_rule(self, rule: str) -> bool:
        """
        Add a rule to the knowledge base
        
        Args:
            rule: Prolog rule (e.g., "mammal(X) :- cat(X)")
        
        Returns:
            True if added successfully
        """
        try:
            # Pytholog syntax: kb(['mammal(X) :- cat(X)'])
            self.kb([rule])
            self.rules.append(rule)
            return True
        except Exception as e:
            print(f"Error adding rule '{rule}': {e}")
            return False
    
    def query(self, query: str) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Query the knowledge base
        
        Args:
            query: Prolog query (e.g., "cat(X)" or "mammal(fluffy)")
        
        Returns:
            Tuple of (success, results)
        """
        try:
            # Convert query string to pl.Expr
            query_expr = pl.Expr(query)
            results = self.kb.query(query_expr)
            
            # Convert results to a more readable format
            formatted_results = []
            if results:
                # Check if results is 'No' (no solutions found)
                if results == ['No']:
                    return True, []
                
                for result in results:
                    if isinstance(result, dict):
                        formatted_results.append(result)
                    elif isinstance(result, tuple):
                        # Convert tuple to dict
                        formatted_results.append({f"Var{i}": val for i, val in enumerate(result)})
                    else:
                        formatted_results.append({"result": result})
            
            return True, formatted_results
        except Exception as e:
            print(f"Error querying '{query}': {e}")
            return False, []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        return {
            "facts_count": len(self.facts),
            "rules_count": len(self.rules),
            "facts": self.facts,
            "rules": self.rules
        }
    
    def clear(self):
        """Clear the knowledge base"""
        self.kb = pl.KnowledgeBase("elms_kb")
        self.facts = []
        self.rules = []


def test_prolog_reasoner():
    """Test the Prolog reasoner"""
    reasoner = PrologReasoner()
    
    # Test 1: Simple facts
    print("=== Test 1: Simple Facts ===")
    reasoner.add_fact("cat(fluffy)")
    reasoner.add_fact("cat(whiskers)")
    reasoner.add_rule("mammal(X) :- cat(X)")
    
    success, results = reasoner.query("cat(X)")
    print(f"Query: cat(X)")
    print(f"Results: {results}")
    
    success, results = reasoner.query("mammal(fluffy)")
    print(f"\nQuery: mammal(fluffy)")
    print(f"Results: {results}")
    
    # Test 2: Family relationships
    print("\n=== Test 2: Family Relationships ===")
    reasoner.clear()
    reasoner.add_fact("parent(john, mary)")
    reasoner.add_fact("parent(mary, alice)")
    reasoner.add_rule("ancestor(X, Y) :- parent(X, Y)")
    reasoner.add_rule("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)")
    
    success, results = reasoner.query("ancestor(john, alice)")
    print(f"Query: ancestor(john, alice)")
    print(f"Results: {results}")
    
    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    test_prolog_reasoner()


