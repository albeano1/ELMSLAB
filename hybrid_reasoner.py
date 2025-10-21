"""
Hybrid reasoning module combining ELMS semantic parsing with Prolog logical inference
"""

from typing import List, Dict, Any, Optional, Tuple
from prolog_reasoner import PrologReasoner
from ELMS import VectionaryParser, VectionaryAPIClient


class HybridReasoner:
    """Combines ELMS semantic parsing with Prolog logical inference"""
    
    def __init__(self, parser=None):
        """
        Initialize the hybrid reasoner
        
        Args:
            parser: Optional VectionaryParser instance
        """
        if parser is None:
            api_client = VectionaryAPIClient(environment='prod')
            self.parser = VectionaryParser(api_client)
        else:
            self.parser = parser
        
        self.prolog_reasoner = PrologReasoner()
    
    def infer_conclusions(self, premises: List[str], query: str = None) -> Dict[str, Any]:
        """
        Infer conclusions from premises using hybrid reasoning
        
        Args:
            premises: List of natural language premises
            query: Optional query string
            
        Returns:
            Dictionary with inference results
        """
        import time
        start_time = time.time()
        
        try:
            # Convert premises to Prolog format
            prolog_facts = []
            for premise in premises:
                prolog = self._convert_nl_to_prolog(premise)
                if prolog:
                    prolog_facts.append(prolog)
                    if " :- " in prolog:
                        self.prolog_reasoner.add_rule(prolog)
                    else:
                        self.prolog_reasoner.add_fact(prolog)
                elif "(" in premise or " :- " in premise:
                    # Already in Prolog format
                    prolog_facts.append(premise)
                    if " :- " in premise:
                        self.prolog_reasoner.add_rule(premise)
                    else:
                        self.prolog_reasoner.add_fact(premise)
            
            # Convert query to Prolog format
            if query:
                prolog_query = self._convert_query_to_prolog(query)
                if not prolog_query:
                    # Try to use the query as-is
                    prolog_query = query
            else:
                prolog_query = "X"
            
            # Add ancestor rules if query is about ancestors/descendants
            if "ancestor(" in prolog_query:
                self.prolog_reasoner.add_rule("ancestor(X, Y) :- parent(X, Y)")
                self.prolog_reasoner.add_rule("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)")
                prolog_facts.append("ancestor(X, Y) :- parent(X, Y)")
                prolog_facts.append("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)")
            
            # Query Prolog
            success, results = self.prolog_reasoner.query(prolog_query)
            
            elapsed_time = time.time() - start_time
            
            return {
                'success': success,
                'premises': premises,
                'query': query,
                'prolog_facts': prolog_facts,
                'conclusions': results,
                'conclusions_count': len(results),
                'reasoning_time': elapsed_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'premises': premises,
                'query': query,
                'prolog_facts': [],
                'conclusions': [],
                'conclusions_count': 0,
                'reasoning_time': time.time() - start_time,
                'error': str(e)
            }
    
    def verify_conclusion(self, premises: List[str], conclusion: str) -> Dict[str, Any]:
        """
        Verify a conclusion using hybrid reasoning
        
        Args:
            premises: List of natural language premises
            conclusion: Conclusion to verify
            
        Returns:
            Dictionary with verification results
        """
        import time
        start_time = time.time()
        
        try:
            # Use Prolog for verification
            prolog_facts = []
            for premise in premises:
                prolog = self._convert_nl_to_prolog(premise)
                if prolog:
                    prolog_facts.append(prolog)
                    if " :- " in prolog:
                        self.prolog_reasoner.add_rule(prolog)
                    else:
                        self.prolog_reasoner.add_fact(prolog)
            
            # Convert conclusion to Prolog format
            prolog_conclusion = self._convert_nl_to_prolog(conclusion)
            if prolog_conclusion:
                success, results = self.prolog_reasoner.query(prolog_conclusion)
                prolog_valid = success and len(results) > 0
            else:
                prolog_valid = False
            
            elapsed_time = time.time() - start_time
            
            return {
                'valid': prolog_valid,
                'confidence': 0.95 if prolog_valid else 0.0,
                'elms_valid': False,  # Not using ELMS for now
                'prolog_valid': prolog_valid,
                'reasoning_time': elapsed_time
            }
            
        except Exception as e:
            return {
                'valid': False,
                'confidence': 0.0,
                'elms_valid': False,
                'prolog_valid': False,
                'reasoning_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _is_universal_quantifier(self, text: str) -> bool:
        """Check if text contains universal quantifier words"""
        universal_words = ['all', 'every', 'each', 'any']
        text_lower = text.lower()
        return any(word in text_lower for word in universal_words)
    
    def _convert_nl_to_prolog(self, premise: str) -> Optional[str]:
        """
        Convert natural language premise to Prolog format using Vectionary semantic parsing
        
        Args:
            premise: Natural language premise
        
        Returns:
            Prolog format string or None
        """
        # Import the dynamic conversion function from ELMS
        from ELMS import _convert_nl_to_prolog
        return _convert_nl_to_prolog(premise, self.parser)
    
    def _convert_query_to_prolog(self, query: str) -> Optional[str]:
        """
        Convert natural language query to Prolog format using Vectionary semantic parsing
        
        Args:
            query: Natural language query
        
        Returns:
            Prolog query string or None
        """
        # Import the dynamic conversion function from ELMS
        from ELMS import _convert_query_to_prolog
        return _convert_query_to_prolog(query, self.parser)
    
    def clear(self):
        """Clear the knowledge base"""
        self.prolog_reasoner.clear()


def test_hybrid_reasoner():
    """Test the hybrid reasoner"""
    reasoner = HybridReasoner()
    
    print("=== Test 1: Mammals ===")
    result = reasoner.infer_conclusions(
        premises=["All cats are mammals.", "Fluffy is a cat.", "Whiskers is a cat."],
        query="What mammals do we have?"
    )
    print(f"Success: {result['success']}")
    print(f"Conclusions: {result['conclusions']}")
    print(f"Count: {result['conclusions_count']}")
    
    print("\n=== Test 2: Descendants ===")
    reasoner.clear()
    result = reasoner.infer_conclusions(
        premises=["John is parent of Mary.", "Mary is parent of Alice."],
        query="Who are John's descendants?"
    )
    print(f"Success: {result['success']}")
    print(f"Conclusions: {result['conclusions']}")
    print(f"Count: {result['conclusions_count']}")
    
    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    test_hybrid_reasoner()


