#!/usr/bin/env python3
"""
Debug script for logical reasoning fix
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.knowledge_base import KnowledgeBase

def test_logical_fix():
    """Test the logical reasoning fix"""
    kb = KnowledgeBase()
    
    # Clear any existing facts
    kb.clear_knowledge()
    
    # Add only the universal statement
    print("Adding fact: 'All birds have wings'")
    result1 = kb.add_fact("All birds have wings")
    print(f"Result: {result1['message']}")
    
    # Test the question
    print("\nTesting question: 'Does Tweety have wings?'")
    result = kb.query_knowledge("Does Tweety have wings?")
    print(f"Answer: {result['answer']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Confidence: {result['confidence']}")
    
    # Check what facts are in the knowledge base
    print(f"\nFacts in knowledge base: {len(kb.facts)}")
    for fact in kb.facts:
        print(f"  - {fact.text}")

if __name__ == "__main__":
    test_logical_fix()
