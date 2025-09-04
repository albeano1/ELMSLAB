#!/usr/bin/env python3
"""
Debug script for logical reasoning issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.knowledge_base import KnowledgeBase

def test_logical_reasoning():
    """Test the logical reasoning fix"""
    kb = KnowledgeBase()
    
    # Clear any existing facts
    kb.clear_knowledge()
    
    # Add the facts
    print("Adding facts:")
    result1 = kb.add_fact("All birds have wings")
    print(f"1. {result1['message']}")
    
    result2 = kb.add_fact("Tweety is a bird")
    print(f"2. {result2['message']}")
    
    # Test the question
    print("\nTesting question: 'Does Tweety have wings?'")
    result = kb.query_knowledge("Does Tweety have wings?")
    print(f"Answer: {result['answer']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Confidence: {result['confidence']}")
    
    # Test manual extraction
    print("\n=== Manual Extraction Test ===")
    question = "Does Tweety have wings?"
    question_lower = question.lower()
    
    # Extract subject and property
    words = question_lower.split()
    subject = None
    property_word = None
    
    for i, word in enumerate(words):
        if word in ['do', 'does'] and i + 1 < len(words):
            subject = words[i + 1]
        if word.endswith('?') or word.endswith('.'):
            property_word = word.rstrip('?.').strip()
    
    print(f"Question: {question}")
    print(f"Subject: '{subject}'")
    print(f"Property: '{property_word}'")
    
    # Test universal statement extraction
    fact = "All birds have wings"
    fact_lower = fact.lower()
    fact_words_list = fact_lower.split()
    category = None
    for i, word in enumerate(fact_words_list):
        if word in ['all', 'every'] and i + 1 < len(fact_words_list):
            category = fact_words_list[i + 1]
            break
    
    print(f"Universal statement: {fact}")
    print(f"Category: '{category}'")
    
    # Test category membership
    membership_fact = "Tweety is a bird"
    membership_fact_lower = membership_fact.lower()
    print(f"Membership fact: {membership_fact}")
    print(f"Subject in fact: {subject in membership_fact_lower}")
    print(f"Category in fact: {category in membership_fact_lower}")
    print(f"Has 'is': {'is' in membership_fact_lower}")
    print(f"Should match: {subject in membership_fact_lower and category in membership_fact_lower and 'is' in membership_fact_lower}")

if __name__ == "__main__":
    test_logical_reasoning()
