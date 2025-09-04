#!/usr/bin/env python3
"""
Debug script for knowledge base issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.knowledge_base import KnowledgeBase

def test_ethan_cheese():
    """Test the Ethan cheese scenario"""
    kb = KnowledgeBase()
    
    # Add the fact
    print("Adding fact: 'Ethan like's cheese'")
    result = kb.add_fact("Ethan like's cheese")
    print(f"Result: {result}")
    
    # Test the question
    print("\nQuerying: 'Does Ethan like cheese?'")
    result = kb.query_knowledge("Does Ethan like cheese?")
    print(f"Result: {result}")
    
    # Test word matching manually
    print("\n=== Manual Word Matching Test ===")
    question = "Does Ethan like cheese?"
    fact = "Ethan like's cheese"
    
    # Test logic type detection
    logic_type = kb._detect_logic_type(question)
    print(f"Logic type: {logic_type}")
    
    # Test word cleaning and normalization
    question_words = set(kb._normalize_verbs(kb._clean_words(question.lower().split())))
    question_words -= {'do', 'does', 'like', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
    print(f"Question words: {question_words}")
    
    fact_words = set(kb._normalize_verbs(kb._clean_words(fact.lower().split())))
    fact_words -= {'all', 'every', 'each', 'some', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
    print(f"Fact words: {fact_words}")
    
    overlap = len(question_words.intersection(fact_words))
    print(f"Overlap: {overlap}")
    print(f"Overlap words: {question_words.intersection(fact_words)}")
    print(f"Should work: {overlap >= 2}")

if __name__ == "__main__":
    test_ethan_cheese()
