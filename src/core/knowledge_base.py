"""
Knowledge Base System

This module provides a persistent knowledge base for storing and querying facts,
with inference capabilities for reasoning about stored knowledge.
"""

import json
import os
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime

from ..models.first_order_logic import FirstOrderFormula, PredicateFormula, QuantifiedFormula, Quantifier
from ..models.propositional_logic import Formula, AtomicFormula, Atom, Implication, Conjunction
from .first_order_parser import FirstOrderLogicConverter
from .temporal_parser import TemporalLogicConverter

@dataclass
class KnowledgeFact:
    """Represents a fact in the knowledge base."""
    id: str
    text: str
    formula: str
    formula_type: str  # "propositional", "first_order", "temporal"
    confidence: float
    timestamp: str
    source: str = "user_input"

class KnowledgeBase:
    """Persistent knowledge base with inference capabilities."""
    
    def __init__(self, storage_file: str = "knowledge_base.json"):
        self.storage_file = storage_file
        self.facts: List[KnowledgeFact] = []
        self.fol_converter = FirstOrderLogicConverter()
        self.temporal_converter = TemporalLogicConverter()
        self.load_knowledge()
    
    def add_fact(self, fact_text: str, source: str = "user_input") -> Dict[str, Any]:
        """Add a fact to the knowledge base."""
        print(f"Knowledge Base: Adding fact: '{fact_text}'")
        
        # Determine logic type and convert
        logic_type = self._detect_logic_type(fact_text)
        
        if logic_type == "temporal":
            result = self.temporal_converter.convert_text_to_temporal_logic(fact_text)
            formula = result.get("temporal_formula", fact_text)
            formula_type = "temporal"
        elif logic_type == "first_order":
            result = self.fol_converter.convert_text_to_first_order_logic(fact_text)
            formula = result.get("first_order_formula", fact_text)
            formula_type = "first_order"
        else:
            # Simple propositional logic
            formula = self._simple_propositional_convert(fact_text)
            formula_type = "propositional"
        
        # Create knowledge fact
        fact_id = f"fact_{len(self.facts) + 1}_{int(datetime.now().timestamp())}"
        fact = KnowledgeFact(
            id=fact_id,
            text=fact_text,
            formula=formula,
            formula_type=formula_type,
            confidence=0.8,
            timestamp=datetime.now().isoformat(),
            source=source
        )
        
        self.facts.append(fact)
        self.save_knowledge()
        
        return {
            "success": True,
            "fact_id": fact_id,
            "fact": asdict(fact),
            "message": f"Added fact: {fact_text} → {formula}"
        }
    
    def query_knowledge(self, question: str) -> Dict[str, Any]:
        """Query the knowledge base with inference."""
        print(f"Knowledge Base: Querying: '{question}'")
        
        # Determine logic type of the question
        logic_type = self._detect_logic_type(question)
        
        # Find relevant facts
        relevant_facts = self._find_relevant_facts(question)
        
        if not relevant_facts:
            return {
                "success": False,
                "answer": "No relevant facts found in knowledge base.",
                "reasoning": "No facts match the query.",
                "relevant_facts": []
            }
        
        # Perform inference
        inference_result = self._perform_inference(question, relevant_facts, logic_type)
        
        return {
            "success": True,
            "question": question,
            "answer": inference_result["answer"],
            "reasoning": inference_result["reasoning"],
            "relevant_facts": [asdict(fact) for fact in relevant_facts],
            "confidence": inference_result["confidence"]
        }
    
    def _detect_logic_type(self, text: str) -> str:
        """Detect the logic type of a text."""
        text_lower = text.lower()
        # Check for temporal indicators
        temporal_indicators = [
            'yesterday', 'tomorrow', 'last', 'next', 'ago', 'will', 'shall',
            'was', 'were', 'had', 'did', 'going to', 'gonna', 'then',
            'afterwards', 'after that', 'subsequently', 'immediately',
            'always', 'forever', 'constantly', 'eventually', 'someday'
        ]
        
        if any(indicator in text_lower for indicator in temporal_indicators):
            return "temporal"
        
        # Check for first-order indicators
        fol_indicators = [
            'all', 'every', 'each', 'any', 'some', 'there exists', 'at least one'
        ]
        
        if any(indicator in text_lower for indicator in fol_indicators):
            return "first_order"
        
        # Check for proper names (capitalized words)
        words = text.split()
        for word in words:
            if word[0].isupper() and word.isalpha() and len(word) > 2:
                common_words = {'the', 'this', 'that', 'these', 'those', 'there', 'here'}
                if word.lower() not in common_words:
                    return "first_order"
        
        # Check for property questions (Do/Does X have Y?, Is X a Y?, etc.)
        property_question_patterns = [
            'do ', 'does ', 'is ', 'are ', 'can ', 'could ', 'should ', 'would ',
            'have ', 'has ', 'had ', 'a ', 'an '
        ]
        
        if any(pattern in text_lower for pattern in property_question_patterns):
            return "first_order"
        return "propositional"
    
    def _simple_propositional_convert(self, text: str) -> str:
        """Simple propositional logic conversion."""
        # Basic conversion - replace spaces with underscores and lowercase
        formula = text.lower().replace(' ', '_').replace(',', '').replace('.', '')
        return formula
    
    def _clean_words(self, words: List[str]) -> List[str]:
        """Clean words by removing punctuation and normalizing."""
        import string
        cleaned = []
        for word in words:
            # Remove punctuation from both ends and normalize apostrophes
            cleaned_word = word.strip(string.punctuation)
            # Handle apostrophes - remove them and keep the base word
            cleaned_word = cleaned_word.replace("'", "").replace("'", "")
            if cleaned_word:  # Only add non-empty words
                cleaned.append(cleaned_word)
        return cleaned
    
    def _normalize_verbs(self, words: List[str]) -> List[str]:
        """Normalize verb forms to base form."""
        normalized = []
        for word in words:
            # Simple verb normalization
            if word.endswith('s') and len(word) > 3:
                # Remove 's' from verbs (likes -> like, has -> ha, etc.)
                base_form = word[:-1]
                # Keep the base form if it's a reasonable word
                if len(base_form) >= 2:
                    normalized.append(base_form)
                else:
                    normalized.append(word)
            else:
                normalized.append(word)
        return normalized
    
    def _find_relevant_facts(self, question: str) -> List[KnowledgeFact]:
        """Find facts relevant to the question."""
        question_lower = question.lower()
        relevant_facts = []
        
        # Enhanced keyword matching with scoring
        for fact in self.facts:
            fact_lower = fact.text.lower()
            
            # Use the same cleaning and normalization as inference
            question_words = set(self._normalize_verbs(self._clean_words(question_lower.split())))
            fact_words = set(self._normalize_verbs(self._clean_words(fact_lower.split())))
            
            # Remove common words
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'all', 'every', 'each', 'some', 'what', 'which', 'how', 'why', 'when', 'where', 'who'}
            question_words -= common_words
            fact_words -= common_words
            
            # Calculate overlap
            overlap = len(question_words.intersection(fact_words))
            
            # Also check for partial word matches and synonyms
            if overlap > 0:
                relevant_facts.append(fact)
            else:
                # Check for partial matches (e.g., "bird" matches "birds")
                for q_word in question_words:
                    for f_word in fact_words:
                        if (q_word in f_word or f_word in q_word) and len(q_word) > 2 and len(f_word) > 2:
                            relevant_facts.append(fact)
                            break
                    if fact in relevant_facts:
                        break
        
        return relevant_facts
    
    def _perform_inference(self, question: str, facts: List[KnowledgeFact], logic_type: str) -> Dict[str, Any]:
        """Perform inference on the question using relevant facts."""
        
        if logic_type == "first_order":
            return self._first_order_inference(question, facts)
        elif logic_type == "temporal":
            return self._temporal_inference(question, facts)
        else:
            return self._propositional_inference(question, facts)
    
    def _first_order_inference(self, question: str, facts: List[KnowledgeFact]) -> Dict[str, Any]:
        """Perform first-order logic inference."""
        question_lower = question.lower()
        
        # Check for "Do/Does X have Y?" pattern
        if ("do" in question_lower or "does" in question_lower) and "have" in question_lower:
            # Extract key terms from question (clean punctuation and normalize verbs)
            question_words = set(self._normalize_verbs(self._clean_words(question_lower.split())))
            question_words -= {'do', 'does', 'have', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
            
            # Extract the subject (first word after "does/do") and property (last word before "?")
            words = question_lower.split()
            subject = None
            property_word = None
            
            for i, word in enumerate(words):
                if word in ['do', 'does'] and i + 1 < len(words):
                    subject = words[i + 1]
                if word.endswith('?') or word.endswith('.'):
                    property_word = word.rstrip('?.').strip()
            
            # Look for direct facts first
            for fact in facts:
                fact_lower = fact.text.lower()
                fact_words = set(self._normalize_verbs(self._clean_words(fact_lower.split())))
                fact_words -= {'all', 'every', 'each', 'some', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
                
                # Direct match: "X has Y" or "X have Y"
                if subject and property_word:
                    if (subject in fact_lower and property_word in fact_lower and 
                        ('has' in fact_lower or 'have' in fact_lower)):
                        return {
                            "answer": "Yes",
                            "reasoning": f"Based on the fact: {fact.text}",
                            "confidence": 0.9
                        }
            
            # Look for universal statements that require additional facts
            for fact in facts:
                fact_lower = fact.text.lower()
                fact_words = set(self._normalize_verbs(self._clean_words(fact_lower.split())))
                fact_words -= {'all', 'every', 'each', 'some', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
                
                # Universal statement: "All X have Y" - need to check if subject is X
                if ('all' in fact_lower or 'every' in fact_lower) and 'have' in fact_lower:
                    # Extract the category from "All X have Y"
                    fact_words_list = fact_lower.split()
                    category = None
                    for i, word in enumerate(fact_words_list):
                        if word in ['all', 'every'] and i + 1 < len(fact_words_list):
                            category = fact_words_list[i + 1]
                            break
                    
                    # Check if we have a fact stating that the subject belongs to this category
                    if subject and category:
                        for other_fact in facts:
                            other_fact_lower = other_fact.text.lower()
                            
                            # Check for both singular and plural forms
                            category_singular = category.rstrip('s') if category.endswith('s') else category
                            category_plural = category + 's' if not category.endswith('s') else category
                            
                            if (subject in other_fact_lower and 
                                (category in other_fact_lower or category_singular in other_fact_lower or category_plural in other_fact_lower) and 
                                ('is' in other_fact_lower or 'are' in other_fact_lower)):
                                return {
                                    "answer": "Yes",
                                    "reasoning": f"Based on the facts: {other_fact.text} and {fact.text}",
                                    "confidence": 0.8
                                }
            
            # If no logical chain found, return unknown
            return {
                "answer": "Unknown",
                "reasoning": "Cannot determine answer without knowing if the subject belongs to the relevant category.",
                "confidence": 0.3
            }
        
        # Check for "Do/Does X like Y?" pattern (only if not already handled by "have" pattern)
        if ("do" in question_lower or "does" in question_lower) and "like" in question_lower and "have" not in question_lower:
            # Extract key terms from question (clean punctuation and normalize verbs)
            question_words = set(self._normalize_verbs(self._clean_words(question_lower.split())))
            question_words -= {'do', 'does', 'like', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
            
            for fact in facts:
                fact_lower = fact.text.lower()
                fact_words = set(self._normalize_verbs(self._clean_words(fact_lower.split())))
                fact_words -= {'all', 'every', 'each', 'some', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
                
                # Check if there's significant word overlap
                overlap = len(question_words.intersection(fact_words))
                if overlap >= 2:  # Need at least 2 words to match for this pattern
                    return {
                        "answer": "Yes",
                        "reasoning": f"Based on the fact: {fact.text}",
                        "confidence": 0.8
                    }
        
        # Check for "Is/Are X a Y?" pattern
        if ("is" in question_lower or "are" in question_lower) and ("a " in question_lower or "an " in question_lower):
            # Extract the subject and category from the question
            words = question_lower.split()
            subject = None
            category = None
            
            # Find subject and category from "Is/Are X a Y?" pattern
            # For "Is a Dino a bird?", we want subject="Dino" and category="bird"
            for i, word in enumerate(words):
                if word in ['is', 'are'] and i + 1 < len(words):
                    # Subject is the word after "is/are", but skip articles
                    next_word = words[i + 1]
                    if next_word in ['a', 'an', 'the'] and i + 2 < len(words):
                        subject = words[i + 2]
                    else:
                        subject = next_word
                if word in ['a', 'an'] and i + 1 < len(words):
                    # Category is the word after "a/an", but clean punctuation
                    category = words[i + 1].rstrip('?.,!')
            
            # Look for direct matches: "X is a Y" or "X are Y"
            for fact in facts:
                fact_lower = fact.text.lower()
                if subject and category:
                    # Check for exact pattern match: "subject is a category" or "subject are category"
                    if (subject in fact_lower and category in fact_lower and 
                        ('is' in fact_lower or 'are' in fact_lower)):
                        return {
                            "answer": "Yes",
                            "reasoning": f"Based on the fact: {fact.text}",
                            "confidence": 0.9
                        }
            
            # If no direct match found, return unknown
            return {
                "answer": "Unknown",
                "reasoning": "No direct fact found stating the relationship between the subject and category.",
                "confidence": 0.3
            }
        
        # Check for "Are X Y?" pattern (without articles)
        if "are" in question_lower:
            question_words = set(self._clean_words(question_lower.split()))
            question_words -= {'are', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
            
            for fact in facts:
                fact_lower = fact.text.lower()
                fact_words = set(self._clean_words(fact_lower.split()))
                fact_words -= {'all', 'every', 'each', 'some', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
                
                overlap = len(question_words.intersection(fact_words))
                if overlap >= 2:  # Need at least 2 words to match for this pattern
                    return {
                        "answer": "Yes",
                        "reasoning": f"Based on the fact: {fact.text}",
                        "confidence": 0.8
                    }
        
        # Check for general property questions
        if any(word in question_lower for word in ['what', 'which', 'how', 'why']):
            question_words = set(self._clean_words(question_lower.split()))
            question_words -= {'what', 'which', 'how', 'why', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
            
            for fact in facts:
                fact_lower = fact.text.lower()
                fact_words = set(self._clean_words(fact_lower.split()))
                fact_words -= {'all', 'every', 'each', 'some', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
                
                overlap = len(question_words.intersection(fact_words))
                if overlap >= 1:
                    return {
                        "answer": "Yes",
                        "reasoning": f"Based on the fact: {fact.text}",
                        "confidence": 0.7
                    }
        
        return {
            "answer": "Unknown",
            "reasoning": "Cannot determine answer from available facts.",
            "confidence": 0.3
        }
    
    def _temporal_inference(self, question: str, facts: List[KnowledgeFact]) -> Dict[str, Any]:
        """Perform temporal logic inference."""
        # Simple temporal reasoning
        question_lower = question.lower()
        
        for fact in facts:
            fact_lower = fact.text.lower()
            
            # Check for temporal relationships
            if any(word in fact_lower for word in question_lower.split()):
                return {
                    "answer": "Yes",
                    "reasoning": f"Based on the temporal fact: {fact.text}",
                    "confidence": 0.7
                }
        
        return {
            "answer": "Unknown",
            "reasoning": "Cannot determine temporal answer from available facts.",
            "confidence": 0.3
        }
    
    def _propositional_inference(self, question: str, facts: List[KnowledgeFact]) -> Dict[str, Any]:
        """Perform propositional logic inference."""
        # Simple propositional reasoning
        question_lower = question.lower()
        
        for fact in facts:
            fact_lower = fact.text.lower()
            
            # Direct match
            if question_lower in fact_lower or fact_lower in question_lower:
                return {
                    "answer": "Yes",
                    "reasoning": f"Direct match with fact: {fact.text}",
                    "confidence": 0.9
                }
        
        return {
            "answer": "Unknown",
            "reasoning": "Cannot determine answer from available facts.",
            "confidence": 0.3
        }
    
    def get_all_facts(self) -> List[Dict[str, Any]]:
        """Get all facts in the knowledge base."""
        return [asdict(fact) for fact in self.facts]
    
    def clear_knowledge(self) -> Dict[str, Any]:
        """Clear all facts from the knowledge base."""
        self.facts = []
        self.save_knowledge()
        return {
            "success": True,
            "message": "Knowledge base cleared."
        }
    
    def save_knowledge(self):
        """Save knowledge base to file."""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump([asdict(fact) for fact in self.facts], f, indent=2)
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
    
    def load_knowledge(self):
        """Load knowledge base from file."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    facts_data = json.load(f)
                    self.facts = [KnowledgeFact(**fact_data) for fact_data in facts_data]
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            self.facts = []
