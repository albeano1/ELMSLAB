"""
Enhanced Temporal Logic Parser

This module provides comprehensive natural language parsing for temporal logic statements,
converting time-related natural language into formal temporal logic formulas with proper
component extraction (subject, action, destination, temporal operators).
"""

import re
import spacy
from typing import Dict, Any, Optional, Tuple, List

from ..models.temporal_logic import (
    PastFormula, FutureFormula, NextFormula, AlwaysFormula, EventuallyFormula,
    past, future, next_formula, always, eventually
)
from ..models.propositional_logic import AtomicFormula, Atom, Implication, Conjunction

class TemporalLogicConverter:
    """Enhanced converter for natural language temporal statements to temporal logic formulas."""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Temporal operator mappings
        self.temporal_operators = {
            # Future operators
            "tomorrow": "◯",        # Next operator
            "next": "◯",           # Next operator
            "eventually": "◊",     # Future operator
            "will": "◊",           # Future operator
            "shall": "◊",          # Future operator
            "going to": "◊",       # Future operator
            "gonna": "◊",          # Future operator
            "someday": "◊",        # Future operator
            "always": "□",         # Always operator
            "forever": "□",        # Always operator
            "constantly": "□",     # Always operator
            
            # Past operators
            "yesterday": "●",      # Previous operator
            "last": "●",           # Previous operator
            "ago": "●",            # Previous operator
            "was": "●",            # Previous operator
            "were": "●",           # Previous operator
            "had": "●",            # Previous operator
            "did": "●",            # Previous operator
            "once": "◈",           # Once operator
            "formerly": "◈",       # Once operator
            
            # Until operators
            "until": "U",          # Until operator
            "before": "U",         # Until operator
            "unless": "W",         # Weak until operator
        }
    
    def extract_subject(self, doc) -> str:
        """Extract the subject from a spaCy document."""
        for token in doc:
            if token.dep_ == "nsubj" or token.dep_ == "nsubjpass":
                return token.text.lower()
        return "unknown"
    
    def extract_action(self, doc) -> str:
        """Extract the main action/verb from a spaCy document."""
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return token.lemma_.lower()
        return "unknown"
    
    def extract_destination(self, doc) -> str:
        """Extract destination or object from a spaCy document."""
        # Look for prepositional objects (e.g., "to school")
        for token in doc:
            if token.dep_ == "pobj" and token.head.dep_ == "prep":
                return token.text.lower()
        
        # Look for direct objects
        for token in doc:
            if token.dep_ == "dobj":
                return token.text.lower()
        
        # Look for nouns that aren't the subject
        for token in doc:
            if token.pos_ == "NOUN" and token.dep_ not in ["nsubj", "nsubjpass"]:
                return token.text.lower()
        
        return "unknown"
    
    def extract_temporal_expression(self, doc) -> str:
        """Extract temporal expressions from a spaCy document."""
        temporal_words = []
        for token in doc:
            if token.text.lower() in self.temporal_operators:
                temporal_words.append(token.text.lower())
        
        # Prioritize specific temporal words over general ones
        if "tomorrow" in temporal_words:
            return "tomorrow"
        elif "yesterday" in temporal_words:
            return "yesterday"
        elif "next" in temporal_words:
            return "next"
        elif "always" in temporal_words:
            return "always"
        elif temporal_words:
            return temporal_words[0]  # Return the first temporal word found
        
        return "unknown"
    
    def build_predicate(self, subject: str, action: str, destination: str = None) -> str:
        """Build a proper predicate from components."""
        if destination and destination != "unknown":
            return f"{action}_to_{destination}({subject})"
        else:
            return f"{action}({subject})"
    
    def detect_temporal_pattern(self, text: str) -> Optional[Tuple[str, str]]:
        """Detect temporal patterns in natural language."""
        text_lower = text.lower()
        
        # Past tense patterns
        past_patterns = [
            r'\b(yesterday|last|ago|was|were|had|did)\b',
            r'\b(it|he|she|they)\s+(was|were|had|did)\b',
            r'\b(rained|snowed|happened)\s+(yesterday|last|ago)\b'
        ]
        
        # Future tense patterns  
        future_patterns = [
            r'\b(tomorrow|next|will|shall|going to|gonna)\b',
            r'\b(it|he|she|they)\s+(will|shall|is going to|are going to)\b',
            r'\b(rain|snow|happen)\s+(tomorrow|next|will)\b'
        ]
        
        # Next/immediate future patterns
        next_patterns = [
            r'\b(next|then|afterwards|after that|subsequently)\b',
            r'\b(immediately|right after|then)\b'
        ]
        
        # Always patterns
        always_patterns = [
            r'\b(always|forever|constantly|perpetually)\b',
            r'\b(every|all the time|continuously)\b'
        ]
        
        # Eventually patterns
        eventually_patterns = [
            r'\b(eventually|someday|sometime|one day|in the future)\b',
            r'\b(will happen|will occur|will be)\b'
        ]
        
        for pattern in past_patterns:
            if re.search(pattern, text_lower):
                return ("past", self._extract_base_statement(text, pattern))
        
        for pattern in future_patterns:
            if re.search(pattern, text_lower):
                return ("future", self._extract_base_statement(text, pattern))
        
        for pattern in next_patterns:
            if re.search(pattern, text_lower):
                return ("next", self._extract_base_statement(text, pattern))
        
        for pattern in always_patterns:
            if re.search(pattern, text_lower):
                return ("always", self._extract_base_statement(text, pattern))
        
        for pattern in eventually_patterns:
            if re.search(pattern, text_lower):
                return ("eventually", self._extract_base_statement(text, pattern))
        
        return None
    
    def _extract_base_statement(self, text: str, pattern: str) -> str:
        """Extract the base statement from temporal text."""
        # Remove temporal markers to get the core statement
        text_lower = text.lower()
        
        # Remove common temporal words
        temporal_words = [
            'yesterday', 'tomorrow', 'last', 'next', 'ago', 'will', 'shall',
            'was', 'were', 'had', 'did', 'going to', 'gonna', 'then',
            'afterwards', 'after that', 'subsequently', 'immediately',
            'right after', 'always', 'forever', 'constantly', 'eventually',
            'someday', 'sometime', 'one day', 'in the future'
        ]
        
        for word in temporal_words:
            text_lower = text_lower.replace(word, '')
        
        # Clean up extra spaces
        text_lower = re.sub(r'\s+', ' ', text_lower).strip()
        
        return text_lower
    
    def convert_text_to_temporal_logic(self, text: str) -> Dict[str, Any]:
        """Convert natural language text to enhanced temporal logic formula."""
        print(f"Enhanced Temporal Parser: Converting text: '{text}'")
        
        if not self.nlp:
            return {
                "original_text": text,
                "temporal_formula": "Error: spaCy model not available",
                "confidence": 0.0,
                "error": "spaCy model not available"
            }
        
        # Parse with spaCy
        doc = self.nlp(text)
        
        # Extract components
        subject = self.extract_subject(doc)
        action = self.extract_action(doc)
        destination = self.extract_destination(doc)
        temporal_expr = self.extract_temporal_expression(doc)
        
        print(f"Enhanced Temporal Parser: Subject='{subject}', Action='{action}', Destination='{destination}', Temporal='{temporal_expr}'")
        
        # Build base predicate
        base_predicate = self.build_predicate(subject, action, destination)
        
        # Determine temporal operator
        temporal_operator = "◊"  # Default to eventually
        operator_name = "eventually"
        
        if temporal_expr != "unknown":
            for word in temporal_expr.split():
                if word in self.temporal_operators:
                    temporal_operator = self.temporal_operators[word]
                    operator_name = word
                    break
        
        # Build temporal logic formula
        temporal_formula = f"{temporal_operator}({base_predicate})"
        
        # Alternative representations
        alternatives = {
            "ltl": temporal_formula,
            "mtl": f"F[1d,1d]({base_predicate})" if temporal_operator == "◯" else temporal_formula,
            "first_order": f"∃t({temporal_expr}(t) ∧ at(t, {base_predicate}))" if temporal_expr != "unknown" else temporal_formula
        }
        
        return {
            "original_text": text,
            "temporal_formula": temporal_formula,
            "confidence": 0.9,
            "reasoning": f"Enhanced temporal parsing with component extraction",
            "components": {
                "temporal_operator": operator_name,
                "predicate": base_predicate,
                "subject": subject,
                "action": action,
                "destination": destination,
                "time_reference": temporal_expr
            },
            "alternative_representations": alternatives
        }
