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
            "every": "□",          # Every (recurring)
            "daily": "□",          # Daily recurring
            "weekly": "□",         # Weekly recurring
            "monthly": "□",        # Monthly recurring
            
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
            "since": "S",          # Since operator
            
            # Until operators
            "until": "U",          # Until operator
            "before": "U",         # Until operator
            "unless": "W",         # Weak until operator
            
            # Duration and bounded time
            "within": "F[0,",      # Bounded future
            "for": "duration",     # Duration marker
            "during": "duration",  # Duration marker
        }
    
    def extract_subject(self, doc) -> str:
        """Extract the subject from a spaCy document."""
        for token in doc:
            if token.dep_ == "nsubj" or token.dep_ == "nsubjpass":
                subject = token.text.lower()
                # Improve subject representation
                if subject == "i":
                    return "speaker"  # Better representation for the pronoun "I"
                return subject
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
        temporal_phrases = []
        
        # Extract individual temporal words
        for token in doc:
            if token.text.lower() in self.temporal_operators:
                temporal_words.append(token.text.lower())
        
        # Extract temporal phrases (e.g., "every Monday", "within 2 hours")
        text_lower = doc.text.lower()
        
        # Check for complex temporal patterns
        if "every" in text_lower:
            # Extract the recurring pattern
            for token in doc:
                if token.text.lower() == "every" and token.i + 1 < len(doc):
                    next_token = doc[token.i + 1]
                    if next_token.pos_ == "NOUN":
                        return f"every {next_token.text.lower()}"
        
        if "within" in text_lower:
            # Extract duration for bounded time
            for token in doc:
                if token.text.lower() == "within" and token.i + 2 < len(doc):
                    duration_token = doc[token.i + 1]
                    unit_token = doc[token.i + 2]
                    if duration_token.pos_ == "NUM" and unit_token.pos_ == "NOUN":
                        return f"within {duration_token.text} {unit_token.text}"
        
        if "until" in text_lower:
            return "until"
        
        if "since" in text_lower:
            return "since"
        
        # Prioritize specific temporal words over general ones
        if "tomorrow" in temporal_words:
            return "tomorrow"
        elif "yesterday" in temporal_words:
            return "yesterday"
        elif "next" in temporal_words:
            return "next"
        elif "always" in temporal_words:
            return "always"
        elif "every" in temporal_words:
            return "every"
        elif temporal_words:
            return temporal_words[0]  # Return the first temporal word found
        
        return "unknown"
    
    def build_predicate(self, subject: str, action: str, destination: str = None) -> str:
        """Build a proper predicate from components."""
        # Normalize action to consistent form
        if action == "start":
            action = "starts"
        elif action == "stop":
            action = "stops"
        elif action == "go":
            action = "goes"
        elif action == "run":
            action = "runs"
        elif action == "wait":
            action = "waits"
        elif action == "study":
            action = "studies"
        elif action == "practice":
            action = "practices"
        elif action == "keep":
            action = "keeps"
        elif action == "try":
            action = "tries"
        elif action == "succeed":
            action = "succeeds"
        elif action == "fail":
            action = "fails"
        elif action == "press":
            action = "presses"
        elif action == "pass":
            action = "passes"
        elif action == "stay":
            action = "stays"
        elif action == "happen":
            action = "happens"
        elif action == "rise":
            action = "rises"
        
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
    
    def build_enhanced_predicate(self, text: str, doc) -> str:
        """
        Build enhanced predicate that can handle multiple actions and complex expressions
        """
        # Check for multiple actions (e.g., "study and practice")
        if " and " in text.lower():
            actions = []
            subjects = []
            
            # Split by "and" and parse each part
            parts = text.lower().split(" and ")
            for part in parts:
                part_doc = self.nlp(part.strip())
                subject = self.extract_subject(part_doc)
                action = self.extract_action(part_doc)
                destination = self.extract_destination(part_doc)
                
                if subject and action:
                    subjects.append(subject)
                    actions.append(self.build_predicate(subject, action, destination))
            
            # If all subjects are the same, combine actions
            if len(set(subjects)) == 1 and subjects[0]:
                return f"({' ∧ '.join(actions)})"
            else:
                return f"({' ∧ '.join(actions)})"
        
        # Check for "or" expressions
        elif " or " in text.lower():
            actions = []
            parts = text.lower().split(" or ")
            for part in parts:
                part_doc = self.nlp(part.strip())
                subject = self.extract_subject(part_doc)
                action = self.extract_action(part_doc)
                destination = self.extract_destination(part_doc)
                
                if subject and action:
                    actions.append(self.build_predicate(subject, action, destination))
            
            return f"({' ∨ '.join(actions)})"
        
        # Check for negation (e.g., "stops raining")
        elif any(word in text.lower() for word in ["stop", "stops", "not", "no"]):
            subject = self.extract_subject(doc)
            action = self.extract_action(doc)
            destination = self.extract_destination(doc)
            
            base_predicate = self.build_predicate(subject, action, destination)
            return f"¬{base_predicate}"
        
        # Default case
        else:
            subject = self.extract_subject(doc)
            action = self.extract_action(doc)
            destination = self.extract_destination(doc)
            return self.build_predicate(subject, action, destination)
    
    def parse_temporal_unless(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse 'X unless Y' as weak until (W operator)
        """
        text_lower = text.lower()
        
        # Check if "unless" is in the text
        if "unless" not in text_lower:
            return None
        
        # Split the sentence at "unless"
        parts = text.split("unless")
        
        if len(parts) != 2:
            return None
        
        # Part 1: What happens (before "unless")
        before_part = parts[0].strip()
        # Part 2: What stops it (after "unless")
        after_part = parts[1].strip()
        
        print(f"Enhanced Temporal Parser: Unless parsing - Before: '{before_part}', After: '{after_part}'")
        
        # Parse the before part
        before_doc = self.nlp(before_part)
        before_subject = self.extract_subject(before_doc)
        before_action = self.extract_action(before_doc)
        before_destination = self.extract_destination(before_doc)
        
        # Parse the after part
        after_doc = self.nlp(after_part)
        after_subject = self.extract_subject(after_doc)
        after_action = self.extract_action(after_doc)
        after_destination = self.extract_destination(after_doc)
        
        # Build formulas with enhanced parsing
        formula1 = self.build_enhanced_predicate(before_part, before_doc)
        formula2 = self.build_enhanced_predicate(after_part, after_doc)
        
        # Combine with W operator (weak until)
        temporal_formula = f"{formula1} W {formula2}"
        
        return {
            "original_text": text,
            "temporal_formula": temporal_formula,
            "confidence": 0.95,
            "reasoning": f"Weak 'unless' operator detected: '{before_part}' continues unless '{after_part}' happens",
            "components": {
                "temporal_operator": "unless",
                "left_formula": formula1,
                "right_formula": formula2,
                "operator": "W",
                "interpretation": f"'{before_part}' continues unless '{after_part}' happens (weak until)"
            },
            "alternative_representations": {
                "ltl": temporal_formula,
                "mtl": temporal_formula,
                "first_order": f"∃t1∃t2({formula1} ∧ {formula2} ∧ t1 ≤ t2)"
            }
        }
    
    def parse_temporal_until(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Correctly parse 'X until Y' as a binary operator
        """
        text_lower = text.lower()
        
        # Check if "until" is in the text
        if "until" not in text_lower:
            return None
        
        # Split the sentence at "until"
        parts = text.split("until")
        
        if len(parts) != 2:
            return None
        
        # Part 1: What happens first (before "until")
        before_part = parts[0].strip()
        # Part 2: What ends it (after "until")
        after_part = parts[1].strip()
        
        print(f"Enhanced Temporal Parser: Until parsing - Before: '{before_part}', After: '{after_part}'")
        
        # Parse the before part
        before_doc = self.nlp(before_part)
        before_subject = self.extract_subject(before_doc)
        before_action = self.extract_action(before_doc)
        before_destination = self.extract_destination(before_doc)
        
        # Parse the after part
        after_doc = self.nlp(after_part)
        after_subject = self.extract_subject(after_doc)
        after_action = self.extract_action(after_doc)
        after_destination = self.extract_destination(after_doc)
        
        # Build formulas with enhanced parsing
        formula1 = self.build_enhanced_predicate(before_part, before_doc)
        formula2 = self.build_enhanced_predicate(after_part, after_doc)
        
        # Combine with U operator
        temporal_formula = f"{formula1} U {formula2}"
        
        return {
            "original_text": text,
            "temporal_formula": temporal_formula,
            "confidence": 0.95,
            "reasoning": f"Binary 'until' operator detected: '{before_part}' continues until '{after_part}' happens",
            "components": {
                "temporal_operator": "until",
                "left_formula": formula1,
                "right_formula": formula2,
                "operator": "U",
                "interpretation": f"'{before_part}' continues until '{after_part}' happens"
            },
            "alternative_representations": {
                "ltl": temporal_formula,
                "mtl": temporal_formula,
                "first_order": f"∃t1∃t2({formula1} ∧ {formula2} ∧ t1 < t2)"
            }
        }
    
    def parse_temporal_now(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse 'now' queries to generate present tense formulas."""
        if not self.nlp:
            return None
            
        doc = self.nlp(text)
        subject = self.extract_subject(doc)
        action = self.extract_action(doc)
        destination = self.extract_destination(doc)
        
        if not subject or not action:
            return None
        
        # Build predicate for "now" - use present continuous form
        if action in ["study", "studying"]:
            predicate = f"studying_now({subject})"
        elif action in ["wait", "waiting"]:
            predicate = f"waiting_now({subject})"
        elif action in ["work", "working"]:
            predicate = f"working_now({subject})"
        elif action in ["run", "running"]:
            predicate = f"running_now({subject})"
        elif action in ["go", "going"]:
            predicate = f"going_now({subject})"
        elif action in ["try", "trying"]:
            predicate = f"trying_now({subject})"
        elif action in ["practice", "practicing"]:
            predicate = f"practicing_now({subject})"
        else:
            # Generic present continuous form
            predicate = f"{action}_now({subject})"
        
        return {
            "original_text": text,
            "temporal_formula": predicate,
            "confidence": 0.95,
            "reasoning": f"Present tense 'now' query detected: {predicate}",
            "components": {
                "temporal_operator": "now",
                "predicate": predicate,
                "subject": subject,
                "action": action,
                "interpretation": f"Current state: {predicate}"
            },
            "alternative_representations": {
                "ltl": predicate,
                "mtl": predicate,
                "first_order": f"∃t({predicate} ∧ t = now)"
            }
        }
    
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
        
        # Check for "now" queries first (present tense)
        if "now" in text.lower() or "currently" in text.lower():
            now_result = self.parse_temporal_now(text)
            if now_result:
                print(f"Enhanced Temporal Parser: Detected 'now' pattern: {now_result}")
                return now_result
        
        # Check for "unless" pattern first (weak until)
        unless_result = self.parse_temporal_unless(text)
        if unless_result:
            print(f"Enhanced Temporal Parser: Detected 'unless' pattern: {unless_result}")
            return unless_result
        
        # Check for "until" pattern (binary operator)
        until_result = self.parse_temporal_until(text)
        if until_result:
            print(f"Enhanced Temporal Parser: Detected 'until' pattern: {until_result}")
            return until_result
        
        # Parse with spaCy for other temporal patterns
        doc = self.nlp(text)
        
        # Extract components
        subject = self.extract_subject(doc)
        action = self.extract_action(doc)
        destination = self.extract_destination(doc)
        temporal_expr = self.extract_temporal_expression(doc)
        
        print(f"Enhanced Temporal Parser: Subject='{subject}', Action='{action}', Destination='{destination}', Temporal='{temporal_expr}'")
        
        # Build base predicate
        base_predicate = self.build_predicate(subject, action, destination)
        
        # Determine temporal operator with enhanced pattern matching
        temporal_operator = "◊"  # Default to eventually
        operator_name = "eventually"
        
        if temporal_expr != "unknown":
            # Handle complex temporal expressions
            if temporal_expr.startswith("every "):
                temporal_operator = "□"
                operator_name = temporal_expr
            elif temporal_expr.startswith("within "):
                # Extract duration for bounded time
                duration_part = temporal_expr.replace("within ", "")
                temporal_operator = f"F[0,{duration_part}]"
                operator_name = temporal_expr
            elif temporal_expr == "until":
                temporal_operator = "U"
                operator_name = "until"
            elif temporal_expr == "since":
                temporal_operator = "S"
                operator_name = "since"
            else:
                # Handle simple temporal words
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
