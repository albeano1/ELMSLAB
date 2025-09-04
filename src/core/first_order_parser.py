"""
First-Order Logic Natural Language Parser

This module extends the propositional logic parser to handle first-order logic
with quantifiers, predicates, and individual constants.
"""

import spacy
import re
from typing import Dict, Any, List, Tuple, Optional, Union
from src.models.first_order_logic import (
    FirstOrderFormula, PredicateFormula, QuantifiedFormula, FirstOrderNegation,
    FirstOrderConjunction, FirstOrderDisjunction, FirstOrderImplication,
    Predicate, Variable, Constant, VariableTerm, ConstantTerm,
    Quantifier, forall, exists, predicate, f_neg, f_conj, f_disj, f_impl
)


class FirstOrderLogicConverter:
    """Converts natural language to first-order logic formulas"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def normalize_identifier(self, text: str) -> str:
        """Convert text to a valid identifier"""
        # Remove articles, common auxiliary verbs, and normalize
        text = re.sub(r'\b(a|an|the|is|are|was|were|will be|can|do|does|did)\b', '', text, flags=re.IGNORECASE)
        text = text.strip()
        return re.sub(r'[^a-z0-9_]', '', text.lower().replace(" ", "_"))
    
    def extract_individual_constant(self, text: str) -> Constant:
        """Extract an individual constant from text (e.g., 'Socrates', 'John')"""
        # Capitalize first letter for proper names
        normalized = self.normalize_identifier(text)
        if normalized and normalized[0].islower():
            normalized = normalized.capitalize()
        return Constant(normalized)
    
    def extract_predicate_name(self, text: str) -> str:
        """Extract predicate name from text"""
        return self.normalize_identifier(text)
    
    def detect_quantifier_pattern(self, text: str) -> Optional[Tuple[Quantifier, str, str]]:
        """
        Detect quantifier patterns in text.
        Returns (quantifier, variable_description, scope) or None
        """
        text_lower = text.lower()
        
        # Universal quantifier patterns
        universal_patterns = [
            r'\ball\s+(\w+)\s+(?:are|is|can|do|will)\s+(.+)',  # "All humans are mortal"
            r'\bevery\s+(\w+)\s+(?:is|are|can|do|will)\s+(.+)',  # "Every student studies"
            r'\beach\s+(\w+)\s+(?:is|are|can|do|will)\s+(.+)',  # "Each bird flies"
            r'\bany\s+(\w+)\s+(?:is|are|can|do|will)\s+(.+)',  # "Any person can learn"
            r'\bevery\s+(\w+)\s+(.+)',  # "Every student studies" (more general)
        ]
        
        for pattern in universal_patterns:
            match = re.search(pattern, text_lower)
            if match:
                variable_desc = match.group(1)
                scope = match.group(2)
                return Quantifier.FORALL, variable_desc, scope
        
        # Existential quantifier patterns
        existential_patterns = [
            r'\bsome\s+(\w+)\s+(?:are|is|can|do|will)\s+(.+)',  # "Some birds cannot fly"
            r'\bthere\s+exists?\s+(?:a\s+)?(\w+)\s+(?:that|which|who)\s+(.+)',  # "There exists a student who studies"
            r'\bat\s+least\s+one\s+(\w+)\s+(?:is|are|can|do|will)\s+(.+)',  # "At least one student passed"
            r'\bsome\s+(\w+)\s+(.+)',  # "Some birds cannot fly" (more general)
        ]
        
        for pattern in existential_patterns:
            match = re.search(pattern, text_lower)
            if match:
                variable_desc = match.group(1)
                scope = match.group(2)
                return Quantifier.EXISTS, variable_desc, scope
        
        return None
    
    def parse_individual_statement(self, text: str) -> FirstOrderFormula:
        """Parse statements about individuals (e.g., 'Socrates is human')"""
        doc = self.nlp(text)
        
        # Find the subject and predicate
        subject = None
        predicate_parts = []
        has_negation = False
        
        for token in doc:
            if token.dep_ == "nsubj":  # Subject
                subject = token.text
            elif token.dep_ == "neg":  # Negation
                has_negation = True
            elif token.dep_ in ["acomp", "attr", "advmod"]:  # Predicate complement
                predicate_parts.append(token.text)
            elif token.dep_ == "ROOT" and token.lemma_ == "be":  # Copula
                continue
        
        if not subject or not predicate_parts:
            # Fallback: treat as a simple predicate
            return predicate(self.extract_predicate_name(text))
        
        # Create predicate: P(subject)
        pred_name = self.extract_predicate_name(" ".join(predicate_parts))
        subject_constant = self.extract_individual_constant(subject)
        
        formula = predicate(pred_name, ConstantTerm(subject_constant))
        
        if has_negation:
            return f_neg(formula)
        
        return formula
    
    def parse_conditional_statement(self, text: str) -> FirstOrderFormula:
        """Parse conditional statements (if...then)"""
        # Extract antecedent and consequent
        match = re.search(r'\bif\b\s*(.*?)\s*\bthen\b\s*(.*)', text, re.IGNORECASE)
        if not match:
            return self.parse_individual_statement(text)
        
        antecedent_text = match.group(1).strip()
        consequent_text = match.group(2).strip()
        
        antecedent = self.parse_individual_statement(antecedent_text)
        consequent = self.parse_individual_statement(consequent_text)
        
        return f_impl(antecedent, consequent)
    
    def parse_quantified_statement(self, text: str) -> FirstOrderFormula:
        """Parse quantified statements (All X are Y, Some X are Y)"""
        quantifier_info = self.detect_quantifier_pattern(text)
        if not quantifier_info:
            return self.parse_individual_statement(text)
        
        quantifier, variable_desc, scope = quantifier_info
        
        # Create variable for the quantified entity
        var_name = variable_desc[0].lower()  # Use first letter as variable name
        variable = Variable(var_name)
        
        # Parse the scope to create the predicate
        # For "All humans are mortal" -> scope is "mortal"
        # We need to create: ∀x(Human(x) → Mortal(x))
        
        # Extract predicate from scope and check for negation
        scope_doc = self.nlp(scope)
        scope_predicate_parts = []
        has_negation = False
        
        for token in scope_doc:
            if token.dep_ == "neg" or token.text.lower() in ["not", "no", "never", "cannot", "can't"]:
                has_negation = True
            elif token.dep_ in ["acomp", "attr", "advmod", "ROOT"] and token.pos_ in ["ADJ", "VERB", "NOUN"]:
                scope_predicate_parts.append(token.text)
        
        if not scope_predicate_parts:
            scope_predicate_parts = [scope]
        
        scope_pred_name = self.extract_predicate_name(" ".join(scope_predicate_parts))
        domain_pred_name = self.extract_predicate_name(variable_desc)
        
        # Create the quantified formula
        if quantifier == Quantifier.FORALL:
            # ∀x(Domain(x) → Scope(x))
            domain_pred = predicate(domain_pred_name, VariableTerm(variable))
            scope_pred = predicate(scope_pred_name, VariableTerm(variable))
            if has_negation:
                scope_pred = f_neg(scope_pred)
            inner_formula = f_impl(domain_pred, scope_pred)
            return forall(variable, inner_formula)
        else:  # EXISTS
            # ∃x(Domain(x) ∧ Scope(x))
            domain_pred = predicate(domain_pred_name, VariableTerm(variable))
            scope_pred = predicate(scope_pred_name, VariableTerm(variable))
            if has_negation:
                scope_pred = f_neg(scope_pred)
            inner_formula = f_conj([domain_pred, scope_pred])
            return exists(variable, inner_formula)
    
    def convert_text_to_first_order_logic(self, text: str) -> Dict[str, Any]:
        """Convert natural language text to first-order logic formula"""
        print(f"FOL Parser: Converting text: '{text}'")
        
        # Try different parsing strategies in order of specificity
        formula = None
        confidence = 0.5
        
        # 1. Check for quantifiers first
        if any(word in text.lower() for word in ['all', 'every', 'each', 'any', 'some', 'there exists', 'at least one']):
            formula = self.parse_quantified_statement(text)
            confidence = 0.8
            print(f"FOL Parser: Detected quantified statement")
        
        # 2. Check for conditionals
        elif re.search(r'\bif\b.*\bthen\b', text, re.IGNORECASE):
            formula = self.parse_conditional_statement(text)
            confidence = 0.8
            print(f"FOL Parser: Detected conditional statement")
        
        # 3. Default to individual statement
        else:
            formula = self.parse_individual_statement(text)
            confidence = 0.6
            print(f"FOL Parser: Parsed as individual statement")
        
        # Extract components for analysis
        variables = list(formula.variables()) if hasattr(formula, 'variables') else []
        constants = list(formula.constants()) if hasattr(formula, 'constants') else []
        predicates = list(formula.predicates()) if hasattr(formula, 'predicates') else []
        
        result = {
            "original_text": text,
            "first_order_formula": str(formula),
            "formula_type": type(formula).__name__,
            "confidence": confidence,
            "variables": [str(v) for v in variables],
            "constants": [str(c) for c in constants],
            "predicates": [str(p) for p in predicates],
            "semantic_analysis": {
                "quantifiers_detected": len([v for v in variables if any(q in str(formula) for q in ['∀', '∃'])]),
                "individual_constants": len(constants),
                "predicate_count": len(predicates),
                "parsing_strategy": "first_order_logic"
            }
        }
        
        print(f"FOL Parser: Result: {result['first_order_formula']}")
        return result


# Test the parser
if __name__ == "__main__":
    converter = FirstOrderLogicConverter()
    
    test_cases = [
        "All humans are mortal",
        "Socrates is human", 
        "Some birds cannot fly",
        "If it rains then the ground is wet",
        "Every student studies",
        "There exists a student who passed"
    ]
    
    for text in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: '{text}'")
        print('='*60)
        result = converter.convert_text_to_first_order_logic(text)
        print(f"Formula: {result['first_order_formula']}")
        print(f"Type: {result['formula_type']}")
        print(f"Variables: {result['variables']}")
        print(f"Constants: {result['constants']}")
        print(f"Predicates: {result['predicates']}")
