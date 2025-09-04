#!/usr/bin/env python3
"""
Test negation parsing directly
"""

import spacy
from typing import Tuple
from abc import ABC, abstractmethod

class Atom:
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Atom) and self.name == other.name
    
    def __hash__(self) -> int:
        return hash(self.name)

class Formula(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def atoms(self) -> set[Atom]:
        pass

class AtomicFormula(Formula):
    def __init__(self, atom: Atom):
        self.atom = atom
    
    def __str__(self) -> str:
        return str(self.atom)
    
    def atoms(self) -> set[Atom]:
        return {self.atom}

class Negation(Formula):
    def __init__(self, formula: Formula):
        self.formula = formula
    
    def __str__(self) -> str:
        if isinstance(self.formula, AtomicFormula):
            return f"¬{self.formula}"
        else:
            return f"¬({self.formula})"
    
    def atoms(self) -> set[Atom]:
        return self.formula.atoms()

def test_negation_parsing(text: str):
    """Test negation parsing with debug output"""
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        print(f"Testing: '{text}'")
        print("-" * 40)
        
        # Debug spaCy output
        for token in doc:
            print(f"{token.text:12} {token.dep_:10} {token.pos_:8} head: {token.head.text}")
        
        # Check for negation
        has_negation = False
        negated_token = None
        
        for token in doc:
            if token.dep_ == "neg":
                has_negation = True
                negated_token = token
                print(f"Found negation: '{token.text}' negating '{token.head.text}'")
                break
        
        # Find the main verb/predicate (ROOT)
        root = None
        for token in doc:
            if token.dep_ == "ROOT":
                root = token
                break
        
        if not root:
            print("No ROOT found!")
            return None
        
        print(f"Root token: '{root.text}' (lemma: '{root.lemma_}')")
        
        # Extract the base proposition
        base_prop = extract_base_proposition(root)
        print(f"Base proposition: '{base_prop}'")
        
        if has_negation:
            formula = Negation(AtomicFormula(Atom(base_prop)))
            print(f"Result: {formula}")
            return formula
        else:
            formula = AtomicFormula(Atom(base_prop))
            print(f"Result: {formula}")
            return formula
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_base_proposition(root_token) -> str:
    """Extract the base proposition from the root token"""
    # Handle copula (is/are/was/were)
    if root_token.lemma_ == "be":
        # "It is raining" -> extract "raining" as the predicate
        attr = [token for token in root_token.children 
               if token.dep_ in ["acomp", "attr", "advmod"]]
        if attr:
            return normalize_text(attr[0].text)
        else:
            return normalize_text(root_token.text)
    else:
        # Regular verb: "The dog barks"
        subj = [token for token in root_token.children if token.dep_ == "nsubj"]
        if subj:
            return f"{normalize_text(subj[0].text)}_{root_token.lemma_}"
        else:
            return normalize_text(root_token.text)

def normalize_text(text: str) -> str:
    """Convert text to valid predicate identifier"""
    return text.lower().replace(" ", "_").replace("-", "_").replace(",", "").replace(".", "")

if __name__ == "__main__":
    test_cases = [
        "It is not raining",
        "The dog is not barking",
        "It is raining",
        "Not today"
    ]
    
    for text in test_cases:
        result = test_negation_parsing(text)
        print(f"Final result: {result}")
        print("\n" + "="*60 + "\n")
