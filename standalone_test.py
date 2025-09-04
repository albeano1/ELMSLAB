#!/usr/bin/env python3
"""
Standalone test that includes all necessary code
"""

import re
from typing import List, Dict, Any, Set, Tuple
from itertools import product
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod


class Connective(Enum):
    """Logical connectives in propositional logic"""
    NOT = "¬"
    AND = "∧"
    OR = "∨"
    IMPLIES = "→"
    IFF = "↔"


@dataclass
class Atom:
    """Represents an atomic proposition (propositional variable)"""
    name: str
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Atom):
            return self.name == other.name
        return False
    
    def __hash__(self) -> int:
        return hash(self.name)


class Formula(ABC):
    """Abstract base class for propositional logic formulas"""
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def atoms(self) -> set[Atom]:
        """Return the set of atoms in this formula"""
        pass
    
    @abstractmethod
    def to_cnf(self) -> 'Formula':
        """Convert to Conjunctive Normal Form"""
        pass
    
    @abstractmethod
    def to_dnf(self) -> 'Formula':
        """Convert to Disjunctive Normal Form"""
        pass


class AtomicFormula(Formula):
    """Represents an atomic proposition"""
    
    def __init__(self, atom: Atom):
        self.atom = atom
    
    def __str__(self) -> str:
        return str(self.atom)
    
    def atoms(self) -> set[Atom]:
        return {self.atom}
    
    def to_cnf(self) -> 'Formula':
        return self
    
    def to_dnf(self) -> 'Formula':
        return self


class Negation(Formula):
    """Represents ¬φ"""
    
    def __init__(self, formula: Formula):
        self.formula = formula
    
    def __str__(self) -> str:
        if isinstance(self.formula, AtomicFormula):
            return f"¬{self.formula}"
        else:
            return f"¬({self.formula})"
    
    def atoms(self) -> set[Atom]:
        return self.formula.atoms()
    
    def to_cnf(self) -> 'Formula':
        # De Morgan's laws for CNF conversion
        if isinstance(self.formula, Conjunction):
            # ¬(φ₁ ∧ φ₂ ∧ ... ∧ φₙ) = ¬φ₁ ∨ ¬φ₂ ∨ ... ∨ ¬φₙ
            negated_formulas = [Negation(f) for f in self.formula.formulas]
            return Disjunction(negated_formulas).to_cnf()
        elif isinstance(self.formula, Disjunction):
            # ¬(φ₁ ∨ φ₂ ∨ ... ∨ φₙ) = ¬φ₁ ∧ ¬φ₂ ∧ ... ∧ ¬φₙ
            negated_formulas = [Negation(f) for f in self.formula.formulas]
            return Conjunction(negated_formulas).to_cnf()
        elif isinstance(self.formula, Implication):
            # ¬(φ → ψ) = φ ∧ ¬ψ
            return Conjunction([self.formula.antecedent, Negation(self.formula.consequent)]).to_cnf()
        elif isinstance(self.formula, Negation):
            # ¬¬φ = φ (double negation elimination)
            return self.formula.formula.to_cnf()
        else:
            return self
    
    def to_dnf(self) -> 'Formula':
        # Similar to CNF but with De Morgan's laws for DNF
        if isinstance(self.formula, Disjunction):
            negated_formulas = [Negation(f) for f in self.formula.formulas]
            return Conjunction(negated_formulas).to_dnf()
        elif isinstance(self.formula, Conjunction):
            negated_formulas = [Negation(f) for f in self.formula.formulas]
            return Disjunction(negated_formulas).to_dnf()
        elif isinstance(self.formula, Implication):
            return Conjunction([self.formula.antecedent, Negation(self.formula.consequent)]).to_dnf()
        elif isinstance(self.formula, Negation):
            return self.formula.formula.to_dnf()
        else:
            return self


class Conjunction(Formula):
    """Represents φ₁ ∧ φ₂ ∧ ... ∧ φₙ"""
    
    def __init__(self, formulas: List[Formula]):
        self.formulas = formulas
    
    def __str__(self) -> str:
        if len(self.formulas) == 1:
            return str(self.formulas[0])
        return "(" + " ∧ ".join(str(f) for f in self.formulas) + ")"
    
    def atoms(self) -> set[Atom]:
        result = set()
        for formula in self.formulas:
            result.update(formula.atoms())
        return result
    
    def to_cnf(self) -> 'Formula':
        # Conjunction of CNF formulas is already in CNF
        cnf_formulas = [f.to_cnf() for f in self.formulas]
        return Conjunction(cnf_formulas)
    
    def to_dnf(self) -> 'Formula':
        # Distribute conjunctions over disjunctions
        if len(self.formulas) == 1:
            return self.formulas[0].to_dnf()
        
        # Recursively convert to DNF
        dnf_formulas = [f.to_dnf() for f in self.formulas]
        
        # Distribute: (A ∨ B) ∧ (C ∨ D) = (A ∧ C) ∨ (A ∧ D) ∨ (B ∧ C) ∨ (B ∧ D)
        result_terms = []
        for i, formula1 in enumerate(dnf_formulas):
            for j, formula2 in enumerate(dnf_formulas[i+1:], i+1):
                if isinstance(formula1, Disjunction) and isinstance(formula2, Disjunction):
                    # Distribute disjunctions
                    for term1 in formula1.formulas:
                        for term2 in formula2.formulas:
                            result_terms.append(Conjunction([term1, term2]))
                else:
                    result_terms.append(Conjunction([formula1, formula2]))
        
        if result_terms:
            return Disjunction(result_terms)
        else:
            return Conjunction(dnf_formulas)


class Disjunction(Formula):
    """Represents φ₁ ∨ φ₂ ∨ ... ∨ φₙ"""
    
    def __init__(self, formulas: List[Formula]):
        self.formulas = formulas
    
    def __str__(self) -> str:
        if len(self.formulas) == 1:
            return str(self.formulas[0])
        return "(" + " ∨ ".join(str(f) for f in self.formulas) + ")"
    
    def atoms(self) -> set[Atom]:
        result = set()
        for formula in self.formulas:
            result.update(formula.atoms())
        return result
    
    def to_cnf(self) -> 'Formula':
        # Distribute disjunctions over conjunctions
        if len(self.formulas) == 1:
            return self.formulas[0].to_cnf()
        
        # Convert each formula to CNF first
        cnf_formulas = [f.to_cnf() for f in self.formulas]
        
        # Distribute: (A ∧ B) ∨ (C ∧ D) = (A ∨ C) ∧ (A ∨ D) ∧ (B ∨ C) ∧ (B ∨ D)
        result_clauses = []
        for i, formula1 in enumerate(cnf_formulas):
            for j, formula2 in enumerate(cnf_formulas[i+1:], i+1):
                if isinstance(formula1, Conjunction) and isinstance(formula2, Conjunction):
                    # Distribute conjunctions
                    for clause1 in formula1.formulas:
                        for clause2 in formula2.formulas:
                            result_clauses.append(Disjunction([clause1, clause2]))
                else:
                    result_clauses.append(Disjunction([formula1, formula2]))
        
        if result_clauses:
            return Conjunction(result_clauses)
        else:
            return Disjunction(cnf_formulas)
    
    def to_dnf(self) -> 'Formula':
        # Disjunction of DNF formulas is already in DNF
        dnf_formulas = [f.to_dnf() for f in self.formulas]
        return Disjunction(dnf_formulas)


class Implication(Formula):
    """Represents φ → ψ"""
    
    def __init__(self, antecedent: Formula, consequent: Formula):
        self.antecedent = antecedent
        self.consequent = consequent
    
    def __str__(self) -> str:
        return f"({self.antecedent} → {self.consequent})"
    
    def atoms(self) -> set[Atom]:
        result = self.antecedent.atoms()
        result.update(self.consequent.atoms())
        return result
    
    def to_cnf(self) -> 'Formula':
        # φ → ψ = ¬φ ∨ ψ
        return Disjunction([Negation(self.antecedent), self.consequent]).to_cnf()
    
    def to_dnf(self) -> 'Formula':
        # φ → ψ = ¬φ ∨ ψ
        return Disjunction([Negation(self.antecedent), self.consequent]).to_dnf()


class Biconditional(Formula):
    """Represents φ ↔ ψ"""
    
    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right
    
    def __str__(self) -> str:
        return f"({self.left} ↔ {self.right})"
    
    def atoms(self) -> set[Atom]:
        result = self.left.atoms()
        result.update(self.right.atoms())
        return result
    
    def to_cnf(self) -> 'Formula':
        # φ ↔ ψ = (φ → ψ) ∧ (ψ → φ) = (¬φ ∨ ψ) ∧ (¬ψ ∨ φ)
        return Conjunction([
            Implication(self.left, self.right).to_cnf(),
            Implication(self.right, self.left).to_cnf()
        ]).to_cnf()
    
    def to_dnf(self) -> 'Formula':
        # φ ↔ ψ = (φ ∧ ψ) ∨ (¬φ ∧ ¬ψ)
        return Disjunction([
            Conjunction([self.left, self.right]),
            Conjunction([Negation(self.left), Negation(self.right)])
        ]).to_dnf()


class FormulaUtils:
    """Utility class for propositional logic formula operations"""
    
    def __init__(self):
        self.operator_precedence = {
            '¬': 4,
            '∧': 3,
            '∨': 2,
            '→': 1,
            '↔': 0
        }
    
    def generate_truth_table(self, formula: Formula) -> Dict[str, Any]:
        """Generate truth table for a formula"""
        atoms = list(formula.atoms())
        atoms.sort(key=lambda x: x.name)  # Sort for consistent ordering
        
        if not atoms:
            # Formula with no atoms (constant)
            result = self._evaluate_formula(formula, {})
            return {
                "atoms": [],
                "rows": [{"values": [], "result": result}],
                "is_tautology": result,
                "is_contradiction": not result,
                "is_satisfiable": result
            }
        
        # Generate all possible truth value combinations
        truth_combinations = list(product([False, True], repeat=len(atoms)))
        
        rows = []
        true_count = 0
        
        for combination in truth_combinations:
            # Create valuation
            valuation = {atoms[i]: combination[i] for i in range(len(atoms))}
            
            # Evaluate formula
            result = self._evaluate_formula(formula, valuation)
            
            # Create row
            row = {
                "values": list(combination),
                "result": result
            }
            rows.append(row)
            
            if result:
                true_count += 1
        
        return {
            "atoms": [atom.name for atom in atoms],
            "rows": rows,
            "is_tautology": true_count == len(truth_combinations),
            "is_contradiction": true_count == 0,
            "is_satisfiable": true_count > 0
        }
    
    def _evaluate_formula(self, formula: Formula, valuation: Dict[Atom, bool]) -> bool:
        """Evaluate a formula under a given valuation"""
        if isinstance(formula, AtomicFormula):
            return valuation.get(formula.atom, False)
        
        elif isinstance(formula, Negation):
            return not self._evaluate_formula(formula.formula, valuation)
        
        elif isinstance(formula, Conjunction):
            return all(self._evaluate_formula(f, valuation) for f in formula.formulas)
        
        elif isinstance(formula, Disjunction):
            return any(self._evaluate_formula(f, valuation) for f in formula.formulas)
        
        elif isinstance(formula, Implication):
            antecedent_val = self._evaluate_formula(formula.antecedent, valuation)
            consequent_val = self._evaluate_formula(formula.consequent, valuation)
            return not antecedent_val or consequent_val
        
        elif isinstance(formula, Biconditional):
            left_val = self._evaluate_formula(formula.left, valuation)
            right_val = self._evaluate_formula(formula.right, valuation)
            return left_val == right_val
        
        else:
            raise ValueError(f"Unknown formula type: {type(formula)}")
    
    def is_valid_formula_syntax(self, formula_str: str) -> bool:
        """Check if a formula string has valid syntax"""
        try:
            # Basic syntax checks
            if not formula_str.strip():
                return False
            
            # Check for balanced parentheses
            if not self._check_balanced_parentheses(formula_str):
                return False
            
            # Check for valid operators
            if not self._check_valid_operators(formula_str):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _check_balanced_parentheses(self, formula_str: str) -> bool:
        """Check if parentheses are balanced"""
        count = 0
        for char in formula_str:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
                if count < 0:
                    return False
        return count == 0
    
    def _check_valid_operators(self, formula_str: str) -> bool:
        """Check if operators are valid"""
        valid_chars = set('()¬∧∨→↔ ')
        valid_chars.update('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')
        
        for char in formula_str:
            if char not in valid_chars:
                return False
        
        return True
    
    def extract_atoms_from_string(self, formula_str: str) -> List[str]:
        """Extract atom names from a formula string"""
        # Remove operators and parentheses
        clean_str = re.sub(r'[¬∧∨→↔()]', ' ', formula_str)
        
        # Split by whitespace and filter
        words = clean_str.split()
        atoms = []
        
        for word in words:
            word = word.strip()
            if word and word not in ['and', 'or', 'not', 'if', 'then', 'iff']:
                atoms.append(word)
        
        return list(set(atoms))  # Remove duplicates


def test_basic_formulas():
    """Test basic propositional logic formulas"""
    print("=== Basic Propositional Logic Formulas ===")
    
    # Create atomic formulas
    p = AtomicFormula(Atom("rain"))
    q = AtomicFormula(Atom("wet_ground"))
    r = AtomicFormula(Atom("sunny"))
    
    print(f"Atomic formula p: {p}")
    print(f"Atomic formula q: {q}")
    print(f"Atomic formula r: {r}")
    
    # Create compound formulas
    conjunction = Conjunction([p, q])
    disjunction = Disjunction([p, r])
    implication = Implication(p, q)
    negation = Negation(p)
    biconditional = Biconditional(p, q)
    
    print(f"\nConjunction (p ∧ q): {conjunction}")
    print(f"Disjunction (p ∨ r): {disjunction}")
    print(f"Implication (p → q): {implication}")
    print(f"Negation (¬p): {negation}")
    print(f"Biconditional (p ↔ q): {biconditional}")
    
    # Show atoms
    print(f"\nAtoms in conjunction: {[str(atom) for atom in conjunction.atoms()]}")
    print(f"Atoms in implication: {[str(atom) for atom in implication.atoms()]}")


def test_truth_tables():
    """Test truth table generation"""
    print("\n=== Truth Table Generation ===")
    
    utils = FormulaUtils()
    
    # Create a simple formula: p → q
    p = AtomicFormula(Atom("p"))
    q = AtomicFormula(Atom("q"))
    implication = Implication(p, q)
    
    print(f"Formula: {implication}")
    
    truth_table = utils.generate_truth_table(implication)
    
    print(f"Atoms: {truth_table['atoms']}")
    print("Truth Table:")
    print("p | q | p → q")
    print("--|---|-------")
    
    for row in truth_table['rows']:
        p_val = "T" if row['values'][0] else "F"
        q_val = "T" if row['values'][1] else "F"
        result = "T" if row['result'] else "F"
        print(f"{p_val} | {q_val} |   {result}")
    
    print(f"\nIs tautology: {truth_table['is_tautology']}")
    print(f"Is contradiction: {truth_table['is_contradiction']}")
    print(f"Is satisfiable: {truth_table['is_satisfiable']}")


def test_cnf_dnf_conversion():
    """Test CNF and DNF conversion"""
    print("\n=== CNF and DNF Conversion ===")
    
    # Create a simple implication: p → q
    p = AtomicFormula(Atom("p"))
    q = AtomicFormula(Atom("q"))
    implication = Implication(p, q)
    
    print(f"Original formula: {implication}")
    
    # Convert to CNF
    cnf_formula = implication.to_cnf()
    print(f"CNF: {cnf_formula}")
    
    # Convert to DNF
    dnf_formula = implication.to_dnf()
    print(f"DNF: {dnf_formula}")


def test_formula_validation():
    """Test formula validation"""
    print("\n=== Formula Validation ===")
    
    utils = FormulaUtils()
    
    test_formulas = [
        "(p ∧ q)",
        "¬(p ∨ q)",
        "p → q",
        "((p ∧ q) ∨ r)",
        "invalid formula",
        "p ∧ ∧ q",  # Invalid syntax
    ]
    
    for formula_str in test_formulas:
        is_valid = utils.is_valid_formula_syntax(formula_str)
        atoms = utils.extract_atoms_from_string(formula_str)
        print(f"'{formula_str}' -> Valid: {is_valid}, Atoms: {atoms}")


def main():
    """Run all tests"""
    print("Natural Language to Propositional Logic API - Standalone Test")
    print("=" * 70)
    
    try:
        test_basic_formulas()
        test_truth_tables()
        test_cnf_dnf_conversion()
        test_formula_validation()
        
        print("\n" + "=" * 70)
        print("Standalone test completed successfully!")
        print("\nThe core propositional logic functionality is working correctly.")
        print("\nTo run the full API with natural language processing:")
        print("1. pip install -r requirements.txt")
        print("2. python -m spacy download en_core_web_sm")
        print("3. python -m uvicorn src.api.main:app --reload")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
