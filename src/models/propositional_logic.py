"""
Propositional Logic Models and Data Structures

This module defines the core data structures for representing propositional logic
formulas, including atoms, connectives, and well-formed formulas (WFFs).
"""

from typing import Union, List, Optional, Dict, Any
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


# Convenience functions for creating formulas
def atom(name: str) -> AtomicFormula:
    """Create an atomic formula"""
    return AtomicFormula(Atom(name))


def neg(formula: Formula) -> Negation:
    """Create a negation"""
    return Negation(formula)


def conj(formulas: List[Formula]) -> Conjunction:
    """Create a conjunction"""
    return Conjunction(formulas)


def disj(formulas: List[Formula]) -> Disjunction:
    """Create a disjunction"""
    return Disjunction(formulas)


def impl(antecedent: Formula, consequent: Formula) -> Implication:
    """Create an implication"""
    return Implication(antecedent, consequent)


def iff(left: Formula, right: Formula) -> Biconditional:
    """Create a biconditional"""
    return Biconditional(left, right)
