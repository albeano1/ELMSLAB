"""
Temporal Logic data structures and operations

This module provides support for temporal logic, including past, future, and next operators
for reasoning about time-dependent statements.
"""

from typing import Union, List, Optional, Dict, Any, Set
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .propositional_logic import Formula, Atom, AtomicFormula, Negation, Conjunction, Disjunction, Implication, Biconditional

class TemporalOperator(Enum):
    """Temporal operators in temporal logic."""
    PAST = "Past"
    FUTURE = "Future" 
    NEXT = "Next"
    ALWAYS = "Always"
    EVENTUALLY = "Eventually"

@dataclass(frozen=True)
class TemporalFormula(Formula):
    """Abstract base class for temporal logic formulas."""
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def atoms(self) -> Set[Atom]:
        """Returns the set of atoms in the formula."""
        pass

@dataclass(frozen=True)
class PastFormula(TemporalFormula):
    """Represents Past(φ) - φ was true in the past."""
    formula: Formula
    
    def __str__(self) -> str:
        if isinstance(self.formula, AtomicFormula):
            return f"Past({self.formula})"
        return f"Past({self.formula})"
    
    def atoms(self) -> Set[Atom]:
        return self.formula.atoms()
    
    def to_cnf(self) -> 'PastFormula':
        """Convert to CNF (simplified for temporal logic)."""
        return PastFormula(self.formula.to_cnf())
    
    def to_dnf(self) -> 'PastFormula':
        """Convert to DNF (simplified for temporal logic)."""
        return PastFormula(self.formula.to_dnf())

@dataclass(frozen=True)
class FutureFormula(TemporalFormula):
    """Represents Future(φ) - φ will be true in the future."""
    formula: Formula
    
    def __str__(self) -> str:
        if isinstance(self.formula, AtomicFormula):
            return f"Future({self.formula})"
        return f"Future({self.formula})"
    
    def atoms(self) -> Set[Atom]:
        return self.formula.atoms()
    
    def to_cnf(self) -> 'FutureFormula':
        """Convert to CNF (simplified for temporal logic)."""
        return FutureFormula(self.formula.to_cnf())
    
    def to_dnf(self) -> 'FutureFormula':
        """Convert to DNF (simplified for temporal logic)."""
        return FutureFormula(self.formula.to_dnf())

@dataclass(frozen=True)
class NextFormula(TemporalFormula):
    """Represents Next(φ) - φ will be true in the next time step."""
    formula: Formula
    
    def __str__(self) -> str:
        if isinstance(self.formula, AtomicFormula):
            return f"Next({self.formula})"
        return f"Next({self.formula})"
    
    def atoms(self) -> Set[Atom]:
        return self.formula.atoms()
    
    def to_cnf(self) -> 'NextFormula':
        """Convert to CNF (simplified for temporal logic)."""
        return NextFormula(self.formula.to_cnf())
    
    def to_dnf(self) -> 'NextFormula':
        """Convert to DNF (simplified for temporal logic)."""
        return NextFormula(self.formula.to_dnf())

@dataclass(frozen=True)
class AlwaysFormula(TemporalFormula):
    """Represents Always(φ) - φ is always true."""
    formula: Formula
    
    def __str__(self) -> str:
        if isinstance(self.formula, AtomicFormula):
            return f"Always({self.formula})"
        return f"Always({self.formula})"
    
    def atoms(self) -> Set[Atom]:
        return self.formula.atoms()
    
    def to_cnf(self) -> 'AlwaysFormula':
        """Convert to CNF (simplified for temporal logic)."""
        return AlwaysFormula(self.formula.to_cnf())
    
    def to_dnf(self) -> 'AlwaysFormula':
        """Convert to DNF (simplified for temporal logic)."""
        return AlwaysFormula(self.formula.to_dnf())

@dataclass(frozen=True)
class EventuallyFormula(TemporalFormula):
    """Represents Eventually(φ) - φ will eventually be true."""
    formula: Formula
    
    def __str__(self) -> str:
        if isinstance(self.formula, AtomicFormula):
            return f"Eventually({self.formula})"
        return f"Eventually({self.formula})"
    
    def atoms(self) -> Set[Atom]:
        return self.formula.atoms()
    
    def to_cnf(self) -> 'EventuallyFormula':
        """Convert to CNF (simplified for temporal logic)."""
        return EventuallyFormula(self.formula.to_cnf())
    
    def to_dnf(self) -> 'EventuallyFormula':
        """Convert to DNF (simplified for temporal logic)."""
        return EventuallyFormula(self.formula.to_dnf())

# Convenience functions for creating temporal formulas
def past(formula: Formula) -> PastFormula:
    """Create a Past formula."""
    return PastFormula(formula)

def future(formula: Formula) -> FutureFormula:
    """Create a Future formula."""
    return FutureFormula(formula)

def next_formula(formula: Formula) -> NextFormula:
    """Create a Next formula."""
    return NextFormula(formula)

def always(formula: Formula) -> AlwaysFormula:
    """Create an Always formula."""
    return AlwaysFormula(formula)

def eventually(formula: Formula) -> EventuallyFormula:
    """Create an Eventually formula."""
    return EventuallyFormula(formula)
