"""
First-Order Logic Models and Data Structures

This module extends propositional logic to support first-order logic with:
- Quantifiers (∀, ∃)
- Predicates (P(x), Q(x,y))
- Individual constants (a, b, c)
- Variables (x, y, z)
- Functions (f(x), g(x,y))
"""

from typing import Union, List, Optional, Dict, Any, Set
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from .propositional_logic import Formula, Atom, AtomicFormula, Negation, Conjunction, Disjunction, Implication, Biconditional


class Quantifier(Enum):
    """Quantifiers in first-order logic"""
    FORALL = "∀"  # Universal quantifier
    EXISTS = "∃"  # Existential quantifier


@dataclass
class Variable:
    """Represents a variable in first-order logic (x, y, z, etc.)"""
    name: str
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Variable):
            return self.name == other.name
        return False
    
    def __hash__(self) -> int:
        return hash(self.name)


@dataclass
class Constant:
    """Represents an individual constant (a, b, c, Socrates, etc.)"""
    name: str
    
    def __str__(self) -> str:
        return self.name
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Constant):
            return self.name == other.name
        return False
    
    def __hash__(self) -> int:
        return hash(self.name)


class Term(ABC):
    """Abstract base class for terms in first-order logic"""
    
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def variables(self) -> Set[Variable]:
        """Return the set of variables in this term"""
        pass
    
    @abstractmethod
    def constants(self) -> Set[Constant]:
        """Return the set of constants in this term"""
        pass


class VariableTerm(Term):
    """Represents a variable as a term"""
    
    def __init__(self, variable: Variable):
        self.variable = variable
    
    def __str__(self) -> str:
        return str(self.variable)
    
    def variables(self) -> Set[Variable]:
        return {self.variable}
    
    def constants(self) -> Set[Constant]:
        return set()


class ConstantTerm(Term):
    """Represents a constant as a term"""
    
    def __init__(self, constant: Constant):
        self.constant = constant
    
    def __str__(self) -> str:
        return str(self.constant)
    
    def variables(self) -> Set[Variable]:
        return set()
    
    def constants(self) -> Set[Constant]:
        return {self.constant}


class FunctionTerm(Term):
    """Represents a function application f(t1, t2, ..., tn)"""
    
    def __init__(self, function_name: str, arguments: List[Term]):
        self.function_name = function_name
        self.arguments = arguments
    
    def __str__(self) -> str:
        if not self.arguments:
            return self.function_name
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.function_name}({args_str})"
    
    def variables(self) -> Set[Variable]:
        result = set()
        for arg in self.arguments:
            result.update(arg.variables())
        return result
    
    def constants(self) -> Set[Constant]:
        result = set()
        for arg in self.arguments:
            result.update(arg.constants())
        return result


class Predicate:
    """Represents a predicate P(t1, t2, ..., tn)"""
    
    def __init__(self, predicate_name: str, arguments: List[Term]):
        self.predicate_name = predicate_name
        self.arguments = arguments
    
    def __str__(self) -> str:
        if not self.arguments:
            return self.predicate_name
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.predicate_name}({args_str})"
    
    def variables(self) -> Set[Variable]:
        result = set()
        for arg in self.arguments:
            result.update(arg.variables())
        return result
    
    def constants(self) -> Set[Constant]:
        result = set()
        for arg in self.arguments:
            result.update(arg.constants())
        return result


class FirstOrderFormula(Formula):
    """Abstract base class for first-order logic formulas"""
    
    @abstractmethod
    def variables(self) -> Set[Variable]:
        """Return the set of variables in this formula"""
        pass
    
    @abstractmethod
    def constants(self) -> Set[Constant]:
        """Return the set of constants in this formula"""
        pass
    
    @abstractmethod
    def predicates(self) -> Set[Predicate]:
        """Return the set of predicates in this formula"""
        pass


class PredicateFormula(FirstOrderFormula):
    """Represents a predicate P(t1, t2, ..., tn)"""
    
    def __init__(self, predicate: Predicate):
        self.predicate = predicate
    
    def __str__(self) -> str:
        return str(self.predicate)
    
    def atoms(self) -> set[Atom]:
        # For compatibility with propositional logic
        return {Atom(str(self.predicate))}
    
    def variables(self) -> Set[Variable]:
        return self.predicate.variables()
    
    def constants(self) -> Set[Constant]:
        return self.predicate.constants()
    
    def predicates(self) -> Set[Predicate]:
        return {self.predicate}
    
    def to_cnf(self) -> 'FirstOrderFormula':
        return self
    
    def to_dnf(self) -> 'FirstOrderFormula':
        return self


class QuantifiedFormula(FirstOrderFormula):
    """Represents ∀x φ or ∃x φ"""
    
    def __init__(self, quantifier: Quantifier, variable: Variable, formula: FirstOrderFormula):
        self.quantifier = quantifier
        self.variable = variable
        self.formula = formula
    
    def __str__(self) -> str:
        return f"{self.quantifier.value}{self.variable}({self.formula})"
    
    def atoms(self) -> set[Atom]:
        # For compatibility with propositional logic
        return self.formula.atoms()
    
    def variables(self) -> Set[Variable]:
        result = self.formula.variables()
        result.add(self.variable)  # The quantified variable
        return result
    
    def constants(self) -> Set[Constant]:
        return self.formula.constants()
    
    def predicates(self) -> Set[Predicate]:
        return self.formula.predicates()
    
    def to_cnf(self) -> 'FirstOrderFormula':
        # Quantifiers distribute over CNF
        if isinstance(self.formula, FirstOrderConjunction):
            cnf_formulas = [f.to_cnf() for f in self.formula.formulas]
            return FirstOrderConjunction([
                QuantifiedFormula(self.quantifier, self.variable, f) for f in cnf_formulas
            ])
        else:
            return QuantifiedFormula(self.quantifier, self.variable, self.formula.to_cnf())
    
    def to_dnf(self) -> 'FirstOrderFormula':
        # Quantifiers distribute over DNF
        if isinstance(self.formula, FirstOrderDisjunction):
            dnf_formulas = [f.to_dnf() for f in self.formula.formulas]
            return FirstOrderDisjunction([
                QuantifiedFormula(self.quantifier, self.variable, f) for f in dnf_formulas
            ])
        else:
            return QuantifiedFormula(self.quantifier, self.variable, self.formula.to_dnf())


class FirstOrderNegation(FirstOrderFormula):
    """Represents ¬φ in first-order logic"""
    
    def __init__(self, formula: FirstOrderFormula):
        self.formula = formula
    
    def __str__(self) -> str:
        if isinstance(self.formula, PredicateFormula):
            return f"¬{self.formula}"
        else:
            return f"¬({self.formula})"
    
    def atoms(self) -> set[Atom]:
        return self.formula.atoms()
    
    def variables(self) -> Set[Variable]:
        return self.formula.variables()
    
    def constants(self) -> Set[Constant]:
        return self.formula.constants()
    
    def predicates(self) -> Set[Predicate]:
        return self.formula.predicates()
    
    def to_cnf(self) -> 'FirstOrderFormula':
        # De Morgan's laws for first-order logic
        if isinstance(self.formula, FirstOrderConjunction):
            negated_formulas = [FirstOrderNegation(f) for f in self.formula.formulas]
            return FirstOrderDisjunction(negated_formulas).to_cnf()
        elif isinstance(self.formula, FirstOrderDisjunction):
            negated_formulas = [FirstOrderNegation(f) for f in self.formula.formulas]
            return FirstOrderConjunction(negated_formulas).to_cnf()
        elif isinstance(self.formula, FirstOrderImplication):
            # ¬(φ → ψ) = φ ∧ ¬ψ
            return FirstOrderConjunction([self.formula.antecedent, FirstOrderNegation(self.formula.consequent)]).to_cnf()
        elif isinstance(self.formula, QuantifiedFormula):
            # ¬∀x φ = ∃x ¬φ and ¬∃x φ = ∀x ¬φ
            new_quantifier = Quantifier.EXISTS if self.formula.quantifier == Quantifier.FORALL else Quantifier.FORALL
            return QuantifiedFormula(new_quantifier, self.formula.variable, FirstOrderNegation(self.formula.formula))
        elif isinstance(self.formula, FirstOrderNegation):
            # ¬¬φ = φ (double negation elimination)
            return self.formula.formula.to_cnf()
        else:
            return self
    
    def to_dnf(self) -> 'FirstOrderFormula':
        # Similar to CNF but with De Morgan's laws for DNF
        if isinstance(self.formula, FirstOrderDisjunction):
            negated_formulas = [FirstOrderNegation(f) for f in self.formula.formulas]
            return FirstOrderConjunction(negated_formulas).to_dnf()
        elif isinstance(self.formula, FirstOrderConjunction):
            negated_formulas = [FirstOrderNegation(f) for f in self.formula.formulas]
            return FirstOrderDisjunction(negated_formulas).to_dnf()
        elif isinstance(self.formula, FirstOrderImplication):
            return FirstOrderConjunction([self.formula.antecedent, FirstOrderNegation(self.formula.consequent)]).to_dnf()
        elif isinstance(self.formula, QuantifiedFormula):
            new_quantifier = Quantifier.EXISTS if self.formula.quantifier == Quantifier.FORALL else Quantifier.FORALL
            return QuantifiedFormula(new_quantifier, self.formula.variable, FirstOrderNegation(self.formula.formula))
        elif isinstance(self.formula, FirstOrderNegation):
            return self.formula.formula.to_dnf()
        else:
            return self


class FirstOrderConjunction(FirstOrderFormula):
    """Represents φ₁ ∧ φ₂ ∧ ... ∧ φₙ in first-order logic"""
    
    def __init__(self, formulas: List[FirstOrderFormula]):
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
    
    def variables(self) -> Set[Variable]:
        result = set()
        for formula in self.formulas:
            result.update(formula.variables())
        return result
    
    def constants(self) -> Set[Constant]:
        result = set()
        for formula in self.formulas:
            result.update(formula.constants())
        return result
    
    def predicates(self) -> Set[Predicate]:
        result = set()
        for formula in self.formulas:
            result.update(formula.predicates())
        return result
    
    def to_cnf(self) -> 'FirstOrderFormula':
        cnf_formulas = [f.to_cnf() for f in self.formulas]
        return FirstOrderConjunction(cnf_formulas)
    
    def to_dnf(self) -> 'FirstOrderFormula':
        # Distribute conjunctions over disjunctions
        if len(self.formulas) == 1:
            return self.formulas[0].to_dnf()
        
        dnf_formulas = [f.to_dnf() for f in self.formulas]
        
        # Distribute: (A ∨ B) ∧ (C ∨ D) = (A ∧ C) ∨ (A ∧ D) ∨ (B ∧ C) ∨ (B ∧ D)
        result_terms = []
        for i, formula1 in enumerate(dnf_formulas):
            for j, formula2 in enumerate(dnf_formulas[i+1:], i+1):
                if isinstance(formula1, FirstOrderDisjunction) and isinstance(formula2, FirstOrderDisjunction):
                    for term1 in formula1.formulas:
                        for term2 in formula2.formulas:
                            result_terms.append(FirstOrderConjunction([term1, term2]))
                else:
                    result_terms.append(FirstOrderConjunction([formula1, formula2]))
        
        if result_terms:
            return FirstOrderDisjunction(result_terms)
        else:
            return FirstOrderConjunction(dnf_formulas)


class FirstOrderDisjunction(FirstOrderFormula):
    """Represents φ₁ ∨ φ₂ ∨ ... ∨ φₙ in first-order logic"""
    
    def __init__(self, formulas: List[FirstOrderFormula]):
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
    
    def variables(self) -> Set[Variable]:
        result = set()
        for formula in self.formulas:
            result.update(formula.variables())
        return result
    
    def constants(self) -> Set[Constant]:
        result = set()
        for formula in self.formulas:
            result.update(formula.constants())
        return result
    
    def predicates(self) -> Set[Predicate]:
        result = set()
        for formula in self.formulas:
            result.update(formula.predicates())
        return result
    
    def to_cnf(self) -> 'FirstOrderFormula':
        # Distribute disjunctions over conjunctions
        if len(self.formulas) == 1:
            return self.formulas[0].to_cnf()
        
        cnf_formulas = [f.to_cnf() for f in self.formulas]
        
        # Distribute: (A ∧ B) ∨ (C ∧ D) = (A ∨ C) ∧ (A ∨ D) ∧ (B ∨ C) ∧ (B ∨ D)
        result_clauses = []
        for i, formula1 in enumerate(cnf_formulas):
            for j, formula2 in enumerate(cnf_formulas[i+1:], i+1):
                if isinstance(formula1, FirstOrderConjunction) and isinstance(formula2, FirstOrderConjunction):
                    for clause1 in formula1.formulas:
                        for clause2 in formula2.formulas:
                            result_clauses.append(FirstOrderDisjunction([clause1, clause2]))
                else:
                    result_clauses.append(FirstOrderDisjunction([formula1, formula2]))
        
        if result_clauses:
            return FirstOrderConjunction(result_clauses)
        else:
            return FirstOrderDisjunction(cnf_formulas)
    
    def to_dnf(self) -> 'FirstOrderFormula':
        dnf_formulas = [f.to_dnf() for f in self.formulas]
        return FirstOrderDisjunction(dnf_formulas)


class FirstOrderImplication(FirstOrderFormula):
    """Represents φ → ψ in first-order logic"""
    
    def __init__(self, antecedent: FirstOrderFormula, consequent: FirstOrderFormula):
        self.antecedent = antecedent
        self.consequent = consequent
    
    def __str__(self) -> str:
        return f"({self.antecedent} → {self.consequent})"
    
    def atoms(self) -> set[Atom]:
        result = self.antecedent.atoms()
        result.update(self.consequent.atoms())
        return result
    
    def variables(self) -> Set[Variable]:
        result = self.antecedent.variables()
        result.update(self.consequent.variables())
        return result
    
    def constants(self) -> Set[Constant]:
        result = self.antecedent.constants()
        result.update(self.consequent.constants())
        return result
    
    def predicates(self) -> Set[Predicate]:
        result = self.antecedent.predicates()
        result.update(self.consequent.predicates())
        return result
    
    def to_cnf(self) -> 'FirstOrderFormula':
        # φ → ψ = ¬φ ∨ ψ
        return FirstOrderDisjunction([FirstOrderNegation(self.antecedent), self.consequent]).to_cnf()
    
    def to_dnf(self) -> 'FirstOrderFormula':
        # φ → ψ = ¬φ ∨ ψ
        return FirstOrderDisjunction([FirstOrderNegation(self.antecedent), self.consequent]).to_dnf()


# Convenience functions for creating first-order formulas
def variable(name: str) -> Variable:
    """Create a variable"""
    return Variable(name)


def constant(name: str) -> Constant:
    """Create a constant"""
    return Constant(name)


def predicate(predicate_name: str, *args: Union[str, Term]) -> PredicateFormula:
    """Create a predicate formula"""
    terms = []
    for arg in args:
        if isinstance(arg, str):
            # Try to determine if it's a variable or constant
            if arg.islower() and len(arg) == 1:  # Single lowercase letter = variable
                terms.append(VariableTerm(Variable(arg)))
            else:  # Otherwise = constant
                terms.append(ConstantTerm(Constant(arg)))
        else:
            terms.append(arg)
    
    return PredicateFormula(Predicate(predicate_name, terms))


def forall(variable: Union[str, Variable], formula: FirstOrderFormula) -> QuantifiedFormula:
    """Create a universal quantifier"""
    if isinstance(variable, str):
        variable = Variable(variable)
    return QuantifiedFormula(Quantifier.FORALL, variable, formula)


def exists(variable: Union[str, Variable], formula: FirstOrderFormula) -> QuantifiedFormula:
    """Create an existential quantifier"""
    if isinstance(variable, str):
        variable = Variable(variable)
    return QuantifiedFormula(Quantifier.EXISTS, variable, formula)


def f_neg(formula: FirstOrderFormula) -> FirstOrderNegation:
    """Create a first-order negation"""
    return FirstOrderNegation(formula)


def f_conj(formulas: List[FirstOrderFormula]) -> FirstOrderConjunction:
    """Create a first-order conjunction"""
    return FirstOrderConjunction(formulas)


def f_disj(formulas: List[FirstOrderFormula]) -> FirstOrderDisjunction:
    """Create a first-order disjunction"""
    return FirstOrderDisjunction(formulas)


def f_impl(antecedent: FirstOrderFormula, consequent: FirstOrderFormula) -> FirstOrderImplication:
    """Create a first-order implication"""
    return FirstOrderImplication(antecedent, consequent)
