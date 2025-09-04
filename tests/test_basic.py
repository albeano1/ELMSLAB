"""
Basic tests for the Natural Language to Propositional Logic API
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.propositional_logic import Atom, AtomicFormula, Negation, Conjunction, Disjunction, Implication
from core.nlp_parser import PropositionalLogicConverter
from utils.formula_utils import FormulaUtils


def test_atomic_formula():
    """Test atomic formula creation"""
    atom = Atom("rain")
    formula = AtomicFormula(atom)
    assert str(formula) == "rain"
    assert formula.atoms() == {atom}


def test_conjunction():
    """Test conjunction creation"""
    p = AtomicFormula(Atom("p"))
    q = AtomicFormula(Atom("q"))
    conj = Conjunction([p, q])
    assert str(conj) == "(p ∧ q)"
    assert len(conj.atoms()) == 2


def test_disjunction():
    """Test disjunction creation"""
    p = AtomicFormula(Atom("p"))
    q = AtomicFormula(Atom("q"))
    disj = Disjunction([p, q])
    assert str(disj) == "(p ∨ q)"
    assert len(disj.atoms()) == 2


def test_implication():
    """Test implication creation"""
    p = AtomicFormula(Atom("p"))
    q = AtomicFormula(Atom("q"))
    impl = Implication(p, q)
    assert str(impl) == "(p → q)"
    assert len(impl.atoms()) == 2


def test_negation():
    """Test negation creation"""
    p = AtomicFormula(Atom("p"))
    neg = Negation(p)
    assert str(neg) == "¬p"
    assert len(neg.atoms()) == 1


def test_nlp_converter():
    """Test natural language to propositional logic conversion"""
    converter = PropositionalLogicConverter()
    
    # Test simple conjunction
    formula, confidence = converter.convert_to_propositional_logic("It is raining and the ground is wet")
    assert confidence > 0
    assert formula is not None


def test_formula_utils():
    """Test formula utility functions"""
    utils = FormulaUtils()
    
    # Test truth table generation
    p = AtomicFormula(Atom("p"))
    truth_table = utils.generate_truth_table(p)
    assert "atoms" in truth_table
    assert "rows" in truth_table
    assert len(truth_table["rows"]) == 2  # 2^1 = 2 rows for one atom


def test_cnf_conversion():
    """Test CNF conversion"""
    p = AtomicFormula(Atom("p"))
    q = AtomicFormula(Atom("q"))
    impl = Implication(p, q)
    cnf = impl.to_cnf()
    
    # Implication should convert to disjunction in CNF
    assert isinstance(cnf, Disjunction)


def test_dnf_conversion():
    """Test DNF conversion"""
    p = AtomicFormula(Atom("p"))
    q = AtomicFormula(Atom("q"))
    impl = Implication(p, q)
    dnf = impl.to_dnf()
    
    # Implication should convert to disjunction in DNF
    assert isinstance(dnf, Disjunction)


if __name__ == "__main__":
    pytest.main([__file__])
