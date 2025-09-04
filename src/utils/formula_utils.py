"""
Utility functions for propositional logic formulas

This module provides utility functions for working with propositional logic
formulas, including truth table generation, validation, and analysis.
"""

import re
from typing import List, Dict, Any, Set, Tuple
from itertools import product

from ..models.propositional_logic import Formula, Atom, AtomicFormula, Negation, Conjunction, Disjunction, Implication, Biconditional


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
    
    def simplify_formula(self, formula: Formula) -> Formula:
        """Simplify a formula by applying logical equivalences"""
        if isinstance(formula, Conjunction):
            # Remove duplicates and simplify
            simplified_formulas = []
            for f in formula.formulas:
                simplified = self.simplify_formula(f)
                if simplified not in simplified_formulas:
                    simplified_formulas.append(simplified)
            
            if len(simplified_formulas) == 1:
                return simplified_formulas[0]
            elif len(simplified_formulas) == 0:
                return AtomicFormula(Atom("true"))
            else:
                return Conjunction(simplified_formulas)
        
        elif isinstance(formula, Disjunction):
            # Remove duplicates and simplify
            simplified_formulas = []
            for f in formula.formulas:
                simplified = self.simplify_formula(f)
                if simplified not in simplified_formulas:
                    simplified_formulas.append(simplified)
            
            if len(simplified_formulas) == 1:
                return simplified_formulas[0]
            elif len(simplified_formulas) == 0:
                return AtomicFormula(Atom("false"))
            else:
                return Disjunction(simplified_formulas)
        
        elif isinstance(formula, Negation):
            simplified_inner = self.simplify_formula(formula.formula)
            
            # Double negation elimination
            if isinstance(simplified_inner, Negation):
                return simplified_inner.formula
            
            return Negation(simplified_inner)
        
        elif isinstance(formula, Implication):
            simplified_antecedent = self.simplify_formula(formula.antecedent)
            simplified_consequent = self.simplify_formula(formula.consequent)
            return Implication(simplified_antecedent, simplified_consequent)
        
        elif isinstance(formula, Biconditional):
            simplified_left = self.simplify_formula(formula.left)
            simplified_right = self.simplify_formula(formula.right)
            return Biconditional(simplified_left, simplified_right)
        
        else:
            return formula
    
    def is_equivalent(self, formula1: Formula, formula2: Formula) -> bool:
        """Check if two formulas are logically equivalent"""
        # Get all atoms from both formulas
        all_atoms = formula1.atoms().union(formula2.atoms())
        atoms_list = list(all_atoms)
        
        if not atoms_list:
            # Both formulas are constants
            val1 = self._evaluate_formula(formula1, {})
            val2 = self._evaluate_formula(formula2, {})
            return val1 == val2
        
        # Check all possible valuations
        for combination in product([False, True], repeat=len(atoms_list)):
            valuation = {atoms_list[i]: combination[i] for i in range(len(atoms_list))}
            
            val1 = self._evaluate_formula(formula1, valuation)
            val2 = self._evaluate_formula(formula2, valuation)
            
            if val1 != val2:
                return False
        
        return True
    
    def find_contradictions(self, formulas: List[Formula]) -> List[Tuple[int, int]]:
        """Find pairs of contradictory formulas"""
        contradictions = []
        
        for i in range(len(formulas)):
            for j in range(i + 1, len(formulas)):
                # Check if one is the negation of the other
                if (isinstance(formulas[i], Negation) and formulas[i].formula == formulas[j]) or \
                   (isinstance(formulas[j], Negation) and formulas[j].formula == formulas[i]):
                    contradictions.append((i, j))
                
                # Check if they are logically contradictory
                elif not self._are_satisfiable_together([formulas[i], formulas[j]]):
                    contradictions.append((i, j))
        
        return contradictions
    
    def _are_satisfiable_together(self, formulas: List[Formula]) -> bool:
        """Check if a list of formulas can all be true simultaneously"""
        all_atoms = set()
        for formula in formulas:
            all_atoms.update(formula.atoms())
        
        atoms_list = list(all_atoms)
        
        if not atoms_list:
            # All formulas are constants
            return all(self._evaluate_formula(f, {}) for f in formulas)
        
        # Check all possible valuations
        for combination in product([False, True], repeat=len(atoms_list)):
            valuation = {atoms_list[i]: combination[i] for i in range(len(atoms_list))}
            
            if all(self._evaluate_formula(f, valuation) for f in formulas):
                return True
        
        return False
    
    def generate_derivation_steps(self, formula: Formula) -> List[str]:
        """Generate step-by-step derivation of a formula"""
        steps = []
        
        if isinstance(formula, Conjunction):
            steps.append(f"Conjunction of {len(formula.formulas)} formulas:")
            for i, f in enumerate(formula.formulas):
                steps.append(f"  {i+1}. {f}")
        
        elif isinstance(formula, Disjunction):
            steps.append(f"Disjunction of {len(formula.formulas)} formulas:")
            for i, f in enumerate(formula.formulas):
                steps.append(f"  {i+1}. {f}")
        
        elif isinstance(formula, Implication):
            steps.append("Implication:")
            steps.append(f"  Antecedent: {formula.antecedent}")
            steps.append(f"  Consequent: {formula.consequent}")
        
        elif isinstance(formula, Biconditional):
            steps.append("Biconditional:")
            steps.append(f"  Left: {formula.left}")
            steps.append(f"  Right: {formula.right}")
        
        elif isinstance(formula, Negation):
            steps.append("Negation:")
            steps.append(f"  Formula: {formula.formula}")
        
        else:
            steps.append(f"Atomic formula: {formula}")
        
        return steps
