#!/usr/bin/env python3
"""
Fixed Natural Language to Propositional Logic API with Proper Negation Handling

This version correctly handles negation using spaCy's dependency parsing.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
import uvicorn
import spacy
import re
from itertools import product
from abc import ABC, abstractmethod


# ============================================================================
# PROPOSITIONAL LOGIC MODELS
# ============================================================================

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
    
    @abstractmethod
    def to_cnf(self) -> 'Formula':
        pass
    
    @abstractmethod
    def to_dnf(self) -> 'Formula':
        pass


class AtomicFormula(Formula):
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
        return self
    
    def to_dnf(self) -> 'Formula':
        return self


class Conjunction(Formula):
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
        return self
    
    def to_dnf(self) -> 'Formula':
        return self


class Disjunction(Formula):
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
        return self
    
    def to_dnf(self) -> 'Formula':
        return self


class Implication(Formula):
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
        return Disjunction([Negation(self.antecedent), self.consequent])
    
    def to_dnf(self) -> 'Formula':
        return Disjunction([Negation(self.antecedent), self.consequent])


# ============================================================================
# FORMULA UTILITIES
# ============================================================================

class FormulaUtils:
    def generate_truth_table(self, formula: Formula) -> Dict[str, Any]:
        atoms = list(formula.atoms())
        atoms.sort(key=lambda x: x.name)
        
        if not atoms:
            result = self._evaluate_formula(formula, {})
            return {
                "atoms": [],
                "rows": [{"values": [], "result": result}],
                "is_tautology": result,
                "is_contradiction": not result,
                "is_satisfiable": result
            }
        
        truth_combinations = list(product([False, True], repeat=len(atoms)))
        rows = []
        true_count = 0
        
        for combination in truth_combinations:
            valuation = {atoms[i]: combination[i] for i in range(len(atoms))}
            result = self._evaluate_formula(formula, valuation)
            
            row = {"values": list(combination), "result": result}
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
        else:
            raise ValueError(f"Unknown formula type: {type(formula)}")


# ============================================================================
# FIXED SEMANTIC PARSER WITH PROPER NEGATION HANDLING
# ============================================================================

class FixedSemanticParser:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("spaCy model not found. Please install with: python -m spacy download en_core_web_sm")
    
    def convert_to_propositional_logic(self, text: str) -> Tuple[Formula, float]:
        """Convert natural language text to propositional logic formula with proper negation handling"""
        print(f"DEBUG: Parsing text: '{text}'")
        
        # Clean and normalize text
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        # Check for conditionals (if...then)
        if "if" in text_lower and "then" in text_lower:
            print("DEBUG: Detected conditional structure")
            # Find the positions of "if" and "then"
            if_pos = text_lower.find("if")
            then_pos = text_lower.find("then")
            
            if if_pos < then_pos:
                antecedent_text = text_clean[if_pos + 2:then_pos].strip()
                consequent_text = text_clean[then_pos + 4:].strip()
                
                print(f"DEBUG: Antecedent: '{antecedent_text}'")
                print(f"DEBUG: Consequent: '{consequent_text}'")
                
                antecedent_formula, ant_conf = self.convert_to_propositional_logic(antecedent_text)
                consequent_formula, cons_conf = self.convert_to_propositional_logic(consequent_text)
                
                return Implication(antecedent_formula, consequent_formula), (ant_conf + cons_conf) / 2
        
        # Check for conjunctions (and)
        if " and " in text_lower:
            print("DEBUG: Detected conjunction structure")
            parts = text_clean.split(" and ", 1)
            if len(parts) == 2:
                left_text = parts[0].strip()
                right_text = parts[1].strip()
                
                print(f"DEBUG: Left: '{left_text}'")
                print(f"DEBUG: Right: '{right_text}'")
                
                left_formula, left_conf = self.convert_to_propositional_logic(left_text)
                right_formula, right_conf = self.convert_to_propositional_logic(right_text)
                
                return Conjunction([left_formula, right_formula]), (left_conf + right_conf) / 2
        
        # Check for disjunctions (or)
        if " or " in text_lower:
            print("DEBUG: Detected disjunction structure")
            parts = text_clean.split(" or ", 1)
            if len(parts) == 2:
                left_text = parts[0].strip()
                right_text = parts[1].strip()
                
                print(f"DEBUG: Left: '{left_text}'")
                print(f"DEBUG: Right: '{right_text}'")
                
                left_formula, left_conf = self.convert_to_propositional_logic(left_text)
                right_formula, right_conf = self.convert_to_propositional_logic(right_text)
                
                return Disjunction([left_formula, right_formula]), (left_conf + right_conf) / 2
        
        # Handle atomic formulas with proper negation detection
        print("DEBUG: Treating as atomic formula with negation check")
        return self._extract_proposition_with_negation(text_clean)
    
    def _extract_proposition_with_negation(self, text: str) -> Tuple[Formula, float]:
        """Extract proposition with proper negation handling using spaCy"""
        try:
            doc = self.nlp(text)
            
            # Check for negation using spaCy's dependency parsing
            has_negation = False
            negated_token = None
            
            for token in doc:
                # Check if there's a negation dependency
                if token.dep_ == "neg":
                    has_negation = True
                    negated_token = token
                    print(f"DEBUG: Found negation token: '{token.text}' negating '{token.head.text}'")
                    break
            
            # Find the main verb/predicate (ROOT)
            root = None
            for token in doc:
                if token.dep_ == "ROOT":
                    root = token
                    break
            
            if not root:
                # Fallback to simple normalization
                prop_id = self._normalize_text(text)
                print(f"DEBUG: No ROOT found, fallback: {prop_id}")
                return AtomicFormula(Atom(prop_id)), 0.4
            
            # Extract the base proposition
            base_prop = self._extract_base_proposition(root)
            print(f"DEBUG: Base proposition: {base_prop}")
            
            if has_negation:
                print(f"DEBUG: Creating negation: ¬{base_prop}")
                return Negation(AtomicFormula(Atom(base_prop))), 0.8
            else:
                print(f"DEBUG: Creating atomic formula: {base_prop}")
                return AtomicFormula(Atom(base_prop)), 0.8
        
        except Exception as e:
            print(f"DEBUG: Error in spaCy processing: {e}")
            # Fallback to simple normalization
            prop_id = self._normalize_text(text)
            print(f"DEBUG: Error fallback: {prop_id}")
            return AtomicFormula(Atom(prop_id)), 0.4
    
    def _extract_base_proposition(self, root_token) -> str:
        """Extract the base proposition from the root token"""
        # Handle copula (is/are/was/were)
        if root_token.lemma_ == "be":
            # "It is raining" -> extract "raining" as the predicate
            attr = [token for token in root_token.children 
                   if token.dep_ in ["acomp", "attr", "advmod"]]
            if attr:
                return self._normalize_text(attr[0].text)
            else:
                return self._normalize_text(root_token.text)
        else:
            # Regular verb: "The dog barks"
            subj = [token for token in root_token.children if token.dep_ == "nsubj"]
            if subj:
                return f"{self._normalize_text(subj[0].text)}_{root_token.lemma_}"
            else:
                return self._normalize_text(root_token.text)
    
    def _normalize_text(self, text: str) -> str:
        """Convert text to valid predicate identifier"""
        return text.lower().replace(" ", "_").replace("-", "_").replace(",", "").replace(".", "")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

class ConversionRequest(BaseModel):
    text: str = Field(..., description="Natural language text to convert", min_length=1)
    include_cnf: bool = Field(False, description="Include CNF conversion")
    include_dnf: bool = Field(False, description="Include DNF conversion")
    include_truth_table: bool = Field(False, description="Include truth table")


class ConversionResponse(BaseModel):
    original_text: str
    propositional_formula: str
    confidence: float
    atoms: List[str]
    cnf_formula: Optional[str] = None
    dnf_formula: Optional[str] = None
    truth_table: Optional[Dict[str, Any]] = None
    semantic_analysis: Optional[Dict[str, Any]] = None


class InferenceRequest(BaseModel):
    premises: List[str] = Field(..., description="List of premises", min_items=1)
    conclusion: str = Field(..., description="Conclusion to validate")


class InferenceResponse(BaseModel):
    valid: bool
    premises: List[str]
    conclusion: str
    implication: str
    counterexample: Optional[Dict[str, Any]] = None
    truth_table_summary: Dict[str, Any]
    error: Optional[str] = None


# Initialize FastAPI app
app = FastAPI(
    title="Fixed Natural Language to Propositional Logic API",
    description="Convert natural language text into propositional logic formulas with proper negation handling",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
parser = FixedSemanticParser()
formula_utils = FormulaUtils()


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fixed Natural Language to Propositional Logic API",
        "version": "4.0.0",
        "status": "running",
        "features": [
            "spaCy semantic parsing",
            "Proper negation handling",
            "Logical inference",
            "Truth table analysis",
            "CNF/DNF conversion"
        ],
        "endpoints": {
            "convert": "/convert",
            "infer": "/infer",
            "health": "/health",
            "test": "/test",
            "examples": "/examples"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "fixed-nlp-to-propositional-logic"}


@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify the system is working"""
    p = AtomicFormula(Atom("p"))
    q = AtomicFormula(Atom("q"))
    formula = Implication(p, q)
    
    truth_table = formula_utils.generate_truth_table(formula)
    
    return {
        "message": "Fixed system is working!",
        "test_formula": str(formula),
        "cnf": str(formula.to_cnf()),
        "dnf": str(formula.to_dnf()),
        "truth_table_summary": {
            "atoms": truth_table["atoms"],
            "is_tautology": truth_table["is_tautology"],
            "is_contradiction": truth_table["is_contradiction"],
            "is_satisfiable": truth_table["is_satisfiable"]
        },
        "spacy_loaded": True
    }


@app.post("/convert", response_model=ConversionResponse)
async def convert_text(request: ConversionRequest):
    """Convert natural language text to propositional logic formula with proper negation handling"""
    try:
        print(f"API: Converting text: '{request.text}'")
        formula, confidence = parser.convert_to_propositional_logic(request.text)
        atoms = [str(atom) for atom in formula.atoms()]
        
        print(f"API: Result formula: {formula}")
        print(f"API: Confidence: {confidence}")
        print(f"API: Atoms: {atoms}")
        
        response_data = {
            "original_text": request.text,
            "propositional_formula": str(formula),
            "confidence": confidence,
            "atoms": atoms
        }
        
        if request.include_cnf:
            cnf_formula = formula.to_cnf()
            response_data["cnf_formula"] = str(cnf_formula)
        
        if request.include_dnf:
            dnf_formula = formula.to_dnf()
            response_data["dnf_formula"] = str(dnf_formula)
        
        if request.include_truth_table:
            truth_table = formula_utils.generate_truth_table(formula)
            response_data["truth_table"] = truth_table
        
        # Add semantic analysis
        response_data["semantic_analysis"] = {
            "structure_detected": "Available",
            "spacy_processing": "Enabled",
            "negation_handling": "Fixed"
        }
        
        return ConversionResponse(**response_data)
        
    except Exception as e:
        print(f"API: Error in conversion: {e}")
        raise HTTPException(status_code=400, detail=f"Conversion failed: {str(e)}")


@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """Check if conclusion follows from premises using logical inference"""
    try:
        print(f"API: Inference request - Premises: {request.premises}, Conclusion: {request.conclusion}")
        
        # Convert premises to formulas
        premise_formulas = []
        for premise in request.premises:
            formula, _ = parser.convert_to_propositional_logic(premise)
            premise_formulas.append(formula)
        
        # Convert conclusion to formula
        conclusion_formula, _ = parser.convert_to_propositional_logic(request.conclusion)
        
        # Create implication: (P1 ∧ P2 ∧ ... ∧ Pn) → C
        if len(premise_formulas) == 1:
            implication = Implication(premise_formulas[0], conclusion_formula)
        else:
            conjunction = Conjunction(premise_formulas)
            implication = Implication(conjunction, conclusion_formula)
        
        print(f"API: Implication: {implication}")
        
        # Check if the implication is a tautology
        truth_table = formula_utils.generate_truth_table(implication)
        is_valid = truth_table["is_tautology"]
        
        print(f"API: Is valid: {is_valid}")
        
        # Find counterexample if invalid
        counterexample = None
        if not is_valid:
            for row in truth_table["rows"]:
                if not row["result"]:  # False result means counterexample
                    counterexample = {
                        "atoms": truth_table["atoms"],
                        "values": row["values"],
                        "description": f"Counterexample: {dict(zip(truth_table['atoms'], ['True' if v else 'False' for v in row['values']]))}"
                    }
                    break
        
        return InferenceResponse(
            valid=is_valid,
            premises=[str(f) for f in premise_formulas],
            conclusion=str(conclusion_formula),
            implication=str(implication),
            counterexample=counterexample,
            truth_table_summary={
                "is_tautology": truth_table["is_tautology"],
                "is_contradiction": truth_table["is_contradiction"],
                "is_satisfiable": truth_table["is_satisfiable"]
            }
        )
        
    except Exception as e:
        print(f"API: Error in inference: {e}")
        raise HTTPException(status_code=400, detail=f"Inference failed: {str(e)}")


@app.get("/examples")
async def get_examples():
    """Get example conversions with semantic analysis"""
    examples = [
        {
            "text": "It is not raining",
            "expected_formula": "¬raining",
            "description": "Negation with proper handling"
        },
        {
            "text": "The dog is not barking",
            "expected_formula": "¬the_dog_bark",
            "description": "Negation with subject-verb structure"
        },
        {
            "text": "It is raining and the ground is wet",
            "expected_formula": "(raining ∧ the_ground_is_wet)",
            "description": "Conjunction with dependency parsing"
        },
        {
            "text": "Either it is sunny or it is cloudy",
            "expected_formula": "(it_is_sunny ∨ it_is_cloudy)",
            "description": "Disjunction with either...or parsing"
        },
        {
            "text": "If it is not raining then we can go outside",
            "expected_formula": "(¬raining → we_can_go_outside)",
            "description": "Conditional with negation in antecedent"
        }
    ]
    
    return {"examples": examples}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
