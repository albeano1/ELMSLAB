"""
Enhanced First-Order Logic API

This API extends the propositional logic system to support first-order logic
with quantifiers, predicates, and individual constants.
"""

import spacy
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union

from src.models.propositional_logic import Formula, Atom, AtomicFormula, Negation, Conjunction, Disjunction, Implication, Biconditional
from src.models.first_order_logic import (
    FirstOrderFormula, PredicateFormula, QuantifiedFormula, FirstOrderNegation,
    FirstOrderConjunction, FirstOrderDisjunction, FirstOrderImplication,
    Predicate, Variable, Constant, VariableTerm, ConstantTerm,
    Quantifier, forall, exists, predicate, f_neg, f_conj, f_disj, f_impl
)
from src.utils.formula_utils import FormulaUtils
from src.core.first_order_parser import FirstOrderLogicConverter

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

app = FastAPI(
    title="Enhanced First-Order Logic API",
    version="6.0.0",
    description="A functional live API that converts natural English into both propositional and first-order logic statements with advanced reasoning capabilities, built for ELMSLAB CPSC26-08."
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

formula_utils = FormulaUtils()
fol_converter = FirstOrderLogicConverter()

def simple_parse_formula(formula_str: str) -> Formula:
    """Simple formula parser for propositional logic"""
    formula_str = formula_str.strip()
    
    # Handle parentheses by finding the outermost level
    if formula_str.startswith('(') and formula_str.endswith(')'):
        # Check if this is the outermost parentheses
        paren_count = 0
        for i, char in enumerate(formula_str):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count == 0 and i < len(formula_str) - 1:
                    # Not the outermost parentheses
                    break
        else:
            # This is the outermost parentheses
            return simple_parse_formula(formula_str[1:-1])
    
    # Handle negation
    if formula_str.startswith('¬'):
        return Negation(simple_parse_formula(formula_str[1:]))
    
    # Handle operators with proper precedence (implication has lowest precedence)
    # Find the rightmost operator at the current precedence level
    
    # First, try implication (lowest precedence)
    paren_count = 0
    for i in range(len(formula_str) - 1, -1, -1):
        char = formula_str[i]
        if char == ')':
            paren_count += 1
        elif char == '(':
            paren_count -= 1
        elif char == '→' and paren_count == 0:
            # Found implication at top level
            left = formula_str[:i].strip()
            right = formula_str[i+1:].strip()
            return Implication(simple_parse_formula(left), simple_parse_formula(right))
    
    # Then try conjunction
    paren_count = 0
    for i in range(len(formula_str) - 1, -1, -1):
        char = formula_str[i]
        if char == ')':
            paren_count += 1
        elif char == '(':
            paren_count -= 1
        elif char == '∧' and paren_count == 0:
            # Found conjunction at top level
            left = formula_str[:i].strip()
            right = formula_str[i+1:].strip()
            return Conjunction([simple_parse_formula(left), simple_parse_formula(right)])
    
    # Then try disjunction
    paren_count = 0
    for i in range(len(formula_str) - 1, -1, -1):
        char = formula_str[i]
        if char == ')':
            paren_count += 1
        elif char == '(':
            paren_count -= 1
        elif char == '∨' and paren_count == 0:
            # Found disjunction at top level
            left = formula_str[:i].strip()
            right = formula_str[i+1:].strip()
            return Disjunction([simple_parse_formula(left), simple_parse_formula(right)])
    
    # Handle atomic formula
    return AtomicFormula(Atom(formula_str))

class PropositionalLogicConverter:
    """Original propositional logic converter for backward compatibility"""
    
    def __init__(self):
        self.nlp = nlp
        self.formula_utils = FormulaUtils()

    def normalize_atom(self, text: str) -> str:
        """Convert text to a valid atom identifier."""
        text = re.sub(r'\b(a|an|the|is|are|was|were|will be|can|do|does|did)\b', '', text, flags=re.IGNORECASE)
        text = text.strip()
        return re.sub(r'[^a-z0-9_]', '', text.lower().replace(" ", "_"))

    def extract_base_proposition(self, doc_or_span: Any) -> tuple[str, bool]:
        """Extract a base proposition from a spaCy Doc or Span, and detect negation."""
        has_negation = False
        root_token = None
        
        for token in doc_or_span:
            if token.dep_ == "neg" or token.text.lower() in ["not", "no", "never"]:
                has_negation = True
            if token.dep_ == "ROOT":
                root_token = token

        prop_text_parts = []
        if root_token:
            if root_token.lemma_ == "be":
                subject = [token for token in root_token.children if token.dep_ == "nsubj"]
                attribute = [token for token in root_token.children if token.dep_ in ["acomp", "attr", "advmod", "dobj"]]
                
                if subject and attribute:
                    prop_text_parts.append(subject[0].text)
                    prop_text_parts.append(attribute[0].text)
                elif attribute:
                    prop_text_parts.append(attribute[0].text)
                elif subject:
                    prop_text_parts.append(subject[0].text)
                    prop_text_parts.append(root_token.lemma_)
                else:
                    prop_text_parts.append(root_token.lemma_)
            else:
                subject = [token for token in root_token.children if token.dep_ == "nsubj"]
                direct_object = [token for token in root_token.children if token.dep_ == "dobj"]
                
                if subject:
                    prop_text_parts.append(subject[0].text)
                prop_text_parts.append(root_token.lemma_)
                if direct_object:
                    prop_text_parts.append(direct_object[0].text)
        else:
            prop_text_parts.append(doc_or_span.text)

        base_prop_id = self.normalize_atom("_".join(prop_text_parts))
        return base_prop_id, has_negation

    def convert_text_to_logic(self, text: str) -> Dict[str, Any]:
        """Converts natural language text to a propositional logic formula."""
        doc = self.nlp(text)
        formula_str = ""
        confidence = 0.6

        # Check for conjunctions (and, but)
        conjunctions = [token for token in doc if token.pos_ == "CCONJ" and token.text.lower() in ["and", "but"]]
        if conjunctions:
            parts = re.split(r'\b(and|but)\b', text, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) == 3:
                left_text = parts[0].strip()
                right_text = parts[2].strip()
                
                left_prop_id, left_neg = self.extract_base_proposition(self.nlp(left_text))
                right_prop_id, right_neg = self.extract_base_proposition(self.nlp(right_text))
                
                left_formula = Negation(AtomicFormula(left_prop_id)) if left_neg else AtomicFormula(left_prop_id)
                right_formula = Negation(AtomicFormula(right_prop_id)) if right_neg else AtomicFormula(right_prop_id)
                
                formula_str = str(Conjunction([left_formula, right_formula]))
                confidence = 0.8
            else:
                prop_id, has_neg = self.extract_base_proposition(doc)
                formula_str = str(Negation(AtomicFormula(prop_id))) if has_neg else str(AtomicFormula(prop_id))
                confidence = 0.6
        
        # Check for disjunctions (or)
        elif any(token.text.lower() == "or" for token in doc if token.pos_ == "CCONJ"):
            parts = re.split(r'\b(or)\b', text, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) == 3:
                left_text = parts[0].strip()
                right_text = parts[2].strip()
                
                left_prop_id, left_neg = self.extract_base_proposition(self.nlp(left_text))
                right_prop_id, right_neg = self.extract_base_proposition(self.nlp(right_text))
                
                left_formula = Negation(AtomicFormula(left_prop_id)) if left_neg else AtomicFormula(left_prop_id)
                right_formula = Negation(AtomicFormula(right_prop_id)) if right_neg else AtomicFormula(right_prop_id)
                
                formula_str = str(Disjunction([left_formula, right_formula]))
                confidence = 0.8
            else:
                prop_id, has_neg = self.extract_base_proposition(doc)
                formula_str = str(Negation(AtomicFormula(prop_id))) if has_neg else str(AtomicFormula(prop_id))
                confidence = 0.6

        # Check for conditionals (if...then)
        elif re.search(r'\bif\b.*\bthen\b', text, re.IGNORECASE):
            match = re.search(r'\bif\b\s*(.*?)\s*\bthen\b\s*(.*)', text, re.IGNORECASE)
            if match:
                antecedent_text = match.group(1).strip()
                consequent_text = match.group(2).strip()
                
                antecedent_prop_id, ant_neg = self.extract_base_proposition(self.nlp(antecedent_text))
                consequent_prop_id, cons_neg = self.extract_base_proposition(self.nlp(consequent_text))
                
                antecedent_formula = Negation(AtomicFormula(antecedent_prop_id)) if ant_neg else AtomicFormula(antecedent_prop_id)
                consequent_formula = Negation(AtomicFormula(consequent_prop_id)) if cons_neg else AtomicFormula(consequent_prop_id)
                
                formula_str = str(Implication(antecedent_formula, consequent_formula))
                confidence = 0.8
            else:
                prop_id, has_neg = self.extract_base_proposition(doc)
                formula_str = str(Negation(AtomicFormula(prop_id))) if has_neg else str(AtomicFormula(prop_id))
                confidence = 0.6

        # Check for biconditionals (if and only if, iff)
        elif re.search(r'\b(if and only if|iff)\b', text, re.IGNORECASE):
            parts = re.split(r'\b(if and only if|iff)\b', text, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) == 3:
                left_text = parts[0].strip()
                right_text = parts[2].strip()
                
                left_prop_id, left_neg = self.extract_base_proposition(self.nlp(left_text))
                right_prop_id, right_neg = self.extract_base_proposition(self.nlp(right_text))
                
                left_formula = Negation(AtomicFormula(left_prop_id)) if left_neg else AtomicFormula(left_prop_id)
                right_formula = Negation(AtomicFormula(right_prop_id)) if right_neg else AtomicFormula(right_prop_id)
                
                formula_str = str(Biconditional(left_formula, right_formula))
                confidence = 0.8
            else:
                prop_id, has_neg = self.extract_base_proposition(doc)
                formula_str = str(Negation(AtomicFormula(prop_id))) if has_neg else str(AtomicFormula(prop_id))
                confidence = 0.6

        else:
            # Default to atomic formula with negation check
            prop_id, has_neg = self.extract_base_proposition(doc)
            formula_str = str(Negation(AtomicFormula(prop_id))) if has_neg else str(AtomicFormula(prop_id))
            confidence = 0.6

        # Extract atoms from the formula string (simple approach)
        try:
            # Simple atom extraction - find all identifiers that aren't operators
            atom_pattern = r'\b[a-z_][a-z0-9_]*\b'
            atoms = list(set(re.findall(atom_pattern, formula_str.lower())))
            # Remove common operators and keywords
            atoms = [atom for atom in atoms if atom not in ['and', 'or', 'not', 'if', 'then', 'iff']]
        except Exception as e:
            print(f"Atom extraction error: {e}")
            # Fallback: create a simple atomic formula
            fallback_atom = self.normalize_atom(text)
            formula_str = fallback_atom
            atoms = [fallback_atom]
            confidence = 0.3

        return {
            "original_text": text,
            "propositional_formula": formula_str,
            "confidence": confidence,
            "atoms": atoms,
            "semantic_analysis": {
                "structure_detected": "Available",
                "spacy_processing": "Enabled",
                "negation_handling": "Fixed"
            }
        }

propositional_converter = PropositionalLogicConverter()

class ConversionRequest(BaseModel):
    text: str
    logic_type: str = "auto"  # "propositional", "first_order", or "auto"
    include_cnf: bool = False
    include_dnf: bool = False
    include_truth_table: bool = False

class InferenceRequest(BaseModel):
    premises: List[str]
    conclusion: str
    logic_type: str = "auto"  # "propositional", "first_order", or "auto"

def detect_logic_type(text: str) -> str:
    """Detect whether text requires propositional or first-order logic"""
    text_lower = text.lower()
    
    # First-order logic indicators
    fol_indicators = [
        'all', 'every', 'each', 'any',  # Universal quantifiers
        'some', 'there exists', 'at least one',  # Existential quantifiers
    ]
    
    # Check for quantifiers
    if any(indicator in text_lower for indicator in fol_indicators):
        return "first_order"
    
    # Check for proper names (individual constants)
    # Look for capitalized words that are likely proper names
    words = text.split()
    for word in words:
        # Skip common capitalized words at start of sentences
        if word[0].isupper() and word.isalpha() and len(word) > 2:
            # Check if it's likely a proper name (not a common word)
            common_words = {'the', 'this', 'that', 'these', 'those', 'there', 'here', 'where', 'when', 'why', 'how', 'what', 'who', 'which'}
            if word.lower() not in common_words:
                return "first_order"
    
    return "propositional"

@app.get("/")
async def root():
    return {
        "message": "Enhanced First-Order Logic API",
        "version": app.version,
        "status": "running",
        "features": [
            "spaCy semantic parsing",
            "Propositional logic support",
            "First-order logic with quantifiers",
            "Universal quantifiers (∀)",
            "Existential quantifiers (∃)",
            "Individual constants",
            "Predicate logic",
            "Advanced inference engine",
            "Automatic logic type detection",
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
    return {"status": "healthy", "message": "Enhanced FOL API is up and running!"}

@app.get("/test")
async def test_logic_engine():
    # Test propositional logic
    prop_formula_str = "(p → q)"
    prop_formula = formula_utils.parse_formula(prop_formula_str)
    prop_truth_table = formula_utils.generate_truth_table(prop_formula)
    
    # Test first-order logic
    fol_result = fol_converter.convert_text_to_first_order_logic("All humans are mortal")
    
    return {
        "message": "Enhanced system is working!",
        "propositional_test": {
            "formula": str(prop_formula),
            "is_tautology": prop_truth_table['is_tautology'],
            "is_contradiction": prop_truth_table['is_contradiction'],
        },
        "first_order_test": {
            "input": "All humans are mortal",
            "formula": fol_result['first_order_formula'],
            "type": fol_result['formula_type'],
            "variables": fol_result['variables'],
            "constants": fol_result['constants']
        }
    }

@app.post("/convert")
async def convert(request: ConversionRequest):
    print(f"Enhanced API: Converting text: '{request.text}'")
    
    # Determine logic type
    if request.logic_type == "auto":
        logic_type = detect_logic_type(request.text)
    else:
        logic_type = request.logic_type
    
    print(f"Enhanced API: Detected logic type: {logic_type}")
    
    if logic_type == "first_order":
        result = fol_converter.convert_text_to_first_order_logic(request.text)
        result["logic_type"] = "first_order"
        result["detected_logic_type"] = logic_type
        
        # Add CNF/DNF if requested (for first-order, this is more complex)
        if request.include_cnf or request.include_dnf:
            try:
                # For now, we'll add a note that CNF/DNF for FOL is complex
                result["note"] = "CNF/DNF conversion for first-order logic is complex and not yet implemented"
            except Exception as e:
                result["error"] = f"CNF/DNF conversion failed: {e}"
        
        # Truth table for first-order logic is not directly applicable
        if request.include_truth_table:
            result["note"] = "Truth tables are not directly applicable to first-order logic with quantifiers"
    
    else:  # propositional
        result = propositional_converter.convert_text_to_logic(request.text)
        result["logic_type"] = "propositional"
        result["detected_logic_type"] = logic_type
        
        # Add CNF/DNF and truth table for propositional logic
        if request.include_cnf or request.include_dnf or request.include_truth_table:
            try:
                formula_obj = simple_parse_formula(result['propositional_formula'])
                
                if request.include_cnf:
                    result['cnf_formula'] = str(formula_obj.to_cnf())
                if request.include_dnf:
                    result['dnf_formula'] = str(formula_obj.to_dnf())
                if request.include_truth_table:
                    tt_summary = formula_utils.generate_truth_table(formula_obj)
                    result['truth_table'] = {
                        "atoms": [str(a) for a in tt_summary['atoms']],
                        "rows": [{"values": r['values'], "result": r['result']} for r in tt_summary['rows']],
                        "is_tautology": tt_summary['is_tautology'],
                        "is_contradiction": tt_summary['is_contradiction'],
                        "is_satisfiable": tt_summary['is_satisfiable'],
                    }
            except Exception as e:
                result["error"] = f"Formula processing failed: {e}"
    
    print(f"Enhanced API: Result: {result.get('propositional_formula', result.get('first_order_formula', 'N/A'))}")
    return result

@app.post("/infer")
async def check_inference(request: InferenceRequest):
    print(f"Enhanced API: Inference request - Premises: {request.premises}, Conclusion: {request.conclusion}")
    
    # Determine logic type
    if request.logic_type == "auto":
        # Check all premises and conclusion
        all_texts = request.premises + [request.conclusion]
        logic_types = [detect_logic_type(text) for text in all_texts]
        logic_type = "first_order" if "first_order" in logic_types else "propositional"
    else:
        logic_type = request.logic_type
    
    print(f"Enhanced API: Using logic type: {logic_type}")
    
    if logic_type == "first_order":
        # For first-order logic, we'll use a simplified approach
        # In a full implementation, this would require more sophisticated reasoning
        
        premise_formulas_str = []
        for p_text in request.premises:
            converted_p = fol_converter.convert_text_to_first_order_logic(p_text)
            premise_formulas_str.append(converted_p['first_order_formula'])
        
        converted_c = fol_converter.convert_text_to_first_order_logic(request.conclusion)
        conclusion_formula_str = converted_c['first_order_formula']
        
        return {
            "valid": "unknown",  # First-order inference is complex
            "premises": premise_formulas_str,
            "conclusion": conclusion_formula_str,
            "logic_type": "first_order",
            "explanation": "First-order logic inference requires sophisticated theorem proving. This is a simplified representation.",
            "note": "Full first-order inference engine not yet implemented",
            "error": None
        }
    
    else:  # propositional logic
        premise_formulas_str = []
        for p_text in request.premises:
            converted_p = propositional_converter.convert_text_to_logic(p_text)
            premise_formulas_str.append(converted_p['propositional_formula'])
        
        converted_c = propositional_converter.convert_text_to_logic(request.conclusion)
        conclusion_formula_str = converted_c['propositional_formula']

        # Combine premises with AND
        combined_premises_str = " ∧ ".join(premise_formulas_str)
        
        # Check if (P1 ∧ P2 ∧ ... ∧ Pn) → C is a tautology
        implication_str = f"({combined_premises_str}) → ({conclusion_formula_str})"
        print(f"Enhanced API: Implication: {implication_str}")

        try:
            print(f"Enhanced API: Parsing implication: {implication_str}")
            implication_formula_obj = simple_parse_formula(implication_str)
            truth_table_summary = formula_utils.generate_truth_table(implication_formula_obj)
            is_valid = truth_table_summary['is_tautology']
            
            counterexample = None
            if not is_valid:
                for row in truth_table_summary['rows']:
                    if not row['result']:
                        counterexample = {
                            "atoms": [str(a) for a in truth_table_summary['atoms']],
                            "values": row['values'],
                            "description": f"Counterexample: {dict(zip([str(a) for a in truth_table_summary['atoms']], ['True' if v else 'False' for v in row['values']]))}"
                        }
                        break
            
            explanation = f"✓ Valid inference: The conclusion logically follows from the premises." if is_valid else f"✗ Invalid inference: The conclusion does not necessarily follow. {counterexample.get('description', 'No specific counterexample found.') if counterexample else 'No specific counterexample found.'}"

            print(f"Enhanced API: Is valid: {is_valid}")

            return {
                "valid": is_valid,
                "premises": premise_formulas_str,
                "conclusion": conclusion_formula_str,
                "implication": implication_str,
                "logic_type": "propositional",
                "explanation": explanation,
                "counterexample": counterexample,
                "truth_table_summary": {
                    "is_tautology": truth_table_summary['is_tautology'],
                    "is_contradiction": truth_table_summary['is_contradiction'],
                    "is_satisfiable": truth_table_summary['is_satisfiable'],
                },
                "error": None
            }
        except Exception as e:
            print(f"Enhanced API: Inference error: {e}")
            print(f"Enhanced API: Problematic implication: {implication_str}")
            raise HTTPException(status_code=400, detail=f"Error processing inference: {e}. Implication: {implication_str}")

@app.get("/examples")
async def get_examples():
    return {
        "propositional_examples": [
            {"text": "It is raining and the ground is wet", "logic": "(it_rain ∧ wet)"},
            {"text": "If it rains, then the ground will be wet", "logic": "(it_rain → wet)"},
            {"text": "Either it is sunny or it is cloudy", "logic": "(sunny ∨ cloudy)"},
            {"text": "It is not raining", "logic": "¬it_rain"},
            {"text": "The sky is blue if and only if it is daytime", "logic": "(sky_blue ↔ daytime)"}
        ],
        "first_order_examples": [
            {"text": "All humans are mortal", "logic": "∀h((humans(h) → mortal(h)))"},
            {"text": "Socrates is human", "logic": "human(Socrates)"},
            {"text": "Some birds cannot fly", "logic": "∃b((birds(b) ∧ ¬fly(b)))"},
            {"text": "Every student studies", "logic": "∀s((student(s) → studies(s)))"},
            {"text": "There exists a student who passed", "logic": "∃s((student(s) ∧ passed(s)))"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
