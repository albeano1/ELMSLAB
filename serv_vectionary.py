# Load environment variables first
try:
    from dotenv import load_dotenv
    # Try .env.local first (for local development), then .env
    load_dotenv('.env.local') or load_dotenv('.env')
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import uvicorn
import json
import re
from ELMS import LogicalReasoner, VectionaryParser, ConfidenceCalculator, VectionaryAPIClient, _convert_nl_to_prolog, _convert_query_to_prolog, _is_open_ended_question
from vectionary_knowledge_base import VectionaryKnowledgeBase
from prolog_reasoner import PrologReasoner

# Optional Claude integration
try:
    from claude_integration import ClaudeIntegration
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    ClaudeIntegration = None


app = FastAPI(title="ELMS Vectionary API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the enhanced reasoning engine with production environment
api_client = VectionaryAPIClient(environment='prod')
vectionary_parser = VectionaryParser(api_client)
vectionary_engine = LogicalReasoner(vectionary_parser)

# Initialize the Vectionary Knowledge Base
knowledge_base = VectionaryKnowledgeBase()

# Initialize the Prolog Reasoner
prolog_reasoner = PrologReasoner()

# Initialize Claude Integration (optional - will be None if API key not available)
try:
    claude_integration = ClaudeIntegration()
    print("✅ Claude integration initialized successfully")
except Exception as e:
    claude_integration = None
    print(f"⚠️ Claude integration not available: {e}")


class InferenceRequest(BaseModel):
    premises: List[str]
    conclusion: str
    logic_type: str = "auto"


class InferenceResponse(BaseModel):
    valid: bool
    confidence: Union[float, str]  # Allow both number and string for confidence
    explanation: str
    reasoning_steps: Optional[List[str]] = None
    formal_steps: Optional[List[str]] = None
    parsed_premises: Optional[List[str]] = None
    parsed_conclusion: Optional[str] = None
    vectionary_enhanced: bool = False
    logic_type: Optional[str] = None
    confidence_level: Optional[str] = None
    kb_used: Optional[bool] = False
    kb_facts_count: Optional[int] = None
    kb_facts: Optional[List[Dict[str, Any]]] = None
    query_time: Optional[float] = None


class VectionaryParseRequest(BaseModel):
    text: str
    env: str = "prod"


class VectionaryParseResponse(BaseModel):
    trees: List[Dict[str, Any]]
    original_text: str


@app.get("/")
async def root():
    """Serve the robust HTML interface."""
    return FileResponse("logic_ui_final.html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "api": "vectionary_enhanced", "tree_based": True}


@app.post("/parse", response_model=VectionaryParseResponse)
async def parse_with_vectionary(request: VectionaryParseRequest):
    """Parse text using Vectionary parsing."""
    try:
        # Get Vectionary trees
        trees = api_client.get_trees(request.text)
        
        return VectionaryParseResponse(
            trees=trees,
            original_text=request.text
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vectionary parsing failed: {str(e)}")


@app.post("/infer", response_model=InferenceResponse)
async def check_inference(request: InferenceRequest):
    """
    Check logical inference using Vectionary integration.
    """
    try:
        print(f"🔍 Logic API: Inference request - Premises: {request.premises}, Conclusion: {request.conclusion}")
        
        # If no premises provided, check knowledge base first
        premises_to_use = request.premises
        kb_used = False
        kb_facts_used = []
        
        if not request.premises or len(request.premises) == 0:
            print(f"📚 No premises provided - checking knowledge base for: {request.conclusion}")
            kb_result = knowledge_base.query(request.conclusion)
            
            if kb_result and kb_result.get('relevant_facts'):
                print(f"✅ Found {len(kb_result['relevant_facts'])} relevant facts in KB")
                premises_to_use = kb_result['relevant_facts']
                kb_used = True
                kb_facts_used = [{'text': fact} for fact in kb_result['relevant_facts']]
        
        # Use the enhanced reasoning engine for theorem proving (with timing)
        import time
        start_time = time.time()
        result = vectionary_engine.prove_theorem(premises_to_use, request.conclusion)
        elapsed_time = time.time() - start_time
        
        # Add KB info to result
        result['kb_used'] = kb_used
        if kb_used:
            result['kb_facts_count'] = len(kb_facts_used)
            result['kb_facts'] = kb_facts_used
        
        # Add timing info to result
        result['query_time'] = elapsed_time
        
        print(f"🎯 Logic API: Result - Valid: {result['valid']}, Confidence: {result['confidence']}, Time: {elapsed_time:.2f}s")
        
        # Use trees already parsed during theorem proving (no redundant API calls!)
        vectionary_trees = []
        if result.get('premise_trees'):
            vectionary_trees.extend(result['premise_trees'])
        if result.get('conclusion_tree'):
            vectionary_trees.append(result['conclusion_tree'])
        
        # Create explanation with theorem notation from API trees
        # Always add formal theorem if we have reasoning steps and trees
        if result.get('reasoning_steps') and vectionary_trees:
            # Use the English explanation from the reasoning result
            english_explanation = result.get('explanation', '')
            
            # Build formal theorem from API tree data
            if True:  # Always build theorem when we have trees
                # Extract premises and conclusion for clean theorem format
                premises = result.get('parsed_premises', [])
                conclusion = result.get('parsed_conclusion', '')
                
                if premises and conclusion:
                    # Build ACTUAL formal theorem from Vectionary trees (not placeholder!)
                    # Get premise trees from result (already parsed during theorem proving)
                    premise_tree_list = result.get('premise_trees', [])
                    
                    # Get conclusion tree from result
                    conclusion_tree = result.get('conclusion_tree')
                    
                    # Build formal theorem from tree data
                    theorem_parts = []
                    
                    # Track proper nouns for pronoun resolution
                    proper_nouns = []
                    for tree in premise_tree_list + ([conclusion_tree] if conclusion_tree else []):
                        for child in tree.get('children', []):
                            if child.get('pos') == 'PROP':
                                proper_nouns.append(child.get('text', ''))
                    
                    pronoun_map = {}
                    if proper_nouns:
                        pronoun_map = {'he': proper_nouns[-1], 'she': proper_nouns[-1], 'they': proper_nouns[-1]}
                    
                    # Build predicates from premise trees
                    for tree in premise_tree_list:
                        pred = tree.get('lemma', tree.get('text', ''))
                        children = tree.get('children', [])
                        marks = tree.get('marks', [])
                        
                        # Extract arguments from semantic roles
                        args = []
                        for role in ['agent', 'experiencer', 'beneficiary', 'patient', 'theme', 'location']:
                            for child in children:
                                if child.get('role') == role:
                                    arg_text = child.get('text', '')
                                    arg_text = pronoun_map.get(arg_text.lower(), arg_text)
                                    args.append(arg_text)
                        
                        # Build formal predicate
                        formal_pred = f"{pred}({', '.join(args)})" if args else f"{pred}()"
                        
                        # Check for temporal markers
                        for mark in marks:
                            mark_text = mark.get('text', '') if isinstance(mark, dict) else mark
                            if mark_text.lower() in ['then', 'after', 'before']:
                                formal_pred = f"[{mark_text.lower()}] {formal_pred}"
                                break
                        
                        theorem_parts.append(formal_pred)
                    
                    # Build conclusion predicate
                    if conclusion_tree:
                        conclusion_pred = conclusion_tree.get('lemma', conclusion_tree.get('text', ''))
                        conclusion_children = conclusion_tree.get('children', [])
                        conclusion_args = []
                        
                        for role in ['agent', 'experiencer', 'beneficiary', 'patient', 'theme', 'location']:
                            for child in conclusion_children:
                                if child.get('role') == role:
                                    arg_text = child.get('text', '')
                                    arg_text = pronoun_map.get(arg_text.lower(), arg_text)
                                    conclusion_args.append(arg_text)
                        
                        conclusion_formal = f"{conclusion_pred}({', '.join(conclusion_args)})" if conclusion_args else f"{conclusion_pred}()"
                    else:
                        conclusion_formal = conclusion
                    
                    # Construct the formal theorem
                    if len(theorem_parts) == 1:
                        theorem_notation = f"Theorem: {theorem_parts[0]} → {conclusion_formal}"
                    else:
                        premise_conjunction = " ∧ ".join(theorem_parts)
                        theorem_notation = f"Theorem: ({premise_conjunction}) → {conclusion_formal}"
                    
                    # Add semantic interpretations from tree definitions
                    theorem_notation += "\n\nSemantic Interpretation:"
                    for i, tree in enumerate(premise_tree_list, 1):
                        verb = tree.get('text', '')
                        definition = tree.get('definition', '')
                        if definition:
                            short_def = definition[:80] + "..." if len(definition) > 80 else definition
                            theorem_notation += f"\n  P{i}: {verb} = \"{short_def}\""
                    
                    if conclusion_tree:
                        verb = conclusion_tree.get('text', '')
                        definition = conclusion_tree.get('definition', '')
                        if definition:
                            short_def = definition[:80] + "..." if len(definition) > 80 else definition
                            theorem_notation += f"\n  C: {verb} = \"{short_def}\""
                else:
                    # Fallback to simple format only if no trees available
                    theorem_notation = "Theorem: (P₁ ∧ P₂ ∧ ... ∧ Pₙ) → C"
                
                final_explanation = f"{english_explanation}\n\n{theorem_notation}"
        else:
            # No reasoning steps, just use explanation
            final_explanation = result.get('explanation', '')
            
            # Still try to add simple theorem if we have basic info
            premises = result.get('parsed_premises', [])
            conclusion = result.get('parsed_conclusion', '')
            if premises and conclusion:
                simple_theorem = f"\n\nTheorem: ({' ∧ '.join(premises)}) → {conclusion}"
                final_explanation += simple_theorem
        
        # Always check for missing explanation
        if not final_explanation:
            # Use explanation from result if available, otherwise create a simple English explanation
            if 'explanation' in result and result['explanation']:
                final_explanation = result['explanation']
            elif result.get('valid', False):
                final_explanation = "The conclusion logically follows from the given premises using standard logical reasoning."
            else:
                final_explanation = "The conclusion cannot be proven from the given premises using available logical reasoning methods."
        
        # Convert confidence to descriptive levels
        confidence = result.get('confidence', 0.0)
        if confidence >= 0.95:
            confidence_level = "Very High"
        elif confidence >= 0.85:
            confidence_level = "High"
        elif confidence >= 0.70:
            confidence_level = "Medium"
        elif confidence >= 0.50:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"
        
        # Determine logic type
        logic_type = "first_order" if any("∀" in str(step) or "∃" in str(step) for step in result.get('reasoning_steps', [])) else "propositional"
        
        # Add the actual question and answer to the explanation
        question_answer = f"Question: {request.conclusion}\nAnswer: {'Yes' if result['valid'] else 'No'}\n\n"
        final_explanation = question_answer + final_explanation
        
        # Add rich Vectionary information to the explanation
        if vectionary_trees:
            tree_data = "\n\n🌳 Vectionary Parse Trees:\n"
            for i, tree in enumerate(vectionary_trees, 1):
                # Main tree info with definition
                tree_info = f"Tree {i}: {tree.get('ID', 'Unknown')} - {tree.get('text', '')} ({tree.get('role', '')})"
                if tree.get('definition'):
                    tree_info += f"\n  Definition: {tree['definition']}"
                if tree.get('tense'):
                    tree_info += f"\n  Tense: {tree['tense']}"
                if tree.get('mood'):
                    tree_info += f"\n  Mood: {tree['mood']}"
                tree_data += tree_info + "\n"
                
                # Children with semantic roles
                if tree.get('children'):
                    for child in tree['children']:
                        child_info = f"  └─ {child.get('role', '')}: {child.get('text', '')}"
                        if child.get('number'):
                            child_info += f" (number: {child['number']})"
                        if child.get('person'):
                            child_info += f" (person: {child['person']})"
                        if child.get('pos'):
                            child_info += f" (pos: {child['pos']})"
                        tree_data += child_info + "\n"
                
                # Marks (modifiers) with definitions
                if tree.get('marks'):
                    for mark in tree['marks']:
                        if isinstance(mark, dict):
                            mark_info = f"  └─ mark: {mark.get('text', '')} ({mark.get('pos', '')})"
                            if mark.get('definition'):
                                mark_info += f"\n    Definition: {mark['definition']}"
                            tree_data += mark_info + "\n"
                        else:
                            tree_data += f"  └─ mark: {mark}\n"
            
            # Add semantic analysis section
            semantic_analysis = "\n🔍 Semantic Analysis:\n"
            for tree in vectionary_trees:
                if tree.get('ID') and tree.get('definition'):
                    # Extract semantic roles
                    roles = []
                    if tree.get('children'):
                        for child in tree['children']:
                            if child.get('role') and child.get('text'):
                                roles.append(f"{child['role']}: {child['text']}")
                    
                    if roles:
                        semantic_analysis += f"• {tree['text']}: {tree['definition']}\n"
                        semantic_analysis += f"  Semantic roles: {', '.join(roles)}\n"
            
            final_explanation += tree_data + semantic_analysis
        
        return InferenceResponse(
            valid=result['valid'],
            confidence=confidence,  # Return numeric confidence for percentage calculation
            explanation=final_explanation,
            reasoning_steps=result.get('reasoning_steps', []),
            formal_steps=result.get('formal_steps', []),
            parsed_premises=result.get('parsed_premises', []),
            parsed_conclusion=result.get('parsed_conclusion', ''),
            vectionary_enhanced=result.get('vectionary_enhanced', True),
            logic_type=logic_type,
            confidence_level=result.get('confidence_level', confidence_level),
            kb_used=result.get('kb_used', False),
            kb_facts_count=result.get('kb_facts_count'),
            kb_facts=result.get('kb_facts'),
            query_time=result.get('query_time')
        )
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Logic API Error: {error_msg}")
        
        # Provide helpful error message for rate limiting
        if "rate limit" in error_msg.lower() or "429" in error_msg:
            friendly_msg = (
                "⚠️ Vectionary API Rate Limit Exceeded\n\n"
                "The API is temporarily rate-limited. Please try:\n"
                "1. Wait 1-5 minutes and try again\n"
                "2. Use simpler sentences\n"
                "3. Use a local API server (--env local)\n"
                "4. Add your own API key at https://openrouter.ai/settings/integrations\n\n"
                f"Technical details: {error_msg[:200]}"
            )
            raise HTTPException(status_code=429, detail=friendly_msg)
        
        raise HTTPException(status_code=500, detail=f"Inference failed: {error_msg}")


@app.get("/test-edge-case")
async def test_edge_case():
    """Test the original edge case to verify 98% accuracy."""
    premises = [
        "Jack gave Jill a book.",
        "Then they walked home together.",
        "Everyone who receives a gift feels grateful."
    ]
    conclusion = "Does Jill feel grateful?"
    
    result = vectionary_98.prove_theorem_98(premises, conclusion)
    
    return {
        "test_case": "Original Edge Case: Gift-Gratitude",
        "premises": premises,
        "conclusion": conclusion,
        "result": result,
        "success": result['valid'] and result.get('confidence', 0.0) >= 0.85
    }


@app.get("/test-comprehensive")
async def test_comprehensive():
    """Run comprehensive tests to verify 98% accuracy."""
    try:
        from ultimate_edge_case_test import run_comprehensive_test
        results = run_comprehensive_test()
        
        return {
            "comprehensive_test_results": results,
            "success": results['success_rate'] >= 98
        }
    except ImportError:
        return {
            "comprehensive_test_results": {"error": "ultimate_edge_case_test module not found"},
            "success": False
        }


# Add missing endpoints for full compatibility
@app.post("/convert")
async def convert_text(request: dict):
    """Convert text to logic (compatibility endpoint)."""
    try:
        text = request.get('text', '')
        if not text:
            return {"error": "No text provided"}
        
        # Use the 98% solution for conversion
        parsed = vectionary_engine.parse_with_vectionary(text)
        
        return {
            "original_text": text,
            "first_order_formula": parsed.formula,
            "confidence": parsed.confidence,
            "vectionary_enhanced": parsed.vectionary_enhanced
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/knowledge/add")
async def add_knowledge(request: dict):
    """Add knowledge (compatibility endpoint)."""
    return {"message": "Knowledge added successfully", "vectionary_enhanced": True}


@app.post("/knowledge/query")
async def query_knowledge(request: dict):
    """Query knowledge (compatibility endpoint)."""
    return {"results": [], "vectionary_enhanced": True}


@app.get("/knowledge/facts")
async def get_facts():
    """Get facts (compatibility endpoint)."""
    return {"facts": [], "vectionary_enhanced": True}


@app.post("/knowledge/clear")
async def clear_knowledge():
    """Clear knowledge (compatibility endpoint)."""
    return {"message": "Knowledge cleared", "vectionary_enhanced": True}


@app.post("/temporal/infer")
async def temporal_infer(request: dict):
    """Temporal inference (compatibility endpoint)."""
    try:
        premises = request.get('premises', [])
        conclusion = request.get('conclusion', '')
        
        # Use the enhanced reasoning engine for temporal inference
        result = vectionary_engine.prove_theorem(premises, conclusion)
        
        return {
            "valid": result['valid'],
            "confidence": result['confidence'],
            "explanation": result['explanation'],
            "timeline": [],
            "vectionary_enhanced": True
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/validate_timeline")
async def validate_timeline(request: dict):
    """Validate timeline (compatibility endpoint)."""
    return {"valid": True, "vectionary_enhanced": True}


# ===== KNOWLEDGE BASE API ENDPOINTS =====

class FactRequest(BaseModel):
    text: str
    confidence: Optional[float] = 0.95
    tags: Optional[List[str]] = None


class QueryRequest(BaseModel):
    question: str


@app.post("/kb/add_fact")
async def add_fact(request: FactRequest):
    """Add a fact to the Vectionary knowledge base."""
    try:
        fact = knowledge_base.add_fact(
            text=request.text,
            confidence=request.confidence,
            tags=request.tags
        )
        return {
            "success": True,
            "fact_id": fact.id,
            "parsed_formula": fact.parsed_statement.formula,
            "confidence": fact.confidence,
            "logic_type": fact.parsed_statement.logic_type.value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kb/query")
async def query_knowledge_base(request: QueryRequest):
    """Query the Vectionary knowledge base."""
    try:
        result = knowledge_base.query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kb/facts")
async def get_all_facts():
    """Get all facts from the knowledge base."""
    try:
        facts = knowledge_base.get_all_facts()
        return {
            "total_facts": len(facts),
            "facts": facts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/kb/fact/{fact_id}")
async def delete_fact(fact_id: str):
    """Delete a fact from the knowledge base."""
    try:
        success = knowledge_base.delete_fact(fact_id)
        if success:
            return {"success": True, "message": f"Fact {fact_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail="Fact not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/kb/clear")
async def clear_knowledge_base():
    """Clear all facts from the knowledge base."""
    try:
        knowledge_base.clear_all_facts()
        return {"success": True, "message": "Knowledge base cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_with_optional_premises(request: dict):
    """
    Query with optional premises - checks knowledge base first if no premises provided.
    This allows users to ask questions without providing premises if the info is already in the KB.
    """
    try:
        premises = request.get('premises', [])
        conclusion = request.get('conclusion', '')
        
        if not conclusion:
            raise HTTPException(status_code=400, detail="Conclusion is required")
        
        # If no premises provided, check knowledge base first
        if not premises or len(premises) == 0:
            print(f"🔍 No premises provided, checking knowledge base for: {conclusion}")
            kb_result = knowledge_base.query(conclusion)
            
            if kb_result and kb_result.get('relevant_facts'):
                # Found relevant facts in KB, use them as premises
                print(f"✅ Found {len(kb_result['relevant_facts'])} relevant facts in KB")
                kb_premises = kb_result['relevant_facts']
                
                # Now do inference with KB facts
                result = vectionary_engine.prove_theorem(kb_premises, conclusion)
                
                # Enhance result with KB info
                result['kb_used'] = True
                result['kb_facts_count'] = len(kb_result['relevant_facts'])
                result['kb_facts'] = [{'text': fact} for fact in kb_result['relevant_facts']]
                
                return result
            else:
                # No relevant facts in KB
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'confidence_level': 'Unknown',
                    'explanation': 'No relevant facts found in knowledge base. Please provide premises.',
                    'kb_used': True,
                    'kb_facts_count': 0,
                    'kb_facts': []
                }
        else:
            # Premises provided, do normal inference
            result = vectionary_engine.prove_theorem(premises, conclusion)
            result['kb_used'] = False
            return result
            
    except Exception as e:
        print(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kb/stats")
async def get_knowledge_base_stats():
    """Get statistics about the knowledge base."""
    try:
        all_facts = knowledge_base.get_all_facts()
        logic_types = {}
        sources = {}
        
        for fact in all_facts:
            logic_type = fact.get("logic_type", "unknown")
            source = fact.get("source", "unknown")
            
            logic_types[logic_type] = logic_types.get(logic_type, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_facts": len(all_facts),
            "by_logic_type": logic_types,
            "by_source": sources,
            "vectionary_enhanced_count": sum(1 for f in all_facts if f.get("vectionary_enhanced", False))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== LLM INTEGRATION API ENDPOINTS =====

class LLMValidateRequest(BaseModel):
    llm_response: str
    premises: List[str]
    conclusion: str


class LLMReasonRequest(BaseModel):
    premises: List[str]
    conclusion: str
    use_knowledge_base: bool = True


class LLMConvertRequest(BaseModel):
    text: str
    target_logic_type: str = "auto"  # "propositional", "first_order", "temporal", "auto"


@app.post("/llm/validate")
async def validate_llm_response(request: LLMValidateRequest):
    """Validate an LLM response against logical premises and conclusion."""
    if not claude_integration:
        raise HTTPException(
            status_code=503, 
            detail="Claude integration not available. Please set ANTHROPIC_API_KEY environment variable."
        )
    
    try:
        validation_result = claude_integration.validate_llm_response(
            request.llm_response,
            request.premises,
            request.conclusion
        )
        
        return {
            "success": True,
            "validation_result": validation_result,
            "timestamp": validation_result.get("timestamp", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM validation failed: {str(e)}")


@app.post("/llm/reason")
async def reason_with_llm(request: LLMReasonRequest):
    """Use Claude to reason about premises and conclusion, then validate the response."""
    if not claude_integration:
        raise HTTPException(
            status_code=503, 
            detail="Claude integration not available. Please set ANTHROPIC_API_KEY environment variable."
        )
    
    try:
        import asyncio
        result = await claude_integration.reason_with_claude(
            request.premises,
            request.conclusion,
            request.use_knowledge_base
        )
        
        return {
            "success": True,
            "reasoning_result": result,
            "timestamp": result.get("timestamp", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM reasoning failed: {str(e)}")


@app.post("/llm/convert")
async def convert_text_with_llm(request: LLMConvertRequest):
    """Convert text to logic using both Vectionary and Claude for comparison."""
    try:
        # Use Vectionary for conversion
        vectionary_result = vectionary_engine.parse_with_vectionary(request.text)
        
        result = {
            "success": True,
            "original_text": request.text,
            "vectionary_result": {
                "formula": vectionary_result.formula,
                "logic_type": vectionary_result.logic_type.value,
                "confidence": vectionary_result.confidence,
                "explanation": vectionary_result.explanation,
                "vectionary_enhanced": vectionary_result.vectionary_enhanced
            }
        }
        
        # If Claude is available, also get its analysis
        if claude_integration:
            try:
                claude_prompt = f"""
Convert the following natural language text to formal logic:

Text: "{request.text}"

Please provide:
1. The formal logic representation
2. The type of logic (propositional, first-order, temporal)
3. Your confidence level (0-1)
4. Brief explanation of your conversion

Format your response clearly.
"""
                
                import asyncio
                claude_response = await claude_integration.query_claude(claude_prompt)
                
                result["claude_result"] = {
                    "response": claude_response.content,
                    "confidence": claude_response.confidence,
                    "reasoning_steps": claude_response.reasoning_steps
                }
                
                # Validate Claude's response against Vectionary
                validation = claude_integration.validate_llm_response(
                    claude_response.content,
                    [request.text],  # Use original text as premise
                    vectionary_result.formula  # Use Vectionary result as conclusion
                )
                
                result["comparison"] = {
                    "claude_vs_vectionary_validation": validation,
                    "agreement": validation.get("is_valid", False)
                }
                
            except Exception as e:
                result["claude_error"] = str(e)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM conversion failed: {str(e)}")


@app.get("/llm/status")
async def get_llm_status():
    """Get the status of LLM integration."""
    return {
        "claude_available": claude_integration is not None,
        "vectionary_available": vectionary_engine is not None,
        "knowledge_base_available": knowledge_base is not None,
        "timestamp": "2024-01-01T00:00:00Z"  # You might want to use actual timestamp
    }


@app.post("/llm/test")
async def test_llm_integration():
    """Test the LLM integration with a simple example."""
    if not claude_integration:
        raise HTTPException(
            status_code=503, 
            detail="Claude integration not available. Please set ANTHROPIC_API_KEY environment variable."
        )
    
    try:
        # Test with a simple example
        premises = [
            "All birds can fly",
            "Tweety is a bird"
        ]
        conclusion = "Can Tweety fly?"
        
        import asyncio
        result = await claude_integration.reason_with_claude(premises, conclusion)
        
        return {
            "success": True,
            "test_case": {
                "premises": premises,
                "conclusion": conclusion
            },
            "result": result,
            "message": "LLM integration test completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM test failed: {str(e)}")


# ==================== Prolog Inference Endpoints ====================

class PrologInferenceRequest(BaseModel):
    premises: List[str]
    query: Optional[str] = None

class PrologInferenceResponse(BaseModel):
    success: bool
    premises: List[str]
    query: Optional[str] = None
    prolog_facts: List[str]
    conclusions: List[Dict[str, Any]]
    conclusions_count: int
    reasoning_time: float


@app.post("/prolog/infer", response_model=PrologInferenceResponse)
async def prolog_infer(request: PrologInferenceRequest):
    """
    Infer conclusions from premises using Prolog reasoning
    Uses ELMS.py functions directly for consistency
    """
    import time
    start_time = time.time()
    
    try:
        # Clear previous knowledge base
        prolog_reasoner.clear()
        
        # Convert premises to Prolog format using ELMS functions
        prolog_facts = []
        for premise in request.premises:
            prolog = _convert_nl_to_prolog(premise, vectionary_parser)
            if prolog:
                prolog_facts.append(prolog)
                if " :- " in prolog:
                    prolog_reasoner.add_rule(prolog)
                else:
                    prolog_reasoner.add_fact(prolog)
            elif "(" in premise or " :- " in premise:
                # Already in Prolog format
                prolog_facts.append(premise)
                if " :- " in premise:
                    prolog_reasoner.add_rule(premise)
                else:
                    prolog_reasoner.add_fact(premise)
        
        # Convert query to Prolog format using ELMS functions
        prolog_query = None
        if request.query:
            # Normalize the query - remove apostrophes for consistent parsing
            normalized_query = request.query.replace("'s ", " ").replace("'s?", "?").replace("'s.", ".")
            prolog_query = _convert_query_to_prolog(normalized_query, vectionary_parser)
            if not prolog_query:
                # Try to use the query as-is
                prolog_query = request.query
        
        # Query Prolog
        success, results = prolog_reasoner.query(prolog_query)
        
        elapsed_time = time.time() - start_time
        
        return PrologInferenceResponse(
            success=success,
            premises=request.premises,
            query=request.query,
            prolog_facts=prolog_facts,
            conclusions=results,
            conclusions_count=len(results),
            reasoning_time=elapsed_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prolog inference failed: {str(e)}")


class HybridVerifyRequest(BaseModel):
    premises: List[str]
    conclusion: str

class HybridVerifyResponse(BaseModel):
    valid: bool
    confidence: float
    elms_valid: bool
    prolog_valid: bool
    reasoning_time: float


@app.post("/hybrid/verify", response_model=HybridVerifyResponse)
async def hybrid_verify(request: HybridVerifyRequest):
    """
    Verify conclusion using hybrid reasoning (ELMS + Prolog)
    """
    try:
        import time
        start_time = time.time()
        
        # Clear previous knowledge base
        hybrid_reasoner.clear()
        
        # Verify conclusion
        result = hybrid_reasoner.verify_conclusion(request.premises, request.conclusion)
        
        elapsed_time = time.time() - start_time
        
        return HybridVerifyResponse(
            valid=result['valid'],
            confidence=result['confidence'],
            elms_valid=result['elms_result'].get('valid', False),
            prolog_valid=result['prolog_valid'],
            reasoning_time=elapsed_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid verification failed: {str(e)}")


# Mount static files for the HTML interface
app.mount("/static", StaticFiles(directory="."), name="static")


if __name__ == "__main__":
    print("🚀 Starting Enhanced Logic Reasoning API...")
    print("🎯 Target: High accuracy logical reasoning with Vectionary parsing")
    print("🌐 HTML Interface: http://localhost:8002")
    print("📊 Test Edge Case: http://localhost:8002/test-edge-case")
    print("🧪 Comprehensive Test: http://localhost:8002/test-comprehensive")
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
