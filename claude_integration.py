"""
Claude API Integration for Enhanced Logic Reasoning

This module integrates Anthropic's Claude API with the ELMS logic system
to validate LLM responses against formal logic and provide reasoning validation.
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue without it
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClaudeResponse:
    """Represents a response from Claude API."""
    content: str
    confidence: float
    reasoning_steps: List[str]
    logic_formula: Optional[str] = None
    validation_result: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ClaudeIntegration:
    """
    Integration with Anthropic's Claude API for logical reasoning validation.
    
    Features:
    - Send natural language queries to Claude
    - Convert Claude responses to formal logic
    - Validate responses against logical rules
    - Generate truth tables for validation
    - Provide detailed reasoning analysis
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude integration.
        
        Args:
            api_key: Anthropic API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.model = os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022')  # Latest Claude model
        
        # Import our logic system components
        try:
            from vectionary_98_percent_solution import Vectionary98PercentSolution
            from vectionary_knowledge_base import VectionaryKnowledgeBase
            self.vectionary_engine = Vectionary98PercentSolution()
            self.knowledge_base = VectionaryKnowledgeBase()
        except ImportError as e:
            logger.warning(f"Could not import Vectionary components: {e}")
            self.vectionary_engine = None
            self.knowledge_base = None
    
    async def query_claude(self, 
                          prompt: str, 
                          system_prompt: Optional[str] = None,
                          max_tokens: int = 1000,
                          temperature: float = 0.1) -> ClaudeResponse:
        """
        Send a query to Claude API and get response.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt for context
            max_tokens: Maximum tokens in response
            temperature: Response randomness (0.0 = deterministic, 1.0 = random)
        
        Returns:
            ClaudeResponse: Structured response from Claude
        """
        if not system_prompt:
            system_prompt = self._get_default_system_prompt()
        
        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['content'][0]['text']
                        
                        # Extract reasoning steps if present
                        reasoning_steps = self._extract_reasoning_steps(content)
                        
                        return ClaudeResponse(
                            content=content,
                            confidence=0.8,  # Default confidence
                            reasoning_steps=reasoning_steps
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(f"Claude API error {response.status}: {error_text}")
        
        except Exception as e:
            logger.error(f"Error querying Claude: {e}")
            raise
    
    def validate_llm_response(self, 
                            llm_response: str, 
                            premises: List[str], 
                            conclusion: str) -> Dict[str, Any]:
        """
        Validate an LLM response against logical premises and conclusion.
        
        Args:
            llm_response: The LLM's response text
            premises: List of logical premises
            conclusion: The conclusion to validate
        
        Returns:
            Dict containing validation results, logic formulas, and analysis
        """
        logger.info(f"Validating LLM response against premises: {premises}")
        logger.info(f"Conclusion: {conclusion}")
        
        try:
            # Step 1: Convert LLM response to logic using Vectionary
            if self.vectionary_engine:
                parsed_response = self.vectionary_engine.parse_with_vectionary_98(llm_response)
                llm_logic = parsed_response.formula
                llm_confidence = parsed_response.confidence
            else:
                # Fallback parsing
                llm_logic = self._simple_parse_to_logic(llm_response)
                llm_confidence = 0.7
            
            # Step 2: Convert premises and conclusion to logic
            if self.vectionary_engine:
                parsed_premises = [self.vectionary_engine.parse_with_vectionary_98(p) for p in premises]
                parsed_conclusion = self.vectionary_engine.parse_with_vectionary_98(conclusion)
                
                premise_formulas = [p.formula for p in parsed_premises]
                conclusion_formula = parsed_conclusion.formula
            else:
                premise_formulas = [self._simple_parse_to_logic(p) for p in premises]
                conclusion_formula = self._simple_parse_to_logic(conclusion)
            
            # Step 3: Check logical consistency
            consistency_result = self._check_logical_consistency(
                premise_formulas, conclusion_formula, llm_logic
            )
            
            # Step 4: Generate truth table for validation
            truth_table = self._generate_truth_table_for_validation(
                premise_formulas, conclusion_formula, llm_logic
            )
            
            # Step 5: Determine if LLM response is valid
            is_valid = consistency_result['consistent'] and consistency_result['confidence'] > 0.7
            
            return {
                "llm_response": llm_response,
                "llm_logic": llm_logic,
                "llm_confidence": llm_confidence,
                "premise_formulas": premise_formulas,
                "conclusion_formula": conclusion_formula,
                "is_valid": is_valid,
                "consistency_analysis": consistency_result,
                "truth_table": truth_table,
                "validation_confidence": consistency_result['confidence'],
                "reasoning": self._generate_validation_reasoning(
                    is_valid, consistency_result, llm_response
                )
            }
        
        except Exception as e:
            logger.error(f"Error validating LLM response: {e}")
            return {
                "llm_response": llm_response,
                "error": str(e),
                "is_valid": False,
                "validation_confidence": 0.0
            }
    
    async def reason_with_claude(self, 
                               premises: List[str], 
                               conclusion: str,
                               use_knowledge_base: bool = True) -> Dict[str, Any]:
        """
        Use Claude to reason about premises and conclusion, then validate the response.
        
        Args:
            premises: List of logical premises
            conclusion: The conclusion to reason about
            use_knowledge_base: Whether to use the knowledge base for context
        
        Returns:
            Dict containing Claude's response and validation results
        """
        # Build context from knowledge base if available
        context = ""
        if use_knowledge_base and self.knowledge_base:
            relevant_facts = self.knowledge_base.get_all_facts()
            if relevant_facts:
                context = "Relevant facts from knowledge base:\n"
                for fact in relevant_facts[:5]:  # Limit to top 5 facts
                    context += f"- {fact['text']}\n"
                context += "\n"
        
        # Create reasoning prompt
        prompt = f"""
{context}Given the following premises and conclusion, please provide a logical analysis:

Premises:
{chr(10).join(f"{i+1}. {p}" for i, p in enumerate(premises))}

Conclusion: {conclusion}

Please:
1. Analyze whether the conclusion logically follows from the premises
2. Explain your reasoning step by step
3. Identify any logical rules or patterns you're using
4. Provide your final answer (Yes/No) with confidence level

Format your response clearly with numbered steps.
"""
        
        try:
            # Get Claude's response
            claude_response = await self.query_claude(prompt)
            
            # Validate the response
            validation_result = self.validate_llm_response(
                claude_response.content, premises, conclusion
            )
            
            return {
                "claude_response": claude_response,
                "validation_result": validation_result,
                "premises": premises,
                "conclusion": conclusion,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error in reasoning with Claude: {e}")
            return {
                "error": str(e),
                "premises": premises,
                "conclusion": conclusion,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for Claude."""
        return """You are an expert in formal logic and logical reasoning. You excel at:

1. Analyzing logical arguments and identifying valid/invalid inferences
2. Applying formal logic rules (Modus Ponens, Modus Tollens, Universal Instantiation, etc.)
3. Explaining your reasoning step by step
4. Identifying logical fallacies and inconsistencies
5. Working with propositional logic, first-order logic, and temporal logic

When analyzing arguments:
- Break down complex statements into logical components
- Identify the logical structure and relationships
- Apply appropriate logical rules
- Provide clear, step-by-step reasoning
- State your conclusion with confidence level

Always be precise, logical, and thorough in your analysis."""
    
    def _extract_reasoning_steps(self, content: str) -> List[str]:
        """Extract reasoning steps from Claude's response."""
        steps = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered steps or bullet points
            if (line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or
                line.startswith(('•', '-', '*'))):
                steps.append(line)
        
        return steps
    
    def _simple_parse_to_logic(self, text: str) -> str:
        """Simple fallback parsing to logic when Vectionary is not available."""
        text_lower = text.lower().strip()
        
        # Handle common patterns
        if 'all' in text_lower and 'are' in text_lower:
            return f"∀x({text_lower.replace(' ', '_').replace('.', '')})"
        elif 'some' in text_lower:
            return f"∃x({text_lower.replace(' ', '_').replace('.', '')})"
        elif 'if' in text_lower and 'then' in text_lower:
            parts = text_lower.split(' then ')
            if len(parts) == 2:
                antecedent = parts[0].replace('if ', '').replace(' ', '_')
                consequent = parts[1].replace(' ', '_')
                return f"({antecedent} → {consequent})"
        else:
            # Simple atomic formula
            return text_lower.replace(' ', '_').replace('.', '').replace('?', '')
    
    def _check_logical_consistency(self, 
                                 premise_formulas: List[str], 
                                 conclusion_formula: str, 
                                 llm_logic: str) -> Dict[str, Any]:
        """Check if LLM logic is consistent with premises and conclusion."""
        try:
            # Use Vectionary engine for consistency checking if available
            if self.vectionary_engine:
                # Convert premise formulas back to natural language for Vectionary
                premise_texts = [self._logic_to_natural_language(f) for f in premise_formulas]
                conclusion_text = self._logic_to_natural_language(conclusion_formula)
                
                # Use Vectionary to check consistency
                result = self.vectionary_engine.prove_theorem_98(
                    premise_texts + [self._logic_to_natural_language(llm_logic)], 
                    conclusion_text
                )
                
                return {
                    "consistent": result.get('valid', False),
                    "confidence": result.get('confidence', 0.5),
                    "explanation": result.get('explanation', 'No explanation available'),
                    "reasoning_steps": result.get('reasoning_steps', [])
                }
            else:
                # Simple consistency check
                return self._simple_consistency_check(premise_formulas, conclusion_formula, llm_logic)
        
        except Exception as e:
            logger.error(f"Error in consistency check: {e}")
            return {
                "consistent": False,
                "confidence": 0.0,
                "explanation": f"Error in consistency check: {str(e)}",
                "reasoning_steps": []
            }
    
    def _simple_consistency_check(self, 
                                premise_formulas: List[str], 
                                conclusion_formula: str, 
                                llm_logic: str) -> Dict[str, Any]:
        """Simple consistency check when Vectionary is not available."""
        # Basic pattern matching for consistency
        consistent = True
        confidence = 0.7
        
        # Check if LLM logic contains elements from premises
        premise_elements = set()
        for formula in premise_formulas:
            elements = formula.replace('(', '').replace(')', '').replace('∀x', '').replace('∃x', '').split()
            premise_elements.update(elements)
        
        llm_elements = set(llm_logic.replace('(', '').replace(')', '').replace('∀x', '').replace('∃x', '').split())
        
        # If LLM logic shares elements with premises, it's more likely consistent
        if premise_elements.intersection(llm_elements):
            confidence = 0.8
        else:
            confidence = 0.5
            consistent = False
        
        return {
            "consistent": consistent,
            "confidence": confidence,
            "explanation": f"Simple consistency check: {'consistent' if consistent else 'inconsistent'} with premises",
            "reasoning_steps": [
                f"Premise elements: {list(premise_elements)}",
                f"LLM logic elements: {list(llm_elements)}",
                f"Shared elements: {list(premise_elements.intersection(llm_elements))}"
            ]
        }
    
    def _generate_truth_table_for_validation(self, 
                                           premise_formulas: List[str], 
                                           conclusion_formula: str, 
                                           llm_logic: str) -> Dict[str, Any]:
        """Generate truth table for validation purposes."""
        try:
            # Extract variables from all formulas
            all_variables = set()
            for formula in premise_formulas + [conclusion_formula, llm_logic]:
                variables = self._extract_variables_from_formula(formula)
                all_variables.update(variables)
            
            if len(all_variables) > 3:  # Limit complexity
                return {"error": "Too many variables for truth table generation"}
            
            # Generate truth table
            variables = list(all_variables)
            combinations = self._generate_truth_combinations(len(variables))
            
            truth_table = {
                "variables": variables,
                "combinations": [],
                "premise_results": [],
                "conclusion_results": [],
                "llm_results": []
            }
            
            for combination in combinations:
                row = {}
                for i, var in enumerate(variables):
                    row[var] = combination[i]
                
                # Evaluate each formula with this combination
                premise_vals = []
                for formula in premise_formulas:
                    val = self._evaluate_formula_with_values(formula, variables, combination)
                    premise_vals.append(val)
                
                conclusion_val = self._evaluate_formula_with_values(conclusion_formula, variables, combination)
                llm_val = self._evaluate_formula_with_values(llm_logic, variables, combination)
                
                truth_table["combinations"].append(row)
                truth_table["premise_results"].append(premise_vals)
                truth_table["conclusion_results"].append(conclusion_val)
                truth_table["llm_results"].append(llm_val)
            
            return truth_table
        
        except Exception as e:
            logger.error(f"Error generating truth table: {e}")
            return {"error": str(e)}
    
    def _extract_variables_from_formula(self, formula: str) -> List[str]:
        """Extract variables from a logical formula."""
        import re
        # Simple variable extraction
        variables = re.findall(r'\b[a-z]+\b', formula.lower())
        # Filter out common logical operators and keywords
        filtered = [v for v in variables if v not in ['and', 'or', 'not', 'if', 'then', 'all', 'some', 'exists', 'for']]
        return list(set(filtered))  # Remove duplicates
    
    def _generate_truth_combinations(self, num_variables: int) -> List[List[bool]]:
        """Generate all possible truth value combinations."""
        if num_variables == 0:
            return [[]]
        elif num_variables == 1:
            return [[False], [True]]
        elif num_variables == 2:
            return [[False, False], [False, True], [True, False], [True, True]]
        elif num_variables == 3:
            return [
                [False, False, False], [False, False, True], [False, True, False], [False, True, True],
                [True, False, False], [True, False, True], [True, True, False], [True, True, True]
            ]
        else:
            # For more variables, return a subset
            return [[False, False, False], [True, True, True]]
    
    def _evaluate_formula_with_values(self, formula: str, variables: List[str], values: List[bool]) -> bool:
        """Evaluate a formula with given variable values."""
        # Simple evaluation - in a real system, this would be more sophisticated
        if '∀x' in formula or '∃x' in formula:
            return True  # Assume quantified formulas are true
        elif any(var in formula.lower() for var in variables):
            # If formula contains variables, return True if any variable is True
            return any(values)
        else:
            return True  # Default to True for atomic formulas
    
    def _logic_to_natural_language(self, logic_formula: str) -> str:
        """Convert logic formula back to natural language for Vectionary."""
        # Simple conversion - in practice, this would be more sophisticated
        if logic_formula.startswith('∀x('):
            content = logic_formula[3:-1]  # Remove ∀x( and )
            return f"All {content.replace('_', ' ')}"
        elif logic_formula.startswith('∃x('):
            content = logic_formula[3:-1]  # Remove ∃x( and )
            return f"Some {content.replace('_', ' ')}"
        elif '→' in logic_formula:
            parts = logic_formula.split(' → ')
            antecedent = parts[0].replace('_', ' ').replace('(', '').replace(')', '')
            consequent = parts[1].replace('_', ' ').replace('(', '').replace(')', '')
            return f"If {antecedent} then {consequent}"
        else:
            return logic_formula.replace('_', ' ').replace('(', '').replace(')', '')
    
    def _generate_validation_reasoning(self, 
                                     is_valid: bool, 
                                     consistency_result: Dict[str, Any], 
                                     llm_response: str) -> str:
        """Generate human-readable validation reasoning."""
        if is_valid:
            return f"✅ LLM response is VALID. {consistency_result['explanation']}"
        else:
            return f"❌ LLM response is INVALID. {consistency_result['explanation']}"


# Example usage and testing
async def test_claude_integration():
    """Test the Claude integration functionality."""
    try:
        # Initialize Claude integration
        claude = ClaudeIntegration()
        
        # Test premises and conclusion
        premises = [
            "All humans are mortal",
            "Socrates is human"
        ]
        conclusion = "Socrates is mortal"
        
        # Test reasoning with Claude
        result = await claude.reason_with_claude(premises, conclusion)
        
        print("Claude Integration Test Results:")
        print(json.dumps(result, indent=2, default=str))
        
        return result
    
    except Exception as e:
        print(f"Test failed: {e}")
        return None


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_claude_integration())
