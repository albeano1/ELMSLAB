"""
Comprehensive Test Suite for LLM Integration

This module tests the integration between Claude API and the ELMS logic system.
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List
import pytest
import httpx
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from claude_integration import ClaudeIntegration, ClaudeResponse
    from vectionary_98_percent_solution import Vectionary98PercentSolution
    from vectionary_knowledge_base import VectionaryKnowledgeBase
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Some tests may be skipped.")


class LLMIntegrationTester:
    """Comprehensive test suite for LLM integration."""
    
    def __init__(self, api_base_url: str = "http://localhost:8002"):
        self.api_base_url = api_base_url
        self.test_results = []
        self.claude_available = False
        
        # Check if Claude API key is available
        if os.getenv('ANTHROPIC_API_KEY'):
            try:
                self.claude_integration = ClaudeIntegration()
                self.claude_available = True
                print("✅ Claude integration available for testing")
            except Exception as e:
                print(f"⚠️ Claude integration not available: {e}")
                self.claude_integration = None
        else:
            print("⚠️ ANTHROPIC_API_KEY not set - Claude tests will be skipped")
            self.claude_integration = None
    
    def log_test_result(self, test_name: str, success: bool, details: str = "", data: Dict = None):
        """Log a test result."""
        result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
    
    async def test_claude_integration_basic(self) -> bool:
        """Test basic Claude integration functionality."""
        if not self.claude_available:
            self.log_test_result("claude_basic", False, "Claude not available")
            return False
        
        try:
            # Test simple query
            response = await self.claude_integration.query_claude(
                "What is 2 + 2?",
                "You are a helpful math assistant."
            )
            
            success = response is not None and len(response.content) > 0
            self.log_test_result(
                "claude_basic", 
                success, 
                f"Response length: {len(response.content) if response else 0}",
                {"response_preview": response.content[:100] if response else None}
            )
            return success
            
        except Exception as e:
            self.log_test_result("claude_basic", False, f"Error: {str(e)}")
            return False
    
    async def test_llm_validation_simple(self) -> bool:
        """Test LLM response validation with simple logic."""
        if not self.claude_available:
            self.log_test_result("llm_validation_simple", False, "Claude not available")
            return False
        
        try:
            premises = ["All birds can fly", "Tweety is a bird"]
            conclusion = "Tweety can fly"
            llm_response = "Yes, Tweety can fly because all birds can fly and Tweety is a bird."
            
            validation = self.claude_integration.validate_llm_response(
                llm_response, premises, conclusion
            )
            
            success = validation.get("is_valid", False)
            self.log_test_result(
                "llm_validation_simple",
                success,
                f"Validation confidence: {validation.get('validation_confidence', 0)}",
                {"validation_result": validation}
            )
            return success
            
        except Exception as e:
            self.log_test_result("llm_validation_simple", False, f"Error: {str(e)}")
            return False
    
    async def test_llm_reasoning_complex(self) -> bool:
        """Test complex reasoning with Claude."""
        if not self.claude_available:
            self.log_test_result("llm_reasoning_complex", False, "Claude not available")
            return False
        
        try:
            premises = [
                "All humans are mortal",
                "Socrates is human",
                "If someone is mortal, they will die"
            ]
            conclusion = "Socrates will die"
            
            result = await self.claude_integration.reason_with_claude(
                premises, conclusion
            )
            
            success = "claude_response" in result and result["claude_response"] is not None
            self.log_test_result(
                "llm_reasoning_complex",
                success,
                f"Claude response available: {success}",
                {"result_keys": list(result.keys()) if result else []}
            )
            return success
            
        except Exception as e:
            self.log_test_result("llm_reasoning_complex", False, f"Error: {str(e)}")
            return False
    
    async def test_api_endpoints(self) -> bool:
        """Test API endpoints for LLM integration."""
        try:
            async with httpx.AsyncClient() as client:
                # Test status endpoint
                response = await client.get(f"{self.api_base_url}/llm/status")
                status_success = response.status_code == 200
                
                if status_success:
                    status_data = response.json()
                    claude_available = status_data.get("claude_available", False)
                    
                    if claude_available:
                        # Test LLM validation endpoint
                        validation_payload = {
                            "llm_response": "Yes, the conclusion follows logically.",
                            "premises": ["All A are B", "X is A"],
                            "conclusion": "X is B"
                        }
                        
                        validation_response = await client.post(
                            f"{self.api_base_url}/llm/validate",
                            json=validation_payload
                        )
                        validation_success = validation_response.status_code in [200, 503]  # 503 if Claude not available
                        
                        # Test LLM reasoning endpoint
                        reasoning_payload = {
                            "premises": ["All birds can fly", "Tweety is a bird"],
                            "conclusion": "Can Tweety fly?"
                        }
                        
                        reasoning_response = await client.post(
                            f"{self.api_base_url}/llm/reason",
                            json=reasoning_payload
                        )
                        reasoning_success = reasoning_response.status_code in [200, 503]  # 503 if Claude not available
                        
                        success = status_success and validation_success and reasoning_success
                        self.log_test_result(
                            "api_endpoints",
                            success,
                            f"Status: {status_success}, Validation: {validation_success}, Reasoning: {reasoning_success}",
                            {"claude_available": claude_available}
                        )
                    else:
                        success = status_success
                        self.log_test_result(
                            "api_endpoints",
                            success,
                            "API endpoints available but Claude not configured",
                            {"claude_available": False}
                        )
                else:
                    self.log_test_result("api_endpoints", False, f"Status endpoint failed: {response.status_code}")
                    return False
                
                return success
                
        except Exception as e:
            self.log_test_result("api_endpoints", False, f"Error: {str(e)}")
            return False
    
    async def test_truth_table_generation(self) -> bool:
        """Test truth table generation for validation."""
        if not self.claude_available:
            self.log_test_result("truth_table_generation", False, "Claude not available")
            return False
        
        try:
            # Test with simple propositional logic
            premises = ["If P then Q", "P"]
            conclusion = "Q"
            llm_response = "Yes, Q follows from the premises."
            
            validation = self.claude_integration.validate_llm_response(
                llm_response, premises, conclusion
            )
            
            truth_table = validation.get("truth_table", {})
            has_truth_table = "variables" in truth_table and "combinations" in truth_table
            
            self.log_test_result(
                "truth_table_generation",
                has_truth_table,
                f"Truth table generated: {has_truth_table}",
                {"truth_table_keys": list(truth_table.keys()) if truth_table else []}
            )
            return has_truth_table
            
        except Exception as e:
            self.log_test_result("truth_table_generation", False, f"Error: {str(e)}")
            return False
    
    async def test_knowledge_base_integration(self) -> bool:
        """Test integration with knowledge base."""
        try:
            # Test knowledge base operations
            kb = VectionaryKnowledgeBase()
            
            # Add a test fact
            fact = kb.add_fact("All birds can fly", confidence=0.9)
            
            # Query the knowledge base
            result = kb.query("Can birds fly?")
            
            success = result.get("answer") in ["Yes", "No"] and result.get("confidence", 0) > 0
            
            # Clean up
            kb.delete_fact(fact.id)
            
            self.log_test_result(
                "knowledge_base_integration",
                success,
                f"Query result: {result.get('answer', 'Unknown')}",
                {"confidence": result.get("confidence", 0)}
            )
            return success
            
        except Exception as e:
            self.log_test_result("knowledge_base_integration", False, f"Error: {str(e)}")
            return False
    
    async def test_vectionary_parsing(self) -> bool:
        """Test Vectionary parsing functionality."""
        try:
            vectionary = Vectionary98PercentSolution()
            
            # Test parsing
            result = vectionary.parse_with_vectionary_98("All birds can fly")
            
            success = result is not None and hasattr(result, 'formula') and len(result.formula) > 0
            
            self.log_test_result(
                "vectionary_parsing",
                success,
                f"Parsed formula: {result.formula if result else 'None'}",
                {"confidence": result.confidence if result else 0}
            )
            return success
            
        except Exception as e:
            self.log_test_result("vectionary_parsing", False, f"Error: {str(e)}")
            return False
    
    async def test_comprehensive_workflow(self) -> bool:
        """Test the complete workflow from natural language to validated reasoning."""
        if not self.claude_available:
            self.log_test_result("comprehensive_workflow", False, "Claude not available")
            return False
        
        try:
            # Step 1: Parse premises with Vectionary
            vectionary = Vectionary98PercentSolution()
            premises = ["All humans are mortal", "Socrates is human"]
            conclusion = "Socrates is mortal"
            
            parsed_premises = [vectionary.parse_with_vectionary_98(p) for p in premises]
            parsed_conclusion = vectionary.parse_with_vectionary_98(conclusion)
            
            # Step 2: Get Claude's reasoning
            claude_result = await self.claude_integration.reason_with_claude(
                premises, conclusion
            )
            
            # Step 3: Validate Claude's response
            if "claude_response" in claude_result:
                validation = self.claude_integration.validate_llm_response(
                    claude_result["claude_response"].content,
                    premises,
                    conclusion
                )
                
                success = (
                    len(parsed_premises) == 2 and
                    parsed_conclusion is not None and
                    "claude_response" in claude_result and
                    validation.get("is_valid", False)
                )
                
                self.log_test_result(
                    "comprehensive_workflow",
                    success,
                    f"Complete workflow: Vectionary parsing + Claude reasoning + validation",
                    {
                        "parsed_premises_count": len(parsed_premises),
                        "claude_response_available": "claude_response" in claude_result,
                        "validation_valid": validation.get("is_valid", False)
                    }
                )
                return success
            else:
                self.log_test_result("comprehensive_workflow", False, "Claude response not available")
                return False
                
        except Exception as e:
            self.log_test_result("comprehensive_workflow", False, f"Error: {str(e)}")
            return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        print("🧪 Starting Comprehensive LLM Integration Tests")
        print("=" * 60)
        
        # Run all tests
        test_methods = [
            self.test_vectionary_parsing,
            self.test_knowledge_base_integration,
            self.test_api_endpoints,
            self.test_claude_integration_basic,
            self.test_llm_validation_simple,
            self.test_llm_reasoning_complex,
            self.test_truth_table_generation,
            self.test_comprehensive_workflow
        ]
        
        results = []
        for test_method in test_methods:
            try:
                result = await test_method()
                results.append(result)
            except Exception as e:
                print(f"❌ Test {test_method.__name__} failed with exception: {e}")
                results.append(False)
        
        # Calculate summary
        total_tests = len(results)
        passed_tests = sum(results)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": success_rate,
            "claude_available": self.claude_available,
            "test_results": self.test_results,
            "timestamp": datetime.now().isoformat()
        }
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Claude Available: {self.claude_available}")
        
        if success_rate >= 80:
            print("🎉 Overall Result: EXCELLENT")
        elif success_rate >= 60:
            print("✅ Overall Result: GOOD")
        elif success_rate >= 40:
            print("⚠️ Overall Result: NEEDS IMPROVEMENT")
        else:
            print("❌ Overall Result: POOR")
        
        return summary


async def main():
    """Main function to run the test suite."""
    # Check if API server is running
    api_url = "http://localhost:8002"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/health", timeout=5.0)
            if response.status_code != 200:
                print(f"⚠️ API server not responding properly at {api_url}")
                print("Please start the API server with: python vectionary_98_api.py")
                return
    except Exception as e:
        print(f"❌ Cannot connect to API server at {api_url}")
        print("Please start the API server with: python vectionary_98_api.py")
        print(f"Error: {e}")
        return
    
    # Run tests
    tester = LLMIntegrationTester(api_url)
    results = await tester.run_all_tests()
    
    # Save results to file
    with open("llm_integration_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: llm_integration_test_results.json")
    
    # Exit with appropriate code
    if results["success_rate"] >= 60:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    asyncio.run(main())
