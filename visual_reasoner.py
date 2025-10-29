"""
Visual Reasoning Component for ELMS
Integrates DeepSeek-OCR with existing logical reasoning capabilities
"""

import os
import json
import tempfile
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import logging

try:
    from ocr_processor import OCRProcessor
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from simple_ocr_processor import SimpleOCRProcessor
    SIMPLE_OCR_AVAILABLE = True
except ImportError:
    SIMPLE_OCR_AVAILABLE = False
# Dynamic converter will be imported as needed
from vectionary_knowledge_base import VectionaryKnowledgeBase
from ELMS import VectionaryParser, VectionaryAPIClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualReasoner:
    """
    Visual reasoning component that combines OCR with logical reasoning.
    
    Features:
    - Extract text and logical content from images/documents
    - Integrate with existing ELMS reasoning pipeline
    - Support for visual question answering
    - Document analysis and reasoning
    - Multi-modal reasoning (text + visual)
    """
    
    def __init__(self, 
                 ocr_model: str = 'deepseek-ai/DeepSeek-OCR',
                 environment: str = 'prod'):
        """
        Initialize the visual reasoner.
        
        Args:
            ocr_model: OCR model name
            environment: Vectionary API environment
        """
        # Store OCR model for lazy initialization
        self.ocr_model = ocr_model
        self.ocr_processor = None
        self._ocr_initialized = False
        
        # Initialize reasoning components
        api_client = VectionaryAPIClient(environment=environment)
        self.parser = VectionaryParser(api_client)
    
    def _initialize_ocr(self):
        """Lazy initialization of OCR processor"""
        if self._ocr_initialized:
            return
        
        try:
            # Initialize OCR processor (try DeepSeek-OCR first, fallback to simple OCR)
            if OCR_AVAILABLE:
                try:
                    self.ocr_processor = OCRProcessor(model_name=self.ocr_model)
                    if not self.ocr_processor.initialized:
                        raise Exception("DeepSeek-OCR not available")
                except Exception as e:
                    logger.warning(f"DeepSeek-OCR failed: {e}")
                    if SIMPLE_OCR_AVAILABLE:
                        self.ocr_processor = SimpleOCRProcessor(model_name=self.ocr_model)
                    else:
                        raise Exception("No OCR processor available")
            elif SIMPLE_OCR_AVAILABLE:
                self.ocr_processor = SimpleOCRProcessor(model_name=self.ocr_model)
            else:
                raise Exception("No OCR processor available")
            
            self._ocr_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            raise
        # Knowledge base removed - using hybrid reasoner only
        
        logger.info("Visual reasoner initialized successfully")
    
    def analyze_document(self, 
                        document_path: str,
                        question: str = None,
                        extract_logical_content: bool = True,
                        add_to_knowledge_base: bool = True) -> Dict[str, Any]:
        """
        Analyze a document and optionally answer questions about it.
        
        Args:
            document_path: Path to document (image, PDF, etc.)
            question: Optional question about the document
            extract_logical_content: Whether to extract logical statements
            add_to_knowledge_base: Whether to add extracted content to KB
        
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing document: {document_path}")
        
        try:
            # Initialize OCR if needed
            self._initialize_ocr()
            
            # Process document with OCR
            ocr_result = self.ocr_processor.process_document(document_path)
            
            if not ocr_result['success']:
                return {
                    'success': False,
                    'error': f"OCR processing failed: {ocr_result.get('error', 'Unknown error')}",
                    'document_path': document_path
                }
            
            # Extract logical content if requested
            logical_content = []
            if extract_logical_content and ocr_result.get('logical_content'):
                logical_content = ocr_result['logical_content']
            
            # Add to knowledge base if requested
            if add_to_knowledge_base and logical_content:
                for statement in logical_content:
                    self.knowledge_base.add_fact(
                        text=statement['text'],
                        confidence=statement.get('confidence', 0.8),
                        source='ocr',
                        tags=['visual', 'document']
                    )
            
            # Answer question if provided
            answer_result = None
            if question:
                answer_result = self.answer_visual_question(
                    document_path, question, use_knowledge_base=add_to_knowledge_base
                )
            
            return {
                'success': True,
                'document_path': document_path,
                'ocr_result': ocr_result,
                'logical_content': logical_content,
                'question_answer': answer_result,
                'knowledge_base_updated': add_to_knowledge_base and logical_content
            }
            
        except Exception as e:
            logger.error(f"Error analyzing document {document_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'document_path': document_path
            }
    
    def answer_visual_question(self, 
                              document_path: str,
                              question: str,
                              use_knowledge_base: bool = True) -> Dict[str, Any]:
        """
        Answer a question about a visual document.
        
        Args:
            document_path: Path to document
            question: Question about the document
            use_knowledge_base: Whether to use knowledge base for reasoning
        
        Returns:
            Dictionary with answer and reasoning
        """
        logger.info(f"Answering visual question: {question}")
        
        try:
            # First, analyze the document WITHOUT adding to knowledge base
            analysis_result = self.analyze_document(
                document_path, 
                extract_logical_content=True,
                add_to_knowledge_base=False  # Don't use knowledge base - use hybrid reasoner only
            )
            
            if not analysis_result['success']:
                return {
                    'success': False,
                    'error': f"Document analysis failed: {analysis_result.get('error')}",
                    'question': question
                }
            
            # Extract premises from the document
            premises = []
            if analysis_result.get('logical_content'):
                premises = [stmt['text'] for stmt in analysis_result['logical_content']]
            
            # Only use OCR text if no logical content was extracted
            if not premises:
                ocr_text = analysis_result['ocr_result'].get('extracted_text', '')
                if ocr_text:
                    # Split OCR text into sentences for additional premises
                    sentences = self._split_into_sentences(ocr_text)
                    premises.extend(sentences[:10])  # Limit to avoid overwhelming
            
            if not premises:
                return {
                    'success': False,
                    'error': "No logical content extracted from document",
                    'question': question
                }
            
            # Use dynamic converter to answer the question
            from prolog_reasoner import PrologReasoner
            
            # Convert premises to Prolog
            from ELMS import _convert_nl_to_prolog, _convert_query_to_prolog
            prolog_facts = []
            for premise in premises:
                prolog = _convert_nl_to_prolog(premise, self.parser)
                if prolog:
                    prolog_facts.append(prolog)
            
            # Convert question to Prolog query
            prolog_query = _convert_query_to_prolog(question, self.parser)
            if not prolog_query:
                return {
                    'success': False,
                    'error': 'Could not convert question to Prolog query'
                }
            
            # Dynamic fix: If query is "who(X)", extract predicate from question's theme role
            # This handles cases where Vectionary assigns "who" as agent instead of using theme
            if prolog_query == "who(X)" or prolog_query.startswith("who("):
                # Re-parse the question to extract the theme (predicate)
                try:
                    parsed = self.parser.parse(question)
                    if parsed and parsed.tree:
                        # Look for theme role in the question
                        theme_value = None
                        if 'children' in parsed.tree:
                            for child in parsed.tree['children']:
                                if child.get('role') == 'theme':
                                    theme_value = child.get('text', '').lower()
                                    break
                        
                        # If theme found, use it as predicate
                        if theme_value:
                            prolog_query = f"{theme_value}(X)"
                        else:
                            # Fallback: find the most common predicate in facts
                            from collections import Counter
                            predicates = [f.split("(")[0] for f in prolog_facts if "(" in f and " :- " not in f]
                            if predicates:
                                most_common = Counter(predicates).most_common(1)[0][0]
                                prolog_query = f"{most_common}(X)"
                except Exception as e:
                    # If parsing fails, use fallback approach
                    from collections import Counter
                    predicates = [f.split("(")[0] for f in prolog_facts if "(" in f and " :- " not in f]
                    if predicates:
                        most_common = Counter(predicates).most_common(1)[0][0]
                        prolog_query = f"{most_common}(X)"
            
            # Query Prolog
            reasoner = PrologReasoner()
            for fact in prolog_facts:
                if " :- " in fact:
                    reasoner.add_rule(fact)
                else:
                    reasoner.add_fact(fact)
            
            start_time = time.time()
            success, results = reasoner.query(prolog_query)
            reasoning_time = time.time() - start_time
            
            if not success:
                return {
                    'success': False,
                    'error': f'Query failed: {prolog_query}'
                }
            
            conclusions = []
            for result in results:
                if isinstance(result, dict):
                    conclusions.extend(result.values())
                else:
                    conclusions.append(result)
            
            reasoning_result = {
                'conclusions': conclusions,
                'reasoning_time': reasoning_time
            }
            
            return {
                'success': True,
                'question': question,
                'answer': reasoning_result.get('conclusions', []),
                'reasoning_steps': reasoning_result.get('reasoning_time', 0),
                'premises_used': premises[:5],  # Show first 5 premises
                'confidence': reasoning_result.get('conclusions_count', 0) > 0
            }
            
        except Exception as e:
            logger.error(f"Error answering visual question: {e}")
            return {
                'success': False,
                'error': str(e),
                'question': question
            }
    
    def compare_documents(self, 
                         document_paths: List[str],
                         comparison_question: str = None) -> Dict[str, Any]:
        """
        Compare multiple documents and answer questions about their relationships.
        
        Args:
            document_paths: List of document paths to compare
            comparison_question: Optional question about the comparison
        
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {len(document_paths)} documents")
        
        try:
            all_analysis_results = []
            all_logical_content = []
            
            # Analyze each document
            for doc_path in document_paths:
                analysis_result = self.analyze_document(
                    doc_path,
                    extract_logical_content=True,
                    add_to_knowledge_base=True
                )
                all_analysis_results.append(analysis_result)
                
                if analysis_result.get('logical_content'):
                    all_logical_content.extend(analysis_result['logical_content'])
            
            # Find relationships between documents
            relationships = self._find_document_relationships(all_logical_content)
            
            # Answer comparison question if provided
            comparison_answer = None
            if comparison_question:
                # Combine all premises from all documents
                all_premises = []
                for analysis in all_analysis_results:
                    if analysis.get('logical_content'):
                        all_premises.extend([stmt['text'] for stmt in analysis['logical_content']])
                
                # Use dynamic converter for comparison
                from prolog_reasoner import PrologReasoner
                
                from ELMS import _convert_nl_to_prolog, _convert_query_to_prolog
                prolog_facts = []
                for premise in all_premises:
                    prolog = _convert_nl_to_prolog(premise, self.parser)
                    if prolog:
                        prolog_facts.append(prolog)
                
                prolog_query = _convert_query_to_prolog(comparison_question, self.parser) if comparison_question else None
                
                reasoner = PrologReasoner()
                for fact in prolog_facts:
                    if " :- " in fact:
                        reasoner.add_rule(fact)
                    else:
                        reasoner.add_fact(fact)
                
                start_time = time.time()
                if prolog_query:
                    success, results = reasoner.query(prolog_query)
                    if not success:
                        results = []
                else:
                    success, results = True, []
                reasoning_time = time.time() - start_time
                
                conclusions = []
                for result in results:
                    if isinstance(result, dict):
                        conclusions.extend(result.values())
                    else:
                        conclusions.append(result)
                
                reasoning_result = {
                    'conclusions': conclusions,
                    'reasoning_time': reasoning_time
                }
                
                comparison_answer = {
                    'answer': reasoning_result.get('conclusions', []),
                    'reasoning_steps': reasoning_result.get('reasoning_time', 0),
                    'confidence': reasoning_result.get('conclusions_count', 0) > 0
                }
            
            return {
                'success': True,
                'document_paths': document_paths,
                'analysis_results': all_analysis_results,
                'relationships': relationships,
                'comparison_answer': comparison_answer,
                'total_logical_statements': len(all_logical_content)
            }
            
        except Exception as e:
            logger.error(f"Error comparing documents: {e}")
            return {
                'success': False,
                'error': str(e),
                'document_paths': document_paths
            }
    
    def extract_visual_facts(self, document_path: str) -> List[Dict[str, Any]]:
        """
        Extract factual statements from a visual document.
        
        Args:
            document_path: Path to document
        
        Returns:
            List of extracted facts with metadata
        """
        logger.info(f"Extracting visual facts from: {document_path}")
        
        try:
            # Analyze document
            analysis_result = self.analyze_document(
                document_path,
                extract_logical_content=True,
                add_to_knowledge_base=False
            )
            
            if not analysis_result['success']:
                return []
            
            # Extract and format facts
            facts = []
            logical_content = analysis_result.get('logical_content', [])
            
            for i, statement in enumerate(logical_content):
                fact = {
                    'id': f'visual_fact_{i}',
                    'text': statement['text'],
                    'statement_type': statement.get('statement_type', 'factual'),
                    'logical_elements': statement.get('logical_elements', {}),
                    'confidence': statement.get('confidence', 0.8),
                    'source': 'visual_ocr',
                    'document_path': document_path,
                    'extraction_method': 'deepseek_ocr'
                }
                facts.append(fact)
            
            logger.info(f"Extracted {len(facts)} visual facts")
            return facts
            
        except Exception as e:
            logger.error(f"Error extracting visual facts: {e}")
            return []
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_document_relationships(self, logical_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find relationships between logical content from different documents.
        
        Args:
            logical_content: List of logical statements from all documents
        
        Returns:
            List of identified relationships
        """
        relationships = []
        
        # Group by predicates to find common themes
        predicate_groups = {}
        for statement in logical_content:
            predicates = statement.get('logical_elements', {}).get('predicates', [])
            for predicate in predicates:
                if predicate not in predicate_groups:
                    predicate_groups[predicate] = []
                predicate_groups[predicate].append(statement)
        
        # Find relationships based on common predicates
        for predicate, statements in predicate_groups.items():
            if len(statements) > 1:
                relationships.append({
                    'type': 'common_predicate',
                    'predicate': predicate,
                    'statements': [stmt['text'] for stmt in statements],
                    'count': len(statements)
                })
        
        # Find relationships based on common constants
        constant_groups = {}
        for statement in logical_content:
            constants = statement.get('logical_elements', {}).get('constants', [])
            for constant in constants:
                if constant not in constant_groups:
                    constant_groups[constant] = []
                constant_groups[constant].append(statement)
        
        for constant, statements in constant_groups.items():
            if len(statements) > 1:
                relationships.append({
                    'type': 'common_constant',
                    'constant': constant,
                    'statements': [stmt['text'] for stmt in statements],
                    'count': len(statements)
                })
        
        return relationships
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the visual knowledge base."""
        all_facts = self.knowledge_base.get_all_facts()
        
        # Filter visual facts
        visual_facts = [fact for fact in all_facts if 'visual' in fact.get('tags', [])]
        
        return {
            'total_facts': len(all_facts),
            'visual_facts': len(visual_facts),
            'facts_by_source': {
                'user': len([f for f in all_facts if f.get('source') == 'user']),
                'ocr': len([f for f in all_facts if f.get('source') == 'ocr']),
                'inferred': len([f for f in all_facts if f.get('source') == 'inferred'])
            },
            'recent_visual_facts': visual_facts[-5:] if visual_facts else []
        }


def test_visual_reasoner():
    """Test the visual reasoner with sample documents."""
    reasoner = VisualReasoner()
    
    print("🔍 Visual Reasoner initialized successfully!")
    # Initialize OCR for testing
    try:
        reasoner._initialize_ocr()
        print(f"OCR Available: {reasoner.ocr_processor.initialized}")
    except Exception as e:
        print(f"OCR Not Available: {e}")
    print(f"Knowledge Base Facts: {len(reasoner.knowledge_base.get_all_facts())}")
    
    # Example usage:
    # result = reasoner.analyze_document("sample_document.jpg")
    # print(f"Analysis result: {result['success']}")
    
    # question_result = reasoner.answer_visual_question(
    #     "sample_document.jpg", 
    #     "What is the main topic of this document?"
    # )
    # print(f"Answer: {question_result.get('answer')}")


if __name__ == "__main__":
    test_visual_reasoner()
