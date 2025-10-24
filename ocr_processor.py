"""
DeepSeek-OCR Integration for ELMS Visual Reasoning
Processes images and documents to extract text and logical content for reasoning
"""

import os
import torch
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import AutoModel, AutoTokenizer
    DEEPSEEK_OCR_AVAILABLE = True
    # Try to import flash_attn, but don't require it
    try:
        import flash_attn
        FLASH_ATTN_AVAILABLE = True
    except ImportError:
        FLASH_ATTN_AVAILABLE = False
        logger.info("flash-attn not available, using standard attention")
except ImportError:
    DEEPSEEK_OCR_AVAILABLE = False
    FLASH_ATTN_AVAILABLE = False
    logger.warning("DeepSeek-OCR dependencies not available. Install with: pip install torch transformers")


class OCRProcessor:
    """
    DeepSeek-OCR processor for extracting text and logical content from images/documents.
    
    Features:
    - Document OCR with layout preservation
    - Figure and table extraction
    - Logical content parsing
    - Multi-resolution processing
    - Integration with ELMS reasoning pipeline
    """
    
    def __init__(self, model_name: str = 'deepseek-ai/DeepSeek-OCR', device: str = 'auto'):
        """
        Initialize the OCR processor.
        
        Args:
            model_name: Hugging Face model name for DeepSeek-OCR
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.initialized = False
        
        if DEEPSEEK_OCR_AVAILABLE:
            self._initialize_model()
        else:
            logger.error("DeepSeek-OCR not available. Please install dependencies.")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device
    
    def _initialize_model(self):
        """Initialize the DeepSeek-OCR model and tokenizer."""
        try:
            logger.info(f"Loading DeepSeek-OCR model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Load model with optimizations
            model_kwargs = {
                'trust_remote_code': True,
                'use_safetensors': True,
                'torch_dtype': torch.bfloat16 if self.device == 'cuda' else torch.float32
            }
            
            # Use flash attention if available
            if FLASH_ATTN_AVAILABLE:
                model_kwargs['_attn_implementation'] = 'flash_attention_2'
                logger.info("Using flash attention for better performance")
            else:
                logger.info("Using standard attention (flash-attn not available)")
            
            self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)
            
            # Move to device and set to eval mode
            self.model = self.model.eval().to(self.device)
            
            self.initialized = True
            logger.info(f"DeepSeek-OCR model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek-OCR model: {e}")
            self.initialized = False
    
    def process_image(self, 
                     image_path: str, 
                     prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
                     base_size: int = 1024,
                     image_size: int = 640,
                     crop_mode: bool = True,
                     save_results: bool = True,
                     output_dir: str = None) -> Dict[str, Any]:
        """
        Process an image using DeepSeek-OCR.
        
        Args:
            image_path: Path to the image file
            prompt: Prompt for the OCR model
            base_size: Base resolution for processing
            image_size: Target image size
            crop_mode: Whether to use crop mode
            save_results: Whether to save results to file
            output_dir: Directory to save results (default: temp)
        
        Returns:
            Dictionary with OCR results and extracted content
        """
        if not self.initialized:
            return {
                'success': False,
                'error': 'DeepSeek-OCR model not initialized',
                'extracted_text': '',
                'markdown_content': '',
                'logical_content': []
            }
        
        try:
            # Validate image file
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': f'Image file not found: {image_path}',
                    'extracted_text': '',
                    'markdown_content': '',
                    'logical_content': []
                }
            
            # Set up output directory
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix='elms_ocr_')
            else:
                os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Processing image: {image_path}")
            
            # Process with DeepSeek-OCR
            result = self.model.infer(
                tokenizer=self.tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=save_results,
                test_compress=True
            )
            
            # Extract content from result
            extracted_text = self._extract_text_from_result(result)
            markdown_content = self._extract_markdown_from_result(result)
            logical_content = self._extract_logical_content(extracted_text)
            
            return {
                'success': True,
                'image_path': image_path,
                'extracted_text': extracted_text,
                'markdown_content': markdown_content,
                'logical_content': logical_content,
                'raw_result': result,
                'output_dir': output_dir,
                'processing_params': {
                    'prompt': prompt,
                    'base_size': base_size,
                    'image_size': image_size,
                    'crop_mode': crop_mode
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'extracted_text': '',
                'markdown_content': '',
                'logical_content': []
            }
    
    def process_document(self, 
                        document_path: str,
                        document_type: str = 'auto',
                        **kwargs) -> Dict[str, Any]:
        """
        Process a document (PDF, image, etc.) for logical content extraction.
        
        Args:
            document_path: Path to the document
            document_type: Type of document ('pdf', 'image', 'auto')
            **kwargs: Additional arguments for processing
        
        Returns:
            Dictionary with document processing results
        """
        if document_type == 'auto':
            document_type = self._detect_document_type(document_path)
        
        if document_type == 'pdf':
            return self._process_pdf(document_path, **kwargs)
        elif document_type in ['image', 'jpg', 'jpeg', 'png', 'bmp', 'tiff']:
            return self.process_image(document_path, **kwargs)
        else:
            return {
                'success': False,
                'error': f'Unsupported document type: {document_type}',
                'extracted_text': '',
                'logical_content': []
            }
    
    def extract_logical_statements(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract logical statements from OCR text for reasoning.
        
        Args:
            text: Extracted text from OCR
        
        Returns:
            List of logical statements with metadata
        """
        logical_statements = []
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        for i, sentence in enumerate(sentences):
            # Clean and normalize sentence
            cleaned = self._clean_sentence(sentence)
            if not cleaned or len(cleaned) < 3:
                continue
            
            # Classify statement type
            statement_type = self._classify_statement_type(cleaned)
            
            # Extract logical elements
            logical_elements = self._extract_logical_elements(cleaned)
            
            if logical_elements:
                logical_statements.append({
                    'id': f'ocr_stmt_{i}',
                    'text': cleaned,
                    'original_text': sentence,
                    'statement_type': statement_type,
                    'logical_elements': logical_elements,
                    'confidence': 0.8,  # OCR confidence
                    'source': 'ocr'
                })
        
        return logical_statements
    
    def _detect_document_type(self, file_path: str) -> str:
        """Detect document type from file extension."""
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return 'pdf'
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
            return 'image'
        else:
            return 'unknown'
    
    def _process_pdf(self, pdf_path: str, **kwargs) -> Dict[str, Any]:
        """Process PDF document by converting to images first."""
        try:
            from pdf2image import convert_from_path
            
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            all_results = []
            all_text = []
            all_logical_content = []
            
            for i, image in enumerate(images):
                # Save temporary image
                temp_image_path = tempfile.mktemp(suffix=f'_page_{i}.png')
                image.save(temp_image_path)
                
                # Process image
                result = self.process_image(temp_image_path, **kwargs)
                all_results.append(result)
                
                if result['success']:
                    all_text.append(result['extracted_text'])
                    all_logical_content.extend(result['logical_content'])
                
                # Clean up temp file
                os.unlink(temp_image_path)
            
            # Combine results
            combined_text = '\n\n'.join(all_text)
            combined_logical = self.extract_logical_statements(combined_text)
            
            return {
                'success': any(r['success'] for r in all_results),
                'document_path': pdf_path,
                'page_count': len(images),
                'extracted_text': combined_text,
                'logical_content': combined_logical,
                'page_results': all_results
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'extracted_text': '',
                'logical_content': []
            }
    
    def _extract_text_from_result(self, result) -> str:
        """Extract text content from DeepSeek-OCR result."""
        if isinstance(result, str):
            return result
        elif isinstance(result, dict) and 'text' in result:
            return result['text']
        elif isinstance(result, list) and len(result) > 0:
            return str(result[0])
        else:
            return str(result)
    
    def _extract_markdown_from_result(self, result) -> str:
        """Extract markdown content from DeepSeek-OCR result."""
        # This would depend on the specific format of DeepSeek-OCR output
        # For now, return the text as markdown
        text = self._extract_text_from_result(result)
        return text
    
    def _extract_logical_content(self, text: str) -> List[Dict[str, Any]]:
        """Extract logical content from OCR text."""
        return self.extract_logical_statements(text)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _clean_sentence(self, sentence: str) -> str:
        """Clean and normalize sentence."""
        import re
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', sentence.strip())
        
        # Remove special characters that might interfere with reasoning
        cleaned = re.sub(r'[^\w\s.,;:!?-]', '', cleaned)
        
        return cleaned
    
    def _classify_statement_type(self, sentence: str) -> str:
        """Classify the type of logical statement."""
        sentence_lower = sentence.lower()
        
        # Question
        if sentence.endswith('?') or any(word in sentence_lower for word in ['who', 'what', 'where', 'when', 'why', 'how']):
            return 'question'
        
        # Universal statement
        elif any(word in sentence_lower for word in ['all', 'every', 'each', 'any']):
            return 'universal'
        
        # Existential statement
        elif any(word in sentence_lower for word in ['some', 'there exists', 'there is']):
            return 'existential'
        
        # Conditional statement
        elif any(word in sentence_lower for word in ['if', 'when', 'unless', 'provided that']):
            return 'conditional'
        
        # Negation
        elif any(word in sentence_lower for word in ['not', 'no', 'never', 'none']):
            return 'negation'
        
        # Factual statement
        else:
            return 'factual'
    
    def _extract_logical_elements(self, sentence: str) -> Dict[str, Any]:
        """Extract logical elements from a sentence."""
        import re
        
        elements = {
            'predicates': [],
            'constants': [],
            'variables': [],
            'quantifiers': [],
            'connectives': []
        }
        
        # Extract predicates (verbs and action words)
        predicate_patterns = [
            r'\b(is|are|was|were|has|have|had|do|does|did|can|could|will|would|should|must)\b',
            r'\b\w+ed\b',  # Past tense verbs
            r'\b\w+ing\b',  # Present participle
        ]
        
        for pattern in predicate_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            elements['predicates'].extend(matches)
        
        # Extract constants (proper nouns and capitalized words)
        constant_matches = re.findall(r'\b[A-Z][a-z]+\b', sentence)
        elements['constants'].extend(constant_matches)
        
        # Extract quantifiers
        quantifier_patterns = [
            r'\b(all|every|each|any|some|many|few|most|no|none)\b',
            r'\b(there exists|there is|for all|for some)\b'
        ]
        
        for pattern in quantifier_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            elements['quantifiers'].extend(matches)
        
        # Extract connectives
        connective_patterns = [
            r'\b(and|or|but|however|therefore|thus|hence|so|because|since|if|when|unless)\b'
        ]
        
        for pattern in connective_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            elements['connectives'].extend(matches)
        
        return elements


def test_ocr_processor():
    """Test the OCR processor with a sample image."""
    processor = OCRProcessor()
    
    if not processor.initialized:
        print("❌ OCR processor not initialized. Install dependencies first.")
        return
    
    # Test with a sample image (you would need to provide one)
    print("🔍 OCR Processor initialized successfully!")
    print(f"Device: {processor.device}")
    print(f"Model: {processor.model_name}")
    
    # Example usage:
    # result = processor.process_image("sample_document.jpg")
    # print(f"Extracted text: {result['extracted_text']}")


if __name__ == "__main__":
    test_ocr_processor()
