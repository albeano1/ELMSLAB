"""
Simplified OCR Processor for ELMS Visual Reasoning
Uses basic OCR capabilities when DeepSeek-OCR is not available
"""

import os
import json
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Union
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not available. Install with: pip install pytesseract")

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")


class SimpleOCRProcessor:
    """
    Simplified OCR processor that works with available dependencies.
    Falls back to Tesseract when DeepSeek-OCR is not available.
    """
    
    def __init__(self, model_name: str = 'deepseek-ai/DeepSeek-OCR', device: str = 'auto'):
        """
        Initialize the simplified OCR processor.
        
        Args:
            model_name: Model name (for compatibility)
            device: Device to run on
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.initialized = False
        self.ocr_method = None
        
        # Try to initialize with available methods
        self._initialize_ocr()
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == 'auto':
            if hasattr(__import__('torch'), 'cuda') and __import__('torch').cuda.is_available():
                return 'cuda'
            elif hasattr(__import__('torch'), 'backends') and hasattr(__import__('torch').backends, 'mps') and __import__('torch').backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device
    
    def _initialize_ocr(self):
        """Initialize OCR with available methods."""
        # Try DeepSeek-OCR first
        if self._try_deepseek_ocr():
            self.ocr_method = 'deepseek'
            self.initialized = True
            logger.info("✅ DeepSeek-OCR initialized successfully")
            return
        
        # Try Tesseract as fallback
        if self._try_tesseract():
            self.ocr_method = 'tesseract'
            self.initialized = True
            logger.info("✅ Tesseract OCR initialized successfully")
            return
        
        # No OCR available
        logger.warning("❌ No OCR method available. Install tesseract or fix DeepSeek-OCR dependencies.")
        self.initialized = False
    
    def _try_deepseek_ocr(self) -> bool:
        """Try to initialize DeepSeek-OCR."""
        try:
            if not TRANSFORMERS_AVAILABLE:
                return False
            
            # Try to load the model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Try to load model with minimal configuration
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=__import__('torch').float32
            )
            
            self.model = self.model.eval()
            if self.device != 'cpu':
                self.model = self.model.to(self.device)
            
            return True
            
        except Exception as e:
            logger.warning(f"DeepSeek-OCR initialization failed: {e}")
            return False
    
    def _try_tesseract(self) -> bool:
        """Try to initialize Tesseract OCR."""
        try:
            if not TESSERACT_AVAILABLE:
                return False
            
            # Test tesseract
            pytesseract.get_tesseract_version()
            return True
            
        except Exception as e:
            logger.warning(f"Tesseract initialization failed: {e}")
            return False
    
    def process_image(self, 
                     image_path: str, 
                     prompt: str = "Extract text from this image",
                     **kwargs) -> Dict[str, Any]:
        """
        Process an image using available OCR method.
        
        Args:
            image_path: Path to the image file
            prompt: Prompt for OCR (used for DeepSeek-OCR)
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with OCR results
        """
        if not self.initialized:
            return {
                'success': False,
                'error': 'OCR processor not initialized',
                'extracted_text': '',
                'markdown_content': '',
                'logical_content': []
            }
        
        try:
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': f'Image file not found: {image_path}',
                    'extracted_text': '',
                    'markdown_content': '',
                    'logical_content': []
                }
            
            logger.info(f"Processing image with {self.ocr_method}: {image_path}")
            
            if self.ocr_method == 'deepseek':
                return self._process_with_deepseek(image_path, prompt, **kwargs)
            elif self.ocr_method == 'tesseract':
                return self._process_with_tesseract(image_path, **kwargs)
            else:
                return {
                    'success': False,
                    'error': 'No OCR method available',
                    'extracted_text': '',
                    'markdown_content': '',
                    'logical_content': []
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
    
    def _process_with_deepseek(self, image_path: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Process image with DeepSeek-OCR."""
        try:
            # This is a simplified version - in practice you'd use the full DeepSeek-OCR pipeline
            result = self.model.infer(
                tokenizer=self.tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=tempfile.mkdtemp(),
                base_size=kwargs.get('base_size', 1024),
                image_size=kwargs.get('image_size', 640),
                crop_mode=kwargs.get('crop_mode', True),
                save_results=False,
                test_compress=True
            )
            
            extracted_text = str(result) if result else ""
            logical_content = self._extract_logical_content(extracted_text)
            
            return {
                'success': True,
                'image_path': image_path,
                'extracted_text': extracted_text,
                'markdown_content': extracted_text,
                'logical_content': logical_content,
                'method': 'deepseek'
            }
            
        except Exception as e:
            logger.error(f"DeepSeek-OCR processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'extracted_text': '',
                'logical_content': []
            }
    
    def _process_with_tesseract(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """Process image with Tesseract OCR."""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': 'Could not load image',
                    'extracted_text': '',
                    'logical_content': []
                }
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract text using Tesseract
            extracted_text = pytesseract.image_to_string(image_rgb)
            
            # Clean up text
            extracted_text = extracted_text.strip()
            
            # Extract logical content
            logical_content = self._extract_logical_content(extracted_text)
            
            return {
                'success': True,
                'image_path': image_path,
                'extracted_text': extracted_text,
                'markdown_content': extracted_text,
                'logical_content': logical_content,
                'method': 'tesseract'
            }
            
        except Exception as e:
            logger.error(f"Tesseract processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'extracted_text': '',
                'logical_content': []
            }
    
    def _extract_logical_content(self, text: str) -> List[Dict[str, Any]]:
        """Extract logical content from OCR text."""
        if not text.strip():
            return []
        
        logical_statements = []
        sentences = self._split_into_sentences(text)
        
        for i, sentence in enumerate(sentences):
            cleaned = self._clean_sentence(sentence)
            if not cleaned or len(cleaned) < 3:
                continue
            
            statement_type = self._classify_statement_type(cleaned)
            logical_elements = self._extract_logical_elements(cleaned)
            
            if logical_elements:
                logical_statements.append({
                    'id': f'ocr_stmt_{i}',
                    'text': cleaned,
                    'original_text': sentence,
                    'statement_type': statement_type,
                    'logical_elements': logical_elements,
                    'confidence': 0.7,  # OCR confidence
                    'source': 'ocr'
                })
        
        return logical_statements
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _clean_sentence(self, sentence: str) -> str:
        """Clean and normalize sentence."""
        import re
        cleaned = re.sub(r'\s+', ' ', sentence.strip())
        cleaned = re.sub(r'[^\w\s.,;:!?-]', '', cleaned)
        return cleaned
    
    def _classify_statement_type(self, sentence: str) -> str:
        """Classify the type of logical statement."""
        sentence_lower = sentence.lower()
        
        if sentence.endswith('?') or any(word in sentence_lower for word in ['who', 'what', 'where', 'when', 'why', 'how']):
            return 'question'
        elif any(word in sentence_lower for word in ['all', 'every', 'each', 'any']):
            return 'universal'
        elif any(word in sentence_lower for word in ['some', 'there exists', 'there is']):
            return 'existential'
        elif any(word in sentence_lower for word in ['if', 'when', 'unless', 'provided that']):
            return 'conditional'
        elif any(word in sentence_lower for word in ['not', 'no', 'never', 'none']):
            return 'negation'
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
        
        # Extract predicates
        predicate_patterns = [
            r'\b(is|are|was|were|has|have|had|do|does|did|can|could|will|would|should|must)\b',
            r'\b\w+ed\b',
            r'\b\w+ing\b',
        ]
        
        for pattern in predicate_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            elements['predicates'].extend(matches)
        
        # Extract constants
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
            combined_logical = self._extract_logical_content(combined_text)
            
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


def test_simple_ocr():
    """Test the simplified OCR processor."""
    processor = SimpleOCRProcessor()
    
    if processor.initialized:
        print("✅ Simple OCR Processor initialized successfully!")
        print(f"   Method: {processor.ocr_method}")
        print(f"   Device: {processor.device}")
    else:
        print("❌ Simple OCR Processor not initialized")
        print("   Install tesseract: pip install pytesseract")
        print("   Or fix DeepSeek-OCR dependencies")


if __name__ == "__main__":
    test_simple_ocr()
