# ELMS-NEXT: High-Performance Logical Reasoning System

## 🎯 Project Overview
A high-accuracy, high-speed logical reasoning system that replaces Vectionary with custom-trained models and optimized parsing engines. Target: <10ms latency, >99% accuracy, >1000 req/s throughput.

## 📚 Current System Context

### **What is Vectionary?**
Vectionary is a semantic parsing API service that converts natural language into structured semantic trees with:
- **Semantic roles**: agent, patient, theme, modifier, goal, etc.
- **POS tags**: NOUN, VERB, ADJ, ADV, PRON, etc.
- **Dependency relationships**: root, modifier, complement, etc.
- **Definitions**: Word definitions and semantic information
- **Tree structure**: Hierarchical representation of sentence structure

**Current Vectionary Integration:**
```python
# Current ELMS.py implementation
class VectionaryAPIClient:
    """Handles communication with Vectionary parsing API"""
    ENDPOINTS = {
        'prod': 'https://us-central1-parsimony-server.cloudfunctions.net/arborize/arborize/mod1'
    }
    
    def get_trees(self, text: str) -> List[Dict]:
        """Get semantic trees from Vectionary API"""
        # Makes HTTP requests to Vectionary service
        # Returns structured semantic trees

class VectionaryParser:
    """Parses text using Vectionary trees into logical formulas"""
    
    def parse(self, text: str) -> ParsedStatement:
        """Parse text into logical statement using Vectionary"""
        trees = self.api_client.get_trees(text)
        # Extract semantic roles, POS tags, dependencies
        # Convert to Prolog facts/rules
```

**Current System Architecture:**
```
ELMSLAB/
├── ELMS.py                    # Core dynamic converter + Vectionary integration
├── serv_vectionary.py         # FastAPI server with Vectionary calls
├── prolog_reasoner.py         # Prolog inference engine
├── vectionary_knowledge_base.py # Knowledge base management
├── elms-chat-react/           # React web interface
└── tests/                     # Test suite
```

**Vectionary Limitations (Why We Need ELMS-NEXT):**
1. **Network Dependency**: Requires internet connection and API calls
2. **Latency**: 100-500ms per parsing request
3. **Rate Limits**: API throttling and usage limits
4. **Cost**: Per-request pricing model
5. **Reliability**: External service dependency
6. **Customization**: Limited control over parsing logic
7. **Offline Usage**: Cannot work without internet

**ELMS-NEXT Goals:**
- Replace Vectionary API calls with local models
- Achieve <10ms parsing latency (50x faster)
- Enable offline operation
- Provide unlimited usage
- Allow custom model training
- Maintain >99% accuracy on logical reasoning

## 📁 Project Structure

```
ELMS-NEXT/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
│
├── core/                                    # Core reasoning engine
│   ├── __init__.py
│   ├── semantic_parser/                     # Enhanced semantic parsing
│   │   ├── __init__.py
│   │   ├── hybrid_parser.py                # Main hybrid parser
│   │   ├── rule_based_parser.py            # Fast rule-based parsing
│   │   ├── pattern_matcher.py              # Pattern matching engine
│   │   ├── cache_manager.py                # Parsing cache
│   │   └── sentence_structure_detector.py  # Custom structure detection
│   │
│   ├── logic_engine/                       # High-speed Prolog engine
│   │   ├── __init__.py
│   │   ├── optimized_prolog.py             # Custom Prolog engine
│   │   ├── fact_index.py                   # O(1) fact lookup
│   │   ├── rule_compiler.py                # Compiled rules
│   │   ├── query_optimizer.py              # Query optimization
│   │   └── parallel_executor.py            # Parallel query execution
│   │
│   ├── knowledge_graph/                    # Graph-based knowledge
│   │   ├── __init__.py
│   │   ├── graph_engine.py                 # Main graph engine
│   │   ├── entity_index.py                 # Entity indexing
│   │   ├── relation_index.py               # Relation indexing
│   │   └── property_index.py               # Property indexing
│   │
│   ├── inference_engine/                   # Multi-strategy inference
│   │   ├── __init__.py
│   │   ├── forward_chaining.py             # Forward chaining
│   │   ├── backward_chaining.py            # Backward chaining
│   │   ├── resolution_engine.py            # Resolution theorem proving
│   │   └── hybrid_inference.py             # Combined strategies
│   │
│   └── utils/                              # Core utilities
│       ├── __init__.py
│       ├── text_preprocessor.py            # Text preprocessing
│       ├── pattern_utils.py                # Pattern utilities
│       └── performance_monitor.py          # Performance monitoring
│
├── models/                                 # Custom trained models
│   ├── __init__.py
│   ├── semantic_classifier/                # Semantic pattern classification
│   │   ├── __init__.py
│   │   ├── model.py                        # Model architecture
│   │   ├── trainer.py                      # Training pipeline
│   │   ├── data_generator.py               # Training data generation
│   │   └── inference.py                    # Model inference
│   │
│   ├── relation_extractor/                 # Relation extraction
│   │   ├── __init__.py
│   │   ├── model.py                        # Relation extraction model
│   │   ├── trainer.py                      # Training pipeline
│   │   ├── data_augmenter.py               # Data augmentation
│   │   └── inference.py                    # Extraction inference
│   │
│   ├── query_optimizer/                    # Query optimization
│   │   ├── __init__.py
│   │   ├── model.py                        # Optimization model
│   │   ├── trainer.py                      # Training pipeline
│   │   └── inference.py                    # Query optimization
│   │
│   └── sentence_structure/                 # Sentence structure detection
│       ├── __init__.py
│       ├── dependency_parser.py            # Dependency parsing
│       ├── constituency_parser.py          # Constituency parsing
│       ├── semantic_role_labeler.py        # Semantic role labeling
│       ├── pos_tagger.py                   # POS tagging
│       └── named_entity_recognizer.py      # NER
│
├── api/                                    # High-performance API
│   ├── __init__.py
│   ├── main.py                             # FastAPI application
│   ├── routes/                             # API routes
│   │   ├── __init__.py
│   │   ├── reasoning.py                    # Reasoning endpoints
│   │   ├── knowledge.py                    # Knowledge management
│   │   └── health.py                       # Health checks
│   ├── middleware/                         # API middleware
│   │   ├── __init__.py
│   │   ├── caching.py                      # Response caching
│   │   ├── rate_limiting.py                # Rate limiting
│   │   └── monitoring.py                   # Performance monitoring
│   └── schemas/                            # Pydantic schemas
│       ├── __init__.py
│       ├── requests.py                     # Request schemas
│       └── responses.py                    # Response schemas
│
├── web/                                    # Next-gen React interface
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── components/                     # React components
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── MessageList.tsx
│   │   │   ├── InputArea.tsx
│   │   │   ├── KnowledgeGraph.tsx
│   │   │   └── PerformanceMonitor.tsx
│   │   ├── hooks/                          # Custom React hooks
│   │   │   ├── useReasoning.ts
│   │   │   ├── useKnowledge.ts
│   │   │   └── usePerformance.ts
│   │   ├── services/                       # API services
│   │   │   ├── api.ts
│   │   │   ├── reasoning.ts
│   │   │   └── knowledge.ts
│   │   ├── types/                          # TypeScript types
│   │   │   ├── reasoning.ts
│   │   │   ├── knowledge.ts
│   │   │   └── performance.ts
│   │   └── styles/                         # Styling
│   │       ├── globals.css
│   │       ├── components.css
│   │       └── animations.css
│   └── dist/                               # Build output
│
├── training/                               # Model training pipeline
│   ├── __init__.py
│   ├── data_generation/                    # Training data generation
│   │   ├── __init__.py
│   │   ├── synthetic_data.py               # Synthetic data generation
│   │   ├── vectionary_augmenter.py         # Vectionary data augmentation
│   │   ├── pattern_templates.py            # Pattern templates
│   │   └── data_validator.py               # Data validation
│   ├── training_scripts/                   # Training scripts
│   │   ├── train_semantic_classifier.py
│   │   ├── train_relation_extractor.py
│   │   ├── train_query_optimizer.py
│   │   └── train_sentence_structure.py
│   ├── configs/                            # Training configurations
│   │   ├── semantic_classifier.yaml
│   │   ├── relation_extractor.yaml
│   │   ├── query_optimizer.yaml
│   │   └── sentence_structure.yaml
│   └── evaluation/                         # Model evaluation
│       ├── __init__.py
│       ├── accuracy_metrics.py
│       ├── performance_metrics.py
│       └── benchmark_suite.py
│
├── benchmarks/                             # Performance testing
│   ├── __init__.py
│   ├── performance_tests.py                # Performance benchmarks
│   ├── accuracy_tests.py                   # Accuracy validation
│   ├── load_tests.py                       # Load testing
│   ├── memory_tests.py                     # Memory usage tests
│   └── data/                               # Test data
│       ├── logical_patterns.json
│       ├── edge_cases.json
│       └── performance_datasets.json
│
├── tests/                                  # Test suite
│   ├── __init__.py
│   ├── unit/                               # Unit tests
│   │   ├── test_semantic_parser.py
│   │   ├── test_logic_engine.py
│   │   ├── test_knowledge_graph.py
│   │   └── test_inference_engine.py
│   ├── integration/                        # Integration tests
│   │   ├── test_api.py
│   │   ├── test_reasoning_pipeline.py
│   │   └── test_web_interface.py
│   └── fixtures/                           # Test fixtures
│       ├── sample_queries.json
│       ├── test_knowledge.json
│       └── expected_results.json
│
├── config/                                 # Configuration files
│   ├── __init__.py
│   ├── settings.py                         # Main settings
│   ├── model_configs.py                    # Model configurations
│   ├── api_configs.py                      # API configurations
│   └── performance_configs.py              # Performance settings
│
├── scripts/                                # Utility scripts
│   ├── setup_environment.py                # Environment setup
│   ├── download_models.py                  # Model downloading
│   ├── benchmark_system.py                 # System benchmarking
│   └── deploy.py                           # Deployment script
│
└── docs/                                   # Documentation
    ├── README.md
    ├── API.md                              # API documentation
    ├── ARCHITECTURE.md                     # System architecture
    ├── PERFORMANCE.md                      # Performance guide
    ├── TRAINING.md                         # Model training guide
    └── DEPLOYMENT.md                       # Deployment guide
```

## 🔄 Migration Strategy from Current System

### **Phase 1: Drop-in Replacement**
```python
# ELMS-NEXT will provide a VectionaryParser-compatible interface
class ELMSNextParser:
    """Drop-in replacement for VectionaryParser"""
    
    def __init__(self, api_client=None):  # Maintain compatibility
        self.hybrid_parser = HybridSemanticParser()
        self.cache = LRUCache(maxsize=10000)
    
    def parse(self, text: str) -> ParsedStatement:
        """Same interface as current VectionaryParser"""
        # Use local models instead of API calls
        tree = self.hybrid_parser.parse(text)
        return self._convert_to_parsed_statement(tree)
```

### **Phase 2: Enhanced Features**
- Add new parsing capabilities not available in Vectionary
- Implement advanced caching and optimization
- Add custom model training capabilities

### **Phase 3: Full Migration**
- Remove Vectionary dependencies
- Optimize for local-only operation
- Add offline capabilities

## 🧠 Best Solutions for Sentence Structure Detection

### 1. **Hybrid Parsing Architecture**
```python
# core/semantic_parser/hybrid_parser.py
class HybridSemanticParser:
    """
    Combines multiple parsing strategies for maximum accuracy and speed
    """
    
    def __init__(self):
        # Fast rule-based parser for common patterns
        self.rule_parser = RuleBasedParser()
        
        # Custom-trained models for complex structures
        self.dependency_parser = DependencyParser()
        self.constituency_parser = ConstituencyParser()
        self.semantic_role_labeler = SemanticRoleLabeler()
        
        # Caching for performance
        self.cache = LRUCache(maxsize=10000)
    
    def parse(self, text: str) -> SemanticTree:
        # 1. Check cache first (fastest)
        if cached := self.cache.get(text):
            return cached
        
        # 2. Try rule-based parsing (very fast, high accuracy for common patterns)
        if result := self.rule_parser.parse(text):
            self.cache[text] = result
            return result
        
        # 3. Use custom models for complex structures
        result = self._parse_with_models(text)
        self.cache[text] = result
        return result
```

### 2. **Rule-Based Parser (Primary)**
```python
# core/semantic_parser/rule_based_parser.py
class RuleBasedParser:
    """
    Ultra-fast rule-based parser for common logical patterns
    Covers 80% of logical reasoning cases with 99.9% accuracy
    """
    
    def __init__(self):
        self.patterns = {
            # Possession patterns
            "possession": [
                r"(\w+)'s\s+(\w+)",  # "Mary's children"
                r"(\w+)\s+of\s+(\w+)",  # "children of Mary"
                r"(\w+)\s+has\s+(\w+)",  # "Mary has children"
            ],
            
            # Quantification patterns
            "quantification": [
                r"All\s+(\w+)\s+are\s+(\w+)",  # "All cats are mammals"
                r"Some\s+(\w+)\s+are\s+(\w+)",  # "Some birds can fly"
                r"Every\s+(\w+)\s+is\s+(\w+)",  # "Every student studies"
            ],
            
            # Relation patterns
            "relation": [
                r"(\w+)\s+is\s+(\w+)\s+of\s+(\w+)",  # "Alice is parent of Bob"
                r"(\w+)\s+gives\s+(\w+)\s+to\s+(\w+)",  # "John gives book to Mary"
                r"(\w+)\s+studies\s+(\w+)",  # "Maria studies regularly"
            ],
            
            # Question patterns
            "question": [
                r"Who\s+are\s+(\w+)\s+who\s+(\w+)",  # "Who are students who study"
                r"What\s+(\w+)\s+do\s+we\s+have",  # "What mammals do we have"
                r"Who\s+makes\s+(\w+)",  # "Who makes decisions"
            ]
        }
    
    def parse(self, text: str) -> Optional[SemanticTree]:
        for pattern_type, patterns in self.patterns.items():
            for pattern in patterns:
                if match := re.match(pattern, text, re.IGNORECASE):
                    return self._build_semantic_tree(pattern_type, match, text)
        return None
```

### 3. **Custom Dependency Parser**
```python
# models/sentence_structure/dependency_parser.py
class DependencyParser:
    """
    Custom-trained dependency parser optimized for logical reasoning
    Smaller, faster, more accurate than general-purpose parsers
    """
    
    def __init__(self):
        self.model = self._load_model()
        self.vocab = self._load_vocab()
    
    def parse(self, text: str) -> DependencyTree:
        # Tokenize and encode
        tokens = self._tokenize(text)
        encoded = self._encode(tokens)
        
        # Predict dependencies
        dependencies = self.model.predict(encoded)
        
        # Build dependency tree
        return self._build_tree(tokens, dependencies)
```

### 4. **Semantic Role Labeler**
```python
# models/sentence_structure/semantic_role_labeler.py
class SemanticRoleLabeler:
    """
    Custom semantic role labeling for logical relations
    Trained specifically on logical reasoning patterns
    """
    
    def __init__(self):
        self.model = self._load_model()
        self.role_mappings = {
            'agent': 'subject',
            'patient': 'object',
            'theme': 'predicate',
            'goal': 'indirect_object',
            'modifier': 'adjective',
            'possessive': 'possessor'
        }
    
    def label_roles(self, dependency_tree: DependencyTree) -> Dict[str, str]:
        # Extract semantic roles from dependency tree
        roles = {}
        
        for token in dependency_tree.tokens:
            if token.dep in self.role_mappings:
                role = self.role_mappings[token.dep]
                roles[role] = token.text.lower()
        
        return roles
```

### 5. **Pattern Matching Engine**
```python
# core/semantic_parser/pattern_matcher.py
class PatternMatcher:
    """
    High-performance pattern matching for logical structures
    Uses compiled regex and optimized matching algorithms
    """
    
    def __init__(self):
        self.compiled_patterns = self._compile_patterns()
        self.pattern_cache = {}
    
    def match_pattern(self, text: str) -> Optional[PatternMatch]:
        # Use compiled patterns for maximum speed
        for pattern_name, pattern in self.compiled_patterns.items():
            if match := pattern.search(text):
                return PatternMatch(
                    name=pattern_name,
                    groups=match.groups(),
                    span=match.span()
                )
        return None
```

### 6. **Vectionary-Compatible Output Format**
```python
# core/semantic_parser/vectionary_compat.py
class VectionaryCompatibleParser:
    """
    Generates Vectionary-compatible output format for seamless migration
    """
    
    def parse(self, text: str) -> Dict:
        """Generate Vectionary-style semantic tree"""
        # Parse with local models
        local_tree = self.hybrid_parser.parse(text)
        
        # Convert to Vectionary format
        return {
            'id': f"{local_tree.lemma}_{local_tree.pos}_{local_tree.index}",
            'char_index': local_tree.char_index,
            'definition': local_tree.definition,
            'dependency': local_tree.dependency,
            'index': local_tree.index,
            'lemma': local_tree.lemma,
            'mood': local_tree.mood,
            'pos': local_tree.pos,
            'tense': local_tree.tense,
            'text': local_tree.text,
            'children': self._convert_children(local_tree.children)
        }
    
    def _convert_children(self, children: List) -> List[Dict]:
        """Convert children to Vectionary format"""
        vectionary_children = []
        for child in children:
            vectionary_children.append({
                'text': child.text,
                'role': child.role,
                'pos': child.pos,
                'number': child.number,
                'person': child.person,
                'children': self._convert_children(child.children) if child.children else []
            })
        return vectionary_children
```

## 🚀 Key Implementation Strategies

### 1. **Performance Optimization**
- **Caching**: Multi-level caching (L1: memory, L2: Redis, L3: disk)
- **Compilation**: Pre-compile common patterns and rules
- **Indexing**: O(1) lookups for facts and relations
- **Parallelization**: Parallel query execution where safe

### 2. **Accuracy Enhancement**
- **Specialized Training**: Models trained specifically on logical reasoning
- **Pattern Templates**: Comprehensive pattern library
- **Validation**: Multi-layer validation of parsing results
- **Fallback Strategies**: Graceful degradation for edge cases

### 3. **Scalability Design**
- **Stateless Architecture**: Horizontal scaling capability
- **Microservices**: Modular, independently scalable components
- **Load Balancing**: Intelligent request distribution
- **Resource Management**: Efficient memory and CPU usage

## 📊 Expected Performance Metrics

- **Latency**: <10ms average, <50ms 99th percentile
- **Accuracy**: >99% on logical reasoning tasks
- **Throughput**: >1000 requests/second
- **Memory**: <500MB for full system
- **Model Size**: <50MB total for all models

## 🛠️ Development Phases

### Phase 1: Core Engine (2-3 weeks)
- [ ] Rule-based parser implementation
- [ ] Basic Prolog engine optimization
- [ ] Caching system
- [ ] API foundation

### Phase 2: Custom Models (3-4 weeks)
- [ ] Training data generation
- [ ] Model training pipelines
- [ ] Model optimization
- [ ] Integration testing

### Phase 3: Advanced Features (2-3 weeks)
- [ ] Knowledge graph implementation
- [ ] Advanced inference strategies
- [ ] Performance optimization
- [ ] Web interface

### Phase 4: Production (1-2 weeks)
- [ ] Comprehensive testing
- [ ] Performance benchmarking
- [ ] Production deployment
- [ ] Monitoring and alerting

This architecture provides a complete roadmap for building a high-performance logical reasoning system that addresses Vectionary's limitations while maintaining accuracy and speed.

## 📘 Semantic Tree Schema (with Definitions)

To enable true reasoning and deterministic conversion to logic, include rich dictionary-style definitions on each node and use them systematically.

### Node fields
```json
{
  "ID": "make_V_1.1",
  "text": "make",
  "lemma": "make",
  "pos": "VERB",
  "dependency": "ROOT",
  "index": 3,
  "char_index": 12,
  "mood": "INDICATIVE",
  "tense": "PRESENT",
  "role": "root",
  "definition": "to build, construct, produce, or originate",
  "sense_id": "make.v.01",           
  "frame_id": "Causation|Creation",  
  "ontology_refs": [
    {"kb": "ELMS-ONT", "concept_id": "ELMS:CREATE", "confidence": 0.92}
  ],
  "children": [
    {"role": "agent", "text": "Directors", "definition": "members of a board with decision authority"},
    {"role": "patient", "text": "decisions", "definition": "acts of making a choice"}
  ]
}
```

### How "definition" improves reasoning
- Predicate grounding: map surface forms to canonical predicates using definition cues
  - Example: definition contains "produce/originate" → predicate_hint: "produce/ originate"
- Sense disambiguation: choose sense_id aligned with role structure and definition terms
- Rule synthesis: extract hypernyms/entailments from definitions to propose rules
  - Example: "director: member of a board with decision authority" → rule: director(X) → makes_decisions(X)
- Ontology linking: attach `ontology_refs` based on definition similarity to ELMS ontology
- Confidence shaping: boost conversions when definition aligns with observed roles

### Deterministic conversion using definitions
```python
# core/semantic_parser/definition_grounding.py
class DefinitionGrounder:
    def ground_predicate(self, node) -> str:
        text = (node.get("definition") or "").lower()
        lemma = node.get("lemma", "").lower()
        # Priority 1: curated mapping by definition keywords
        if any(k in text for k in ["produce", "originate", "create"]):
            return "produce"
        if any(k in text for k in ["decide", "decision", "determine"]):
            return "make_decisions"
        # Priority 2: frame/roleset hints
        frame = node.get("frame_id") or ""
        if "Giving" in frame:
            return "give"
        # Priority 3: fallback to lemma
        return lemma

    def propose_rules_from_definition(self, noun_node) -> list[str]:
        defs = (noun_node.get("definition") or "").lower()
        rules = []
        if "decision authority" in defs or "authority to decide" in defs:
            rules.append("makes_decisions(X) :- director(X).")
        if "offspring" in defs and noun_node.get("lemma") == "child":
            rules.append("children(Y,X) :- parent(X,Y).")
        return rules
```

### Storage recommendations
- Always populate `definition` for content words (VERB, NOUN, ADJ) when available
- Persist `sense_id`, `frame_id`, and `ontology_refs` for auditability
- Cache grounding results by `canonical_key` (lemma+roles+args)

### Query-time use
- Use definition-grounded predicate names for query formation to avoid collective nouns (e.g., "directors" → variable over `director(X)`)
- When answer type is ambiguous, prefer definitions that imply individual-level predicates

