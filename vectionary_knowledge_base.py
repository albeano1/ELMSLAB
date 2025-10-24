"""
Vectionary-Based Knowledge Base for Enhanced Logic Reasoning
"""

import json
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict, field
from collections import defaultdict

from ELMS import LogicalReasoner, VectionaryParser, VectionaryAPIClient, ParsedStatement, LogicType


@dataclass
class VectionaryFact:
    """Represents a fact in the knowledge base with Vectionary parsing"""
    id: str
    text: str
    parsed_statement: ParsedStatement
    confidence: float
    source: str  # 'user', 'inferred', 'learned'
    timestamp: datetime
    tags: List[str] = field(default_factory=list)
    related_facts: List[str] = field(default_factory=list)


class VectionaryKnowledgeBase:
    """
    Enhanced Knowledge Base using Vectionary parsing.
    
    Features:
    - Store facts with Vectionary parsing
    - Query using natural language with automatic parsing
    - Infer new facts using logical reasoning
    - Temporal reasoning support
    - Contradiction detection
    - Confidence-based fact ranking
    """
    
    def __init__(self, storage_file: str = "vectionary_knowledge_base.json"):
        self.storage_file = storage_file
        self.facts: Dict[str, VectionaryFact] = {}
        self.fact_counter = 0
        # Initialize the enhanced reasoning engine
        api_client = VectionaryAPIClient(environment='prod')
        vectionary_parser = VectionaryParser(api_client)
        self.vectionary_engine = LogicalReasoner(vectionary_parser)
        
        # Indexes for efficient querying
        self.indexes = {
            'predicates': defaultdict(set),
            'constants': defaultdict(set),
            'variables': defaultdict(set),
            'atoms': defaultdict(set),
            'temporal': defaultdict(set),
            'logic_type': defaultdict(set)
        }
        
        self.load_from_file()
    
    def add_fact(self, text: str, confidence: float = 0.95, 
                 source: str = "user", tags: List[str] = None) -> VectionaryFact:
        """
        Add a fact to the knowledge base with Vectionary parsing.
        
        Args:
            text: Natural language fact
            confidence: Initial confidence (0-1)
            source: Source of the fact
            tags: Optional tags for categorization
        
        Returns:
            VectionaryFact: The created fact object
        """
        print(f"📝 Adding fact to knowledge base: {text}")
        
        # Parse the fact using Vectionary
        parsed_statement = self.vectionary_engine.parser.parse(text)
        
        # Generate ID
        self.fact_counter += 1
        fact_id = f"fact_{self.fact_counter}"
        
        # Create fact
        fact = VectionaryFact(
            id=fact_id,
            text=text,
            parsed_statement=parsed_statement,
            confidence=confidence,
            source=source,
            timestamp=datetime.now(),
            tags=tags or [],
            related_facts=[]
        )
        
        # Store fact
        self.facts[fact_id] = fact
        
        # Update indexes
        self._update_indexes(fact)
        
        # Try to infer new facts
        if source != "inferred":
            self._infer_new_facts(fact)
        
        # Save to file
        self.save_to_file()
        
        print(f"✅ Fact added: {fact_id} (confidence: {confidence})")
        return fact
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the knowledge base using natural language.
        
        Args:
            question: Natural language question
        
        Returns:
            Dict containing answer, confidence, reasoning, and relevant facts
        """
        print(f"🔍 Querying knowledge base: {question}")
        
        # Parse the question using Vectionary
        parsed_query = self.vectionary_engine.parser.parse(question)
        
        # Find relevant facts
        relevant_facts = self._find_relevant_facts(parsed_query)
        
        if not relevant_facts:
            return {
                "answer": "Unknown",
                "confidence": 0.0,
                "reasoning": "No relevant facts found in knowledge base",
                "relevant_facts": [],
                "parsed_query": parsed_query.formula
            }
        
        # Use Vectionary reasoning to answer the question
        premises = [fact.text for fact in relevant_facts]
        
        try:
            result = self.vectionary_engine.prove_theorem(premises, question)
            
            return {
                "answer": "Yes" if result.get('valid', False) else "No",
                "confidence": self._map_confidence(result.get('confidence', 0.0)),
                "reasoning": result.get('explanation', 'Based on Vectionary reasoning'),
                "reasoning_steps": result.get('reasoning_steps', []),
                "relevant_facts": [fact.text for fact in relevant_facts[:5]],
                "parsed_query": parsed_query.formula,
                "logic_type": result.get('logic_type', 'unknown')
            }
        except Exception as e:
            print(f"❌ Error in Vectionary reasoning: {e}")
            return {
                "answer": "Unknown",
                "confidence": 0.0,
                "reasoning": f"Error in reasoning: {str(e)}",
                "relevant_facts": [fact.text for fact in relevant_facts[:3]],
                "parsed_query": parsed_query.formula
            }
    
    def get_all_facts(self) -> List[Dict[str, Any]]:
        """Get all facts in the knowledge base."""
        return [
            {
                "id": fact.id,
                "text": fact.text,
                "formula": fact.parsed_statement.formula,
                "logic_type": fact.parsed_statement.logic_type.value,
                "confidence": fact.confidence,
                "source": fact.source,
                "timestamp": fact.timestamp.isoformat(),
                "tags": fact.tags,
                "predicates": fact.parsed_statement.predicates,
                "constants": fact.parsed_statement.constants,
                "vectionary_enhanced": fact.parsed_statement.vectionary_enhanced
            }
            for fact in self.facts.values()
        ]
    
    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact from the knowledge base."""
        if fact_id in self.facts:
            fact = self.facts[fact_id]
            # Remove from indexes
            self._remove_from_indexes(fact)
            # Delete fact
            del self.facts[fact_id]
            self.save_to_file()
            print(f"🗑️ Deleted fact: {fact_id}")
            return True
        return False
    
    def clear_all_facts(self):
        """Clear all facts from the knowledge base."""
        self.facts.clear()
        self.fact_counter = 0
        self.indexes = {
            'predicates': defaultdict(set),
            'constants': defaultdict(set),
            'variables': defaultdict(set),
            'atoms': defaultdict(set),
            'temporal': defaultdict(set),
            'logic_type': defaultdict(set)
        }
        self.save_to_file()
        print("🗑️ All facts cleared from knowledge base")
    
    def _find_relevant_facts(self, parsed_query: ParsedStatement) -> List[VectionaryFact]:
        """Find facts relevant to a parsed query."""
        relevant_facts = []
        
        # Search by predicates
        for predicate in parsed_query.predicates:
            for fact_id in self.indexes['predicates'].get(predicate, set()):
                if fact_id in self.facts:
                    relevant_facts.append(self.facts[fact_id])
        
        # Search by constants
        for constant in parsed_query.constants:
            for fact_id in self.indexes['constants'].get(constant, set()):
                if fact_id in self.facts:
                    relevant_facts.append(self.facts[fact_id])
        
        # Search by atoms
        for atom in parsed_query.atoms:
            for fact_id in self.indexes['atoms'].get(atom, set()):
                if fact_id in self.facts:
                    relevant_facts.append(self.facts[fact_id])
        
        # If no specific matches, return all facts (for general queries)
        if not relevant_facts:
            relevant_facts = list(self.facts.values())
        
        # Remove duplicates and sort by confidence
        unique_facts = list({fact.id: fact for fact in relevant_facts}.values())
        unique_facts.sort(key=lambda f: f.confidence, reverse=True)
        
        return unique_facts[:10]  # Return top 10 most relevant facts
    
    def _update_indexes(self, fact: VectionaryFact):
        """Update search indexes with a new fact."""
        parsed = fact.parsed_statement
        
        # Index by predicates
        for predicate in parsed.predicates:
            self.indexes['predicates'][predicate].add(fact.id)
        
        # Index by constants
        for constant in parsed.constants:
            self.indexes['constants'][constant].add(fact.id)
        
        # Index by variables
        for variable in parsed.variables:
            self.indexes['variables'][variable].add(fact.id)
        
        # Index by atoms
        for atom in parsed.atoms:
            self.indexes['atoms'][atom].add(fact.id)
        
        # Index by logic type
        self.indexes['logic_type'][parsed.logic_type.value].add(fact.id)
        
        # Index temporal facts
        if parsed.logic_type == LogicType.TEMPORAL or parsed.temporal_markers:
            self.indexes['temporal']['all'].add(fact.id)
    
    def _remove_from_indexes(self, fact: VectionaryFact):
        """Remove a fact from all indexes."""
        parsed = fact.parsed_statement
        
        for predicate in parsed.predicates:
            if predicate in self.indexes['predicates']:
                self.indexes['predicates'][predicate].discard(fact.id)
        
        for constant in parsed.constants:
            if constant in self.indexes['constants']:
                self.indexes['constants'][constant].discard(fact.id)
        
        for variable in parsed.variables:
            if variable in self.indexes['variables']:
                self.indexes['variables'][variable].discard(fact.id)
        
        for atom in parsed.atoms:
            if atom in self.indexes['atoms']:
                self.indexes['atoms'][atom].discard(fact.id)
        
        if parsed.logic_type.value in self.indexes['logic_type']:
            self.indexes['logic_type'][parsed.logic_type.value].discard(fact.id)
        
        if 'all' in self.indexes['temporal']:
            self.indexes['temporal']['all'].discard(fact.id)
    
    def _infer_new_facts(self, new_fact: VectionaryFact):
        """Try to infer new facts using Vectionary reasoning."""
        print(f"🔍 Attempting to infer new facts from: {new_fact.text}")
        
        # Get related facts
        related_facts = self._find_relevant_facts(new_fact.parsed_statement)
        
        # Try to apply logical inference rules
        for existing_fact in related_facts[:5]:  # Limit to top 5 to avoid combinatorial explosion
            if existing_fact.id == new_fact.id:
                continue
            
            # Try to infer using Modus Ponens, Universal Instantiation, etc.
            try:
                # Combine the two facts and see if we can infer something new
                premises = [existing_fact.text, new_fact.text]
                
                # Check if this combination leads to a new conclusion
                # (This is a simplified approach - a full system would be more sophisticated)
                inferred_confidence = min(existing_fact.confidence, new_fact.confidence) * 0.85
                
                # Store relationship
                if existing_fact.id not in new_fact.related_facts:
                    new_fact.related_facts.append(existing_fact.id)
                if new_fact.id not in existing_fact.related_facts:
                    existing_fact.related_facts.append(new_fact.id)
                    
            except Exception as e:
                # Inference failed, continue
                pass
    
    def _map_confidence(self, raw_confidence: float) -> float:
        """Map Vectionary confidence to knowledge base confidence."""
        if raw_confidence >= 0.95:
            return 0.95
        elif raw_confidence >= 0.9:
            return 0.9
        elif raw_confidence >= 0.8:
            return 0.8
        elif raw_confidence >= 0.7:
            return 0.7
        else:
            return 0.5
    
    def save_to_file(self):
        """Save the knowledge base to a JSON file."""
        try:
            data = {
                "facts": {},
                "fact_counter": self.fact_counter,
                "indexes": {}
            }
            
            # Serialize facts
            for fact_id, fact in self.facts.items():
                data["facts"][fact_id] = {
                    "id": fact.id,
                    "text": fact.text,
                    "parsed_statement": {
                        "original_text": fact.parsed_statement.original_text,
                        "formula": fact.parsed_statement.formula,
                        "logic_type": fact.parsed_statement.logic_type.value,
                        "confidence": fact.parsed_statement.confidence,
                        "variables": fact.parsed_statement.variables,
                        "constants": fact.parsed_statement.constants,
                        "predicates": fact.parsed_statement.predicates,
                        "atoms": fact.parsed_statement.atoms,
                        "explanation": fact.parsed_statement.explanation,
                        "vectionary_enhanced": fact.parsed_statement.vectionary_enhanced
                    },
                    "confidence": fact.confidence,
                    "source": fact.source,
                    "timestamp": fact.timestamp.isoformat(),
                    "tags": fact.tags,
                    "related_facts": fact.related_facts
                }
            
            # Serialize indexes (convert sets to lists)
            for index_name, index_data in self.indexes.items():
                data["indexes"][index_name] = {
                    key: list(value) for key, value in index_data.items()
                }
            
            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"💾 Knowledge base saved to {self.storage_file}")
            
        except Exception as e:
            print(f"❌ Error saving knowledge base: {e}")
    
    def load_from_file(self):
        """Load the knowledge base from a JSON file."""
        try:
            with open(self.storage_file, 'r') as f:
                data = json.load(f)
            
            # Load facts
            facts_data = data.get("facts", {})
            if isinstance(facts_data, list):
                # Handle case where facts is stored as a list
                facts_data = {}
            for fact_id, fact_data in facts_data.items():
                parsed_data = fact_data["parsed_statement"]
                parsed_statement = ParsedStatement(
                    original_text=parsed_data["original_text"],
                    formula=parsed_data["formula"],
                    logic_type=LogicType(parsed_data["logic_type"]),
                    confidence=parsed_data["confidence"],
                    variables=parsed_data["variables"],
                    constants=parsed_data["constants"],
                    predicates=parsed_data["predicates"],
                    atoms=parsed_data["atoms"],
                    explanation=parsed_data["explanation"],
                    vectionary_enhanced=parsed_data.get("vectionary_enhanced", False)
                )
                
                fact = VectionaryFact(
                    id=fact_data["id"],
                    text=fact_data["text"],
                    parsed_statement=parsed_statement,
                    confidence=fact_data["confidence"],
                    source=fact_data["source"],
                    timestamp=datetime.fromisoformat(fact_data["timestamp"]),
                    tags=fact_data.get("tags", []),
                    related_facts=fact_data.get("related_facts", [])
                )
                
                self.facts[fact_id] = fact
            
            # Load counter
            self.fact_counter = data.get("fact_counter", 0)
            
            # Load indexes (convert lists back to sets)
            for index_name, index_data in data.get("indexes", {}).items():
                self.indexes[index_name] = defaultdict(set, {
                    key: set(value) for key, value in index_data.items()
                })
            
            print(f"📂 Loaded {len(self.facts)} facts from {self.storage_file}")
            
        except FileNotFoundError:
            print(f"📂 No existing knowledge base found, starting fresh")
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")

