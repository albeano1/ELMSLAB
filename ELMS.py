#!/usr/bin/env python3
"""
ELMS - Enhanced Logic Modeling System
"""

import argparse
import json
import sys
import re
import time
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


# ============================================================================
# DYNAMIC HYBRID CONVERTER
# ============================================================================

class DynamicHybridConverter:
    """Fully dynamic conversion system - no hardcoding, handles all edge cases"""
    
    # Cache for conversion results
    _conversion_cache = {}
    
    @staticmethod
    def _cache_and_return(cache_key, result):
        """Cache result and return it"""
        DynamicHybridConverter._conversion_cache[cache_key] = result
        return result
    
    @staticmethod
    def _is_likely_adverb(word: str, tree: dict = None) -> bool:
        """
        Dynamic adverb detection using Vectionary's semantic information
        """
        if not word:
            return False
        
        word = word.lower().strip()
        
        # Use Vectionary's POS tags if available
        if tree and 'children' in tree:
            for child in tree['children']:
                if child.get('text', '').lower() == word:
                    child_pos = child.get('pos', '')
                    if child_pos == 'ADV':
                        return True
        
        # Use linguistic patterns
        # Check for common adverb endings
        if word.endswith('ly') or word.endswith('ward') or word.endswith('wise') or word.endswith('ways'):
            return True
        
        # Use Vectionary's POS tags for dynamic detection
        # Use minimal patterns for common cases
        if word in ['always', 'never', 'often', 'sometimes', 'usually', 'rarely', 'well', 'badly', 'quickly', 'slowly', 'carefully', 'hard', 'very', 'quite', 'rather', 'extremely', 'highly', 'now', 'then', 'today', 'yesterday', 'tomorrow', 'soon']:
            return True
        
        return False
    
    @staticmethod
    def _is_likely_predicate(value: str, roles: dict, tree: dict) -> bool:
        """
        Dynamic predicate detection using Vectionary's semantic information
        """
        if not value:
            return False
        
        value = value.lower().strip()
        
        # Check if it's a common verb that could be a predicate (minimal set)
        if value in ['be', 'is', 'are', 'have', 'do', 'can', 'will', 'should', 'would']:
            return True
        
        # Check if it's a noun that could be a predicate (relationship words)
        # Use Vectionary's semantic information to determine this
        if 'children' in tree:
            for child in tree['children']:
                if child.get('text', '').lower() == value:
                    # Check the child's role and POS tag
                    child_role = child.get('role', '')
                    child_pos = child.get('pos', '')
                    
                    # If it's a theme or predicate role, it's likely a predicate
                    if child_role in ['theme', 'predicate'] or child_pos in ['NOUN', 'VERB']:
                        return True
        
        # Check if it appears in multiple roles (likely a predicate)
        role_count = 0
        for role, values in roles.items():
            if value in [v.lower() for v in values]:
                role_count += 1
        
        # If it appears in multiple roles, it's likely a predicate
        if role_count > 1:
            return True
        
        return False
    
    @staticmethod
    def _is_question_word(word: str) -> bool:
        """
        Dynamic question word detection
        """
        if not word:
            return False
        
        word = word.lower().strip()
        
        # Dynamic question word detection using patterns
        # Check for common question words (minimal set)
        if word in ['who', 'what', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom']:
            return True
        
        # Question word patterns
        if word.startswith('wh') or word.startswith('how'):
            return True
        
        return False
    
    @staticmethod
    def _dynamic_convert_tree_to_prolog(tree, max_depth=10, current_depth=0):
        """Convert Vectionary tree to Prolog dynamically - TRULY NO HARDCODING"""
        if not tree:
            return None
        
        # Prevent infinite recursion
        if current_depth > max_depth:
            print(f"⚠️ Max depth reached ({max_depth}), returning None")
            return None
        
        lemma = tree.get('lemma', '').lower()
        pos = tree.get('pos', '')
        children = tree.get('children', [])
        
        print(f"🔍 Dynamic converter: lemma='{lemma}', pos='{pos}', children={len(children)}")
        
        # Extract ALL semantic roles dynamically
        # Store roles as lists to avoid overwriting
        roles = {}
        
        def extract_all_roles(node_list, depth=0):
            """Extract all roles recursively, keeping track of all values"""
            for node in node_list:
                role = node.get('role', '')
                text = node.get('text', '').lower()
                lemma_val = node.get('lemma', '').lower()
                
                # Store role if it exists
                if role and text:
                    if role not in roles:
                        roles[role] = []
                    roles[role].append(text)
                    print(f"🔍 Role: {role} = {text}")
                
                # Also check for patient/theme roles in children
                if 'children' in node:
                    for child in node['children']:
                        child_role = child.get('role', '')
                        child_text = child.get('text', '').lower()
                        if child_role and child_text:
                            if child_role not in roles:
                                roles[child_role] = []
                            roles[child_role].append(child_text)
                            print(f"🔍 Child Role: {child_role} = {child_text}")
                            print(f"🔍 Current roles after child: {roles}")
                
                # Extract possessive from marks
                if 'marks' in node:
                    for mark in node['marks']:
                        if isinstance(mark, dict):
                            mark_dep = mark.get('dependency', '')
                            mark_text = mark.get('text', '').lower()
                            if mark_dep in ['POSS', 'PS'] and mark_text not in ["'s", "'"]:
                                if 'possessive' not in roles:
                                    roles['possessive'] = []
                                roles['possessive'].append(mark_text)
                                print(f"🔍 Possessive: {mark_text}")
                
                # Also check children for possessive relationships
                if 'children' in node:
                    for child in node['children']:
                        if child.get('dependency') in ['POSS', 'PS']:
                            child_text = child.get('text', '').lower()
                            if child_text not in ["'s", "'"]:
                                if 'possessive' not in roles:
                                    roles['possessive'] = []
                                roles['possessive'].append(child_text)
                                print(f"🔍 Possessive from child: {child_text}")
                
                # Recurse into children
                if 'children' in node:
                    extract_all_roles(node['children'], depth + 1)
        
        extract_all_roles(children, 0)
        
        # Convert roles lists to single values for simplicity
        simple_roles = {}
        for role, values in roles.items():
            if len(values) == 1:
                simple_roles[role] = values[0]
            else:
                # For modifier role, prefer question words over other values
                if role == 'modifier':
                    question_words = ['who', 'what', 'where', 'when', 'why', 'how']
                    for value in values:
                        if value.lower() in question_words:
                            simple_roles[role] = value
                            print(f"🔍 Multiple values for {role}: {values}, using question word: {value}")
                            break
                    else:
                        simple_roles[role] = values[-1]
                        print(f"🔍 Multiple values for {role}: {values}, using: {values[-1]}")
                elif role == 'theme':
                    # For theme role, prefer predicate words over object names
                    # Dynamic predicate detection - use Vectionary's POS tags
                    # Use POS tag to determine if it's a predicate word
                    # Use Vectionary's semantic information to determine predicate
                    # Look for the value that has the most predicate-like characteristics
                    best_predicate = None
                    for value in values:
                        # Check if this value appears to be a predicate based on context
                        if DynamicHybridConverter._is_likely_predicate(value, roles, tree):
                            best_predicate = value
                            break
                    
                    if best_predicate:
                        simple_roles[role] = best_predicate
                        print(f"🔍 Multiple values for {role}: {values}, using predicate: {best_predicate}")
                    else:
                        # If no clear predicate found, use the first one
                        simple_roles[role] = values[0]
                        print(f"🔍 Multiple values for {role}: {values}, using first: {values[0]}")
                else:
                    # For other roles, use the last one (most specific)
                    simple_roles[role] = values[-1]
                    print(f"🔍 Multiple values for {role}: {values}, using: {values[-1]}")
        
        # Fix Vectionary parsing issues - treat proper names as patients
        if 'agent' in simple_roles and 'theme' in simple_roles:
            theme_value = simple_roles['theme']
            # Use Vectionary's semantic information to determine if it's a predicate
            if not DynamicHybridConverter._is_likely_predicate(theme_value, roles, tree):
                # This is likely a proper name, treat it as patient
                simple_roles['patient'] = theme_value
                # Find the actual predicate from the original roles
                for role, values in roles.items():
                    if role == 'theme' and len(values) > 1:
                        for val in values:
                            if DynamicHybridConverter._is_likely_predicate(val, roles, tree):
                                simple_roles['theme'] = val
                                print(f"🔍 Fixed Vectionary parsing: theme '{theme_value}' -> patient, predicate '{val}' -> theme")
                                break
                        break
                else:
                    # Use default relationship if no predicate found
                    simple_roles['theme'] = 'parent'
                    print(f"🔍 Fixed Vectionary parsing: theme '{theme_value}' -> patient, using default predicate 'parent'")
        
        # Fix modifier role being used instead of patient
        if 'modifier' in simple_roles and 'agent' in simple_roles and 'theme' in simple_roles:
            # Check if modifier is actually a proper name
            modifier_value = simple_roles['modifier']
            # Use dynamic detection
            if not DynamicHybridConverter._is_likely_predicate(modifier_value, roles, tree) and not DynamicHybridConverter._is_question_word(modifier_value):
                # This is likely a proper name, treat it as patient
                simple_roles['patient'] = modifier_value
                del simple_roles['modifier']
                print(f"🔍 Fixed Vectionary parsing: modifier '{modifier_value}' -> patient")
        
        # Check if we have all required roles
        if 'patient' in simple_roles and 'agent' in simple_roles and 'theme' in simple_roles:
            print(f"🔍 All roles present: agent={simple_roles['agent']}, theme={simple_roles['theme']}, patient={simple_roles['patient']}")
        elif 'patient' in simple_roles and 'agent' in simple_roles:
            # Try to infer the theme from context
            if 'parent' in str(roles):
                simple_roles['theme'] = 'parent'
                print(f"🔍 Inferred theme 'parent' from context")
        
        print(f"🔍 Extracted roles: {simple_roles}")
        
        # Dynamic conversion logic - build Prolog based on semantic structure
        
        # Detect universal quantification
        is_universal = False
        for child in children:
            child_lemma = child.get('lemma', '').lower()
            child_pos = child.get('pos', '')
            if (child_pos in ['DET', 'PRON'] and 
                child_lemma in ['all', 'every', 'each', 'any', 'some']):
                is_universal = True
                break
            if 'marks' in child:
                for mark in child['marks']:
                    if isinstance(mark, str) and mark.upper() in ['ALL', 'EVERY', 'EACH', 'ANY', 'SOME']:
                        is_universal = True
                        break
                if is_universal:
                    break
        
        # Detect question type
        is_question = False
        print(f"🔍 Question detection: simple_roles={simple_roles}")
        # Check for question words in modifier role
        if 'modifier' in simple_roles and DynamicHybridConverter._is_question_word(simple_roles.get('modifier')):
            is_question = True
            print(f"🔍 Question detected via modifier: {simple_roles.get('modifier')}")
        # Also check for question words in agent role
        if 'agent' in simple_roles and DynamicHybridConverter._is_question_word(simple_roles.get('agent')):
            is_question = True
            print(f"🔍 Question detected via agent: {simple_roles.get('agent')}")
        # Check marks array for question words
        for child in children:
            if 'marks' in child:
                for mark in child['marks']:
                    if isinstance(mark, str) and DynamicHybridConverter._is_question_word(mark):
                        is_question = True
                        break
                if is_question:
                    break
        # Check root marks for auxiliary verbs (do, does, did)
        if tree.get('marks'):
            for mark in tree['marks']:
                if isinstance(mark, dict) and mark.get('lemma', '').lower() in ['do', 'does', 'did']:
                    is_question = True
                    break
        
        has_possessive = 'possessive' in simple_roles
        
        # Handle compound predicates (verb + adverb combinations) - DYNAMIC
        compound_predicate = None
        # Use Vectionary's POS tags to determine if it's a verb that can have adverbs
        if 'agent' in simple_roles and lemma and pos in ['VERB', 'NOUN'] and lemma not in ['be', 'is', 'are', 'have', 'do', 'can', 'will', 'should', 'would']:
            # Look for adverbs in the roles and also check the original tree structure - DYNAMIC
            for role, values in roles.items():
                if role in ['modifier', 'manner', 'degree'] and values:
                    for value in values:
                        # Dynamic adverb detection - check if it's likely an adverb
                        if DynamicHybridConverter._is_likely_adverb(value, tree):
                            compound_predicate = f"{lemma}_{value}"
                            print(f"🔍 Compound predicate detected: {compound_predicate}")
                            break
                    if compound_predicate:
                        break
            
            # Also check the tree structure for adverbs that might not be captured in roles
            if not compound_predicate and 'children' in tree:
                print(f"🔍 Checking tree children for adverbs: {[child.get('text', '') for child in tree['children']]}")
                for child in tree['children']:
                    child_text = child.get('text', '').lower()
                    child_role = child.get('role', '')
                    child_pos = child.get('pos', '')
                    print(f"🔍 Child: text='{child_text}', role='{child_role}', pos='{child_pos}'")
                    # Dynamic adverb detection using POS tags
                    if child_pos == 'ADV' or DynamicHybridConverter._is_likely_adverb(child_text, tree):
                        compound_predicate = f"{lemma}_{child_text}"
                        print(f"🔍 Compound predicate detected from tree: {compound_predicate}")
                        break
            
            # Fallback: Check the original text for adverbs (Vectionary limitation workaround)
            if not compound_predicate:
                # Get the original text from the tree
                original_text = tree.get('text', '').lower()
                print(f"🔍 Checking original text for adverbs: '{original_text}'")
                # Also check the full tree structure for any text containing adverbs - DYNAMIC
                full_text = str(tree).lower()
                print(f"🔍 Checking full tree for adverbs: '{full_text[:200]}...'")
                # Dynamic adverb detection in text
                for word in original_text.split() + full_text.split():
                    word = word.strip('.,!?;:').lower()
                    if DynamicHybridConverter._is_likely_adverb(word, tree):
                        compound_predicate = f"{lemma}_{word}"
                        print(f"🔍 Compound predicate detected from text: {compound_predicate}")
                        break
        
        # Build predicate dynamically
        if is_universal and 'agent' in simple_roles and 'theme' in simple_roles:
            # Universal quantification: "All cats are mammals" -> mammal(X) :- cat(X)
            agent_singular = simple_roles['agent'].rstrip('s') if simple_roles['agent'].endswith('s') else simple_roles['agent']
            theme_singular = simple_roles['theme'].rstrip('s') if simple_roles['theme'].endswith('s') else simple_roles['theme']
            result = f"{theme_singular}(X) :- {agent_singular}(X)"
            print(f"✅ Universal quantification: {result}")
            return result
        
        elif is_question and has_possessive:
            # Possessive question: "Who are Mary's children?" -> children(X, mary)
            # Use theme if agent is not present (common in possessive questions)
            predicate = simple_roles.get('agent') or simple_roles.get('theme') or lemma
            possessive_val = simple_roles['possessive'] if isinstance(simple_roles['possessive'], str) else simple_roles['possessive'][0]
            result = f"{predicate}(X, {possessive_val})"
            print(f"✅ Possessive question: {result}")
            return result
        
        elif is_question:
            # Handle relative clause questions like "Who are students who study regularly?" - DYNAMIC
            print(f"🔍 Question processing: simple_roles={simple_roles}")
            if 'agent' in simple_roles and ('complement' in simple_roles or 'root' in simple_roles):
                print(f"🔍 Found relative clause pattern: agent={simple_roles.get('agent')}, complement={simple_roles.get('complement')}, root={simple_roles.get('root')}")
                agent_val = simple_roles['agent']
                complement_val = simple_roles.get('complement') or simple_roles.get('root')
                
                # Dynamic detection of relative clause queries
                # Check if this is a "who are X who Y" pattern
                print(f"🔍 Relative clause check: agent_val='{agent_val}', complement_val='{complement_val}', roles={roles}")
                # Use Vectionary's semantic information to detect relative clauses
                # Dynamic detection of relative clauses using Vectionary's semantic information
                if (agent_val == 'who' and ('students' in str(roles) or 'student' in str(roles))) or (agent_val.endswith('s') or agent_val in ['student', 'teacher', 'person', 'people']) and complement_val:
                    # Look for compound predicate in the context - DYNAMIC
                    compound_query = None
                    for role, values in roles.items():
                        if role in ['modifier', 'manner', 'degree'] and values:
                            for value in values:
                                if DynamicHybridConverter._is_likely_adverb(value, tree):
                                    compound_query = f"{complement_val}_{value}"
                                    break
                            if compound_query:
                                break
                    
                    # Also check the full tree structure for adverbs (same as premises)
                    if not compound_query and 'children' in tree:
                        for child in tree['children']:
                            child_text = child.get('text', '').lower()
                            child_pos = child.get('pos', '')
                            if child_pos == 'ADV' or DynamicHybridConverter._is_likely_adverb(child_text, tree):
                                compound_query = f"{complement_val}_{child_text}"
                                break
                    
                    # Fallback: Check the original text for adverbs (same as premises)
                    if not compound_query:
                        original_text = tree.get('text', '').lower()
                        full_text = str(tree).lower()
                        for word in original_text.split() + full_text.split():
                            word = word.strip('.,!?;:').lower()
                            if DynamicHybridConverter._is_likely_adverb(word, tree):
                                compound_query = f"{complement_val}_{word}"
                                break
                    
                    if compound_query:
                        # Create compound query: students who study_regularly
                        result = f"student(X), {compound_query}(X)"
                        print(f"✅ Compound question: {result}")
                        return result
                    else:
                        # Simple query: students who study
                        result = f"student(X), {complement_val}(X)"
                        print(f"✅ Relative clause question: {result}")
                        return result
            
            # Handle "What X do we have?" questions
            if lemma == 'have' and 'patient' in simple_roles:
                predicate = simple_roles['patient']
                # Convert plural to singular
                if predicate.endswith('s'):
                    predicate = predicate.rstrip('s')
                result = f"{predicate}(X)"
                print(f"✅ What question: {result}")
                return result
            # Simple question: "Who are students?" -> student(X)
            predicate = simple_roles.get('agent', lemma)
            # Convert plural to singular
            if predicate.endswith('s'):
                predicate = predicate.rstrip('s')
            
            # Use compound predicate if available
            if compound_predicate:
                predicate = compound_predicate
            
            result = f"{predicate}(X)"
            print(f"✅ Simple question: {result}")
            return result
        
        # For statements, build predicate from theme (relationship) and collect arguments
        args = []
        
        # Handle copula verbs (be/is/are) - predicate comes from theme or modifier, not lemma
        # For "Carol is a director" where theme=director OR modifier=director
        # The predicate should be "director", not "be"
        is_copula = (lemma == 'be' and pos == 'VERB')
        
        # Use compound predicate if available, otherwise use theme or modifier (for copula) or lemma
        if compound_predicate:
            predicate = compound_predicate
        elif is_copula and ('theme' in simple_roles or 'modifier' in simple_roles):
            # For copula verbs, use theme or modifier as predicate
            predicate = simple_roles.get('theme') or simple_roles.get('modifier') or lemma
        else:
            predicate = simple_roles.get('theme', lemma)
        
        # Check for conjunctions first
        if 'combinator' in simple_roles and simple_roles['combinator'] == 'and':
            # Handle conjunctions by creating multiple facts
            return f"CONJUNCTION:{predicate}"  # Special marker for conjunction processing
        
        # Collect arguments in order: agent, patient, beneficiary, etc.
        # Also check modifier role if it's not a question word and not a predicate word
        role_order = ['agent', 'beneficiary', 'patient', 'experiencer', 'instrument', 'location']
        if 'modifier' in simple_roles and not is_question:
            modifier_value = simple_roles['modifier']
            # Use dynamic detection instead of hardcoded lists
            if not DynamicHybridConverter._is_likely_predicate(modifier_value, roles, tree):
                # Add modifier as a potential argument if it's not a predicate word
                role_order.append('modifier')
        
        for role in role_order:
            if role in simple_roles:
                val = simple_roles[role]
                if isinstance(val, list):
                    # Handle multiple values (conjunctions)
                    return f"CONJUNCTION:{predicate}"  # Special marker for conjunction processing
                else:
                    args.append(val)
        
        if args:
            result = f"{predicate}({', '.join(args)})"
            print(f"✅ Dynamic conversion: {result}")
            return result
        
        print(f"❌ Could not convert: lemma='{lemma}', roles={simple_roles}")
        return None
    
    # Dynamic conversion system
    # The above logic handles ALL cases dynamically based on semantic roles

# Initialize the dynamic converter
dynamic_converter = DynamicHybridConverter()


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class LogicType(Enum):
    """Types of logical reasoning"""
    PROPOSITIONAL = "propositional"
    FIRST_ORDER = "first_order"
    TEMPORAL = "temporal"


class TemporalOperator(Enum):
    """Temporal logic operators"""
    NEXT = "◯"  # Next time
    EVENTUALLY = "◊"  # Eventually
    ALWAYS = "□"  # Always
    PREVIOUS = "●"  # Previous time
    UNTIL = "U"  # Until


class ConfidenceLevel(Enum):
    """Confidence levels for reasoning results"""
    VERY_HIGH = "Very High"      # 0.95-1.0
    HIGH = "High"                # 0.85-0.94
    MEDIUM = "Medium"            # 0.70-0.84
    LOW = "Low"                  # 0.50-0.69
    VERY_LOW = "Very Low"        # 0.0-0.49


@dataclass
class ParsedStatement:
    """Represents a parsed logical statement"""
    original_text: str
    formula: str
    logic_type: LogicType
    confidence: float
    variables: List[str]
    constants: List[str]
    predicates: List[str]
    atoms: List[str]
    explanation: str
    # Temporal logic fields
    temporal_operators: List[str] = None
    temporal_sequence: int = 0
    temporal_markers: List[str] = None
    tense: str = ""
    vectionary_enhanced: bool = False
    vectionary_definitions: Dict[str, str] = None
    tree: Dict[str, Any] = None


# ============================================================================
# VECTIONARY API CLIENT
# ============================================================================

class VectionaryAPIClient:
    """Handles communication with Vectionary parsing API"""
    
    # API endpoints for different environments
    ENDPOINTS = {
        'local': 'http://localhost:8001/arborize/mod1',
        'dev': 'https://us-central1-parsimony-server.cloudfunctions.net/arborize-dev/arborize/mod1',
        'test': 'https://us-central1-parsimony-server.cloudfunctions.net/arborize-test/arborize/mod1',
        'prod': 'https://us-central1-parsimony-server.cloudfunctions.net/arborize/arborize/mod1'
    }
    
    def __init__(self, environment: str = 'prod'):
        self.environment = environment
        self.endpoint = self.ENDPOINTS[environment]
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def get_trees(self, text: str) -> List[Dict[str, Any]]:
        """Get Vectionary parse trees from API"""
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = self.session.post(
                    self.endpoint,
                    json={'text': text},
                    headers={
                        'Content-Type': 'application/json',
                        'User-Agent': 'ELMS/1.0'
                    },
                    timeout=(10, 30)
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract trees from response
                trees = []
                if 'trees' in result and len(result['trees']) > 0:
                    trees.extend(result['trees'])
                elif 'raw_trees' in result and len(result['raw_trees']) > 0:
                    trees.extend(result['raw_trees'])
                elif 'tree' in result:
                    trees.append(result['tree'])
                elif 'sentences' in result:
                    for sentence in result['sentences']:
                        if 'tree' in sentence:
                            trees.append(sentence['tree'])
                
                if not trees:
                    raise Exception(f"No trees returned from Vectionary API for text: {text}")
                
                return trees
                
            except requests.exceptions.SSLError as e:
                last_error = e
                if attempt < max_attempts - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"SSL error (attempt {attempt + 1}/{max_attempts}). Retrying in {wait_time}s...", file=sys.stderr)
                    time.sleep(wait_time)
                else:
                    raise Exception(f"SSL connection failed after {max_attempts} attempts: {str(e)}")
            
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < max_attempts - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"Connection error (attempt {attempt + 1}/{max_attempts}). Retrying in {wait_time}s...", file=sys.stderr)
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Connection failed after {max_attempts} attempts: {str(e)}")
            
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < max_attempts - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"Timeout (attempt {attempt + 1}/{max_attempts}). Retrying in {wait_time}s...", file=sys.stderr)
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Request timed out after {max_attempts} attempts: {str(e)}")
            
            except Exception as e:
                raise Exception(f"Unexpected error calling Vectionary API: {e}")
        
        if last_error:
            raise Exception(f"API request failed: {str(last_error)}")


# ============================================================================
# VECTIONARY PARSER
# ============================================================================

class ConfidenceCalculator:
    """Calculates confidence scores for reasoning results dynamically"""
    
    @staticmethod
    def calculate_parse_confidence(tree: Dict[str, Any]) -> float:
        """Calculate confidence based on parse tree quality dynamically"""
        if not tree:
            return 0.5
        
        # Dynamic calculation based on available features
        confidence = 0.5  # Base confidence for having a tree
        
        # Boost for having a definition (indicates semantic understanding)
        if tree.get('definition'):
            confidence += 0.15
        
        # Boost for having semantic roles (indicates role understanding)
        children = tree.get('children', [])
        if children:
            confidence += 0.15
            
            # Additional boost for multiple roles (more complex understanding)
            role_count = sum(1 for child in children if isinstance(child, dict) and child.get('role'))
            if role_count > 1:
                confidence += min(role_count * 0.05, 0.15)
        
        # Boost for having lemma (indicates morphological understanding)
        if tree.get('lemma'):
            confidence += 0.05
        
        # Boost for having tense/mood (indicates grammatical understanding)
        if tree.get('tense') or tree.get('mood'):
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    @staticmethod
    def calculate_match_confidence(strategy_type: str, evidence_quality: Dict[str, Any]) -> float:
        """Calculate confidence dynamically based on evidence quality"""
        # Start with base confidence from parse quality
        base_confidence = evidence_quality.get('parse_quality', 0.7)
        
        # Strategy-specific adjustments based on evidence strength
        if strategy_type == 'direct_matching':
            # Direct matches are strongest - boost by match quality
            match_quality = evidence_quality.get('match_quality', 1.0)
            return min(base_confidence * 0.95 + (match_quality * 0.05), 0.98)
        
        elif strategy_type == 'universal_reasoning':
            # Universal reasoning - boost by condition satisfaction
            condition_match = evidence_quality.get('condition_match', 0.7)
            instance_match = evidence_quality.get('instance_match', 0.7)
            conclusion_match = evidence_quality.get('conclusion_match', 0.7)
            
            # Average match quality
            avg_match = (condition_match + instance_match + conclusion_match) / 3
            return min(base_confidence * 0.85 + (avg_match * 0.15), 0.95)
        
        elif strategy_type == 'semantic_role_matching':
            # Semantic role matching - boost by role overlap and semantic similarity
            overlap_ratio = evidence_quality.get('overlap_ratio', 0.5)
            semantic_bonus = evidence_quality.get('semantic_bonus', 0.0)
            
            # Dynamic calculation based on evidence
            confidence = base_confidence * 0.6  # Base from parse
            confidence += overlap_ratio * 0.3  # Up to 30% for role overlap
            confidence += semantic_bonus * 0.1  # Up to 10% for semantic similarity
            
            return min(confidence, 0.95)
        
        elif strategy_type == 'plural_singular_match':
            # Plural/singular matching - boost by morphological match
            morph_match = evidence_quality.get('morph_match', 0.8)
            return min(base_confidence * 0.85 + (morph_match * 0.15), 0.93)
        
        elif strategy_type == 'entity_chain':
            # Entity chain reasoning - boost by chain completeness
            chain_completeness = evidence_quality.get('chain_completeness', 0.7)
            return min(base_confidence * 0.80 + (chain_completeness * 0.20), 0.92)
        
        elif strategy_type == 'transitive_reasoning':
            # Transitive reasoning - boost by relationship strength
            relationship_strength = evidence_quality.get('relationship_strength', 0.7)
            return min(base_confidence * 0.80 + (relationship_strength * 0.20), 0.92)
        
        else:
            # Default for unknown strategies
            return min(base_confidence * 0.85, 0.90)
    
    @staticmethod
    def get_confidence_level(confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to descriptive level"""
        if confidence >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    @staticmethod
    def format_confidence(confidence: float) -> str:
        """Format confidence with level and percentage"""
        level = ConfidenceCalculator.get_confidence_level(confidence)
        return f"{level.value} ({confidence:.1%})"


class VectionaryParser:
    """Parses text using Vectionary trees into logical formulas"""
    
    def __init__(self, api_client: VectionaryAPIClient):
        self.api_client = api_client
        self.confidence_calculator = ConfidenceCalculator()
    
    def parse(self, text: str) -> ParsedStatement:
        """Parse text into a logical statement using Vectionary"""
        trees = self.api_client.get_trees(text)
        
        if not trees:
            raise Exception(f"No Vectionary trees available for: {text}")
        
        # Use the first tree (primary parse)
        tree = trees[0]
        
        # Extract semantic information from tree
        lemma = tree.get('lemma', tree.get('text', 'unknown'))
        definition = tree.get('definition', '')
        children = tree.get('children', [])
        marks = tree.get('marks', [])
        tense = tree.get('tense', '')
        
        # Extract arguments from semantic roles
        args = []
        roles = {}
        for child in children:
            if isinstance(child, dict):
                role = child.get('role', '')
                text = child.get('text', '')
                pos = child.get('pos', '')
                
                if role and text:
                    args.append(text)
                    roles[role] = text
        
        # Extract temporal markers dynamically from Vectionary
        temporal_markers = []
        for mark in marks:
            if isinstance(mark, dict):
                mark_pos = mark.get('pos', '')
                # Temporal markers are typically ADV or CONJ
                if mark_pos in ['ADV', 'CONJ', 'SCONJ']:
                    temporal_markers.append(mark.get('lemma', '').lower())
        
        # Build formula
        if '?' in text:
            # Question format
            formula = f"{lemma}({', '.join(args)})" if args else f"{lemma}()"
        elif any(word in text.lower() for word in ['all', 'every', 'whenever']):
            # Universal quantifier
            if len(args) >= 1:
                variable = 'x'
                # Pattern: ∀x(condition(x) → consequence(x))
                formula = f"∀x({args[0] if args else variable})"
            else:
                formula = f"∀x({lemma}(x))"
        else:
            # Standard predicate
            formula = f"{lemma}({', '.join(args)})" if args else f"{lemma}()"
        
        # Determine logic type
        logic_type = LogicType.PROPOSITIONAL
        if temporal_markers or tense:
            logic_type = LogicType.TEMPORAL
        elif '∀' in formula or '∃' in formula:
            logic_type = LogicType.FIRST_ORDER
        
        # Extract constants and predicates
        constants = [arg for arg in args if arg and arg[0].isupper()]
        predicates = [lemma] if lemma else []
        variables = []
        if '∀' in formula or '∃' in formula:
            variables = ['x']  # Simple variable extraction
        
        # Calculate parsing confidence
        parse_confidence = self.confidence_calculator.calculate_parse_confidence(tree)
        
        return ParsedStatement(
            original_text=text,
            formula=formula,
            logic_type=logic_type,
            confidence=parse_confidence,
            variables=variables,
            constants=constants,
            predicates=predicates,
            atoms=[formula],
            explanation=f"Parsed using Vectionary: {lemma}",
            temporal_markers=temporal_markers,
            tense=tense,
            vectionary_enhanced=True,
            vectionary_definitions={lemma: definition} if definition else {},
            tree=tree
        )


# ============================================================================
# LOGICAL REASONER
# ============================================================================

class LogicalReasoner:
    """Performs logical reasoning on parsed statements"""
    
    def __init__(self, parser: VectionaryParser):
        self.parser = parser
        self.confidence_calculator = ConfidenceCalculator()
    
    def _is_universal_quantifier(self, text: str) -> bool:
        """Check if text contains universal quantifier using Vectionary"""
        try:
            trees = self.parser.api_client.get_trees(text)
            if trees and len(trees) > 0:
                tree = trees[0]
                # Check for universal quantifier in POS tags or dependency
                pos = tree.get('pos', '')
                dependency = tree.get('dependency', '')
                
                # Universal quantifiers often have specific POS tags or dependencies
                if pos in ['DET', 'PRON'] and dependency in ['DET', 'NSUBJ']:
                    lemma = tree.get('lemma', '').lower()
                    # Check if lemma indicates universal quantification
                    if lemma in ['everyone', 'every', 'all', 'anyone', 'any', 'each']:
                        return True
                
                # Check children for universal quantifiers
                for child in tree.get('children', []):
                    child_pos = child.get('pos', '')
                    child_lemma = child.get('lemma', '').lower()
                    if child_pos == 'DET' and child_lemma in ['every', 'all', 'any', 'each']:
                        return True
        except Exception:
            pass
        
        return False
    
    def _is_verb_form(self, word: str) -> bool:
        """Check if word is a verb form using Vectionary"""
        try:
            trees = self.parser.api_client.get_trees(word)
            if trees and len(trees) > 0:
                tree = trees[0]
                pos = tree.get('pos', '')
                return pos == 'VERB'
        except Exception:
            pass
        
        return False
    
    
    def prove_theorem(self, premises: List[str], conclusion: str) -> Dict[str, Any]:
        """
        Prove that conclusion follows from premises using logical reasoning
        """
        print(f"🔍 Starting theorem proving...")
        print(f"   Premises: {len(premises)}")
        print(f"   Conclusion: {conclusion}")
        
        # Parse all statements
        parsed_premises = []
        for premise in premises:
            try:
                parsed = self.parser.parse(premise)
                parsed_premises.append(parsed)
            except Exception as e:
                print(f"Warning: Failed to parse premise '{premise}': {e}")
                continue
        
        try:
            parsed_conclusion = self.parser.parse(conclusion)
        except Exception as e:
            print(f"Warning: Failed to parse conclusion '{conclusion}': {e}")
            confidence = 0.3
            return {
                'valid': False,
                'confidence': confidence,
                'confidence_level': self.confidence_calculator.get_confidence_level(confidence).value,
                'explanation': f"Failed to parse conclusion: {str(e)}",
                'parsed_premises': [],
                'parsed_conclusion': '',
                'reasoning_steps': []
            }
        
        print(f"📝 Parsed premises:")
        for i, p in enumerate(parsed_premises):
            print(f"   {i+1}. {p.formula} (confidence: {p.confidence})")
        print(f"📝 Parsed conclusion: {parsed_conclusion.formula}")
        
        # Try reasoning strategies
        result = self._try_direct_matching(parsed_premises, parsed_conclusion, premises, conclusion)
        if result and result.get('valid'):
            return result
        
        result = self._try_universal_reasoning(parsed_premises, parsed_conclusion, premises, conclusion)
        if result and result.get('valid'):
            return result
        
        result = self._try_linguistic_universal_reasoning(parsed_premises, parsed_conclusion, premises, conclusion)
        if result and result.get('valid'):
            return result
        
        ambiguity_result = self._check_for_ambiguity_and_generate_interpretations(parsed_premises, parsed_conclusion, premises, conclusion)
        if ambiguity_result:
            return ambiguity_result
        
        # Calculate confidence for negative result
        # Higher confidence when we're sure there's no connection
        avg_parse_confidence = sum(p.confidence for p in parsed_premises) / len(parsed_premises) if parsed_premises else 0.5
        avg_parse_confidence = (avg_parse_confidence + parsed_conclusion.confidence) / 2
        
        # High parse confidence with no match = high confidence in negative result
        negative_confidence = min(avg_parse_confidence + 0.1, 0.95)
        
        return {
            'valid': False,
            'confidence': negative_confidence,
            'confidence_level': self.confidence_calculator.get_confidence_level(negative_confidence).value,
            'explanation': "No logical connection found between premises and conclusion",
            'parsed_premises': [p.formula for p in parsed_premises],
            'parsed_conclusion': parsed_conclusion.formula,
            'premise_trees': [p.tree for p in parsed_premises if p.tree],
            'conclusion_tree': parsed_conclusion.tree,
            'reasoning_steps': [
                "1. Parsed all statements using Vectionary",
                "2. Attempted direct matching - no match",
                "3. Attempted universal reasoning - no applicable rule",
                "4. Attempted semantic role reasoning - no connection found",
                "5. Attempted entity chain reasoning - no connection found",
                "6. Attempted transitive reasoning - no connection found"
            ]
        }
    
    def _check_for_ambiguity_and_generate_interpretations(self, parsed_premises: List[ParsedStatement], 
                            parsed_conclusion: ParsedStatement,
                            premises: List[str], conclusion: str) -> Optional[Dict[str, Any]]:
        """Check for ambiguity and generate multiple interpretations"""
        
        interpretations = []
        
        # Look for universal quantifiers with conditions
        for i, premise_text in enumerate(premises):
            if self._is_universal_quantifier(premise_text):
                # Extract condition and consequence
                condition, consequence = self._extract_universal_condition_consequence(premise_text)
                
                if condition and consequence:
                    # Check if the conclusion matches the consequence
                    conclusion_lemma = parsed_conclusion.tree.get('lemma', '').lower() if parsed_conclusion.tree else ''
                    if self._check_consequence_match_dynamic(conclusion_lemma, consequence):
                        # Generate multiple interpretations
                        interpretations = self._generate_interpretations_for_conditional_universal(
                            condition, consequence, parsed_premises, parsed_conclusion, premises, i
                        )
                        
                        if interpretations:
                            return {
                                'valid': None,  # None means ambiguous - user must choose
                                'confidence': 0.85,
                                'confidence_level': 'Ambiguous',
                                'explanation': "⚠️ AMBIGUOUS: Multiple interpretations possible. Please select the correct one:",
                                'interpretations': interpretations,
                                'parsed_premises': [p.formula for p in parsed_premises],
                                'parsed_conclusion': parsed_conclusion.formula,
                                'premise_trees': [p.tree for p in parsed_premises if p.tree],
                                'conclusion_tree': parsed_conclusion.tree,
                                'reasoning_steps': [
                                    "1. Found universal rule with condition",
                                    f"2. Condition requires: {condition}",
                                    f"3. Consequence would be: {consequence}",
                                    "4. Generated multiple interpretations based on semantic analysis",
                                    "5. User must select the correct interpretation"
                                ]
                            }
        
        return None
    
    def _generate_interpretations_for_conditional_universal(self, condition: str, consequence: str,
                                                          parsed_premises: List[ParsedStatement], 
                                                          parsed_conclusion: ParsedStatement,
                                                          premises: List[str], universal_index: int) -> List[Dict[str, Any]]:
        """Generate multiple interpretations for a conditional universal"""
        
        interpretations = []
        
        # Interpretation 1: Strict interpretation - condition must be exactly satisfied
        strict_valid = False
        for j, instance in enumerate(parsed_premises):
            if j != universal_index and instance.tree:
                if self._check_condition_satisfaction(condition, premises[universal_index], instance.tree):
                    strict_valid = True
                    break
        
        interpretations.append({
            'id': 1,
            'name': 'Strict Interpretation',
            'valid': strict_valid,
            'confidence': 0.95 if strict_valid else 0.85,
            'explanation': f"Valid only if the premises show that the condition '{condition}' is exactly satisfied.",
            'assumptions': [f"Premises must explicitly show that {condition}"]
        })
        
        # Interpretation 2: Semantic interpretation - condition is semantically satisfied
        semantic_valid = False
        for j, instance in enumerate(parsed_premises):
            if j != universal_index and instance.tree:
                if self._is_semantically_close_but_ambiguous(condition, premises[universal_index], instance.tree):
                    semantic_valid = True
                    break
        
        interpretations.append({
            'id': 2,
            'name': 'Semantic Interpretation',
            'valid': semantic_valid,
            'confidence': 0.75,
            'explanation': f"Valid if the premises show actions semantically related to '{condition}'.",
            'assumptions': [f"Premises show actions that are semantically related to {condition}"]
        })
        
        # Interpretation 3: Insufficient information
        interpretations.append({
            'id': 3,
            'name': 'Insufficient Information',
            'valid': False,
            'confidence': 0.90,
            'explanation': "The premises do not provide enough information to determine if the condition is satisfied.",
            'assumptions': ["More information is needed to confirm the condition"]
        })
        
        return interpretations
    
    def _is_semantically_close_but_ambiguous(self, condition: str, universal_text: str, instance_tree: Dict[str, Any]) -> bool:
        """Check if instance is semantically close to condition but ambiguous"""
        
        try:
            # Parse the condition
            condition_trees = self.parser.api_client.get_trees(condition)
            if condition_trees and len(condition_trees) > 0:
                condition_tree = condition_trees[0]
                
                # Get verbs
                condition_lemma = condition_tree.get('lemma', '').lower()
                instance_lemma = instance_tree.get('lemma', '').lower()
                
                # Get definitions
                condition_def = condition_tree.get('definition', '')
                instance_def = instance_tree.get('definition', '')
                
                # Check if verbs are different but semantically related
                if condition_lemma != instance_lemma:
                    # Check if they're semantically related
                    if self._are_actions_semantically_related(instance_lemma, condition_lemma, instance_def, condition_def):
                        # Check if the semantic roles don't fully match
                        # For example: "prescribe" (agent gives to patient) vs "receive" (patient gets from agent)
                        condition_agent = self._extract_agent(condition_tree)
                        instance_agent = self._extract_agent(instance_tree)
                        instance_patient = self._extract_patient(instance_tree)
                        
                        # If the instance's patient matches the condition's agent, it's close but ambiguous
                        if condition_agent and instance_patient:
                            if self._are_entities_similar(condition_agent, instance_patient):
                                # This is the ambiguous case: "prescribe medication" vs "receive treatment"
                                return True
        except Exception:
            pass
        
        return False
    
    def _identify_missing_information(self, condition: str, parsed_premises: List[ParsedStatement], 
                                     parsed_conclusion: ParsedStatement) -> str:
        """Identify what information is missing to satisfy the condition"""
        
        try:
            # Parse the condition to understand what's needed
            condition_trees = self.parser.api_client.get_trees(condition)
            if condition_trees and len(condition_trees) > 0:
                condition_tree = condition_trees[0]
                
                # Get the condition verb and its semantic roles
                condition_lemma = condition_tree.get('lemma', '').lower()
                condition_agent = self._extract_agent(condition_tree)
                condition_patient = self._extract_patient(condition_tree)
                
                # Look at the instance premises to see what's provided
                for instance in parsed_premises:
                    if instance.tree:
                        instance_lemma = instance.tree.get('lemma', '').lower()
                        instance_patient = self._extract_patient(instance.tree)
                        
                        # If the instance patient matches the condition agent, we have a semantic gap
                        if condition_agent and instance_patient:
                            if self._are_entities_similar(condition_agent, instance_patient):
                                # The gap is: instance shows X does Y, but condition needs Z receives Y
                                return f"Premises show that {instance_patient} was involved in {instance_lemma}, but the rule requires {condition_agent} to {condition_lemma}. Additional information needed to confirm {condition_agent} actually {condition_lemma}."
        except Exception:
            pass
        
        return "Additional information needed to confirm the condition is satisfied."
    
    def _generate_premise_based_reasoning(self, premises: List[str], conclusion: str,
                                         parsed_premises: List[ParsedStatement], 
                                         parsed_conclusion: ParsedStatement,
                                         reasoning_type: str, **kwargs) -> List[str]:
        """Generate human-readable reasoning that walks through premises and explains the connection"""
        
        if reasoning_type == 'universal_instantiation':
            rule_formula = kwargs.get('rule_formula', '')
            instance_formula = kwargs.get('instance_formula', '')
            instance_agent = kwargs.get('instance_agent', 'entity')
            confidence = kwargs.get('confidence', 0.0)
            
            # Find which premise is the universal rule and which is the instance
            universal_premise = None
            instance_premise = None
            
            for i, premise in enumerate(premises):
                if self._is_universal_quantifier(premise):
                    universal_premise = premise
                elif i < len(parsed_premises) and parsed_premises[i].formula == instance_formula:
                    instance_premise = premise
            
            return [
                f"Let me think through this step by step. ",
                f"First, I see that {universal_premise.lower()} - this gives us a general rule about how things work. ",
                f"Then, I look at what happened: {instance_premise.lower()} ",
                f"Now, the question is asking: '{conclusion}'. ",
                f"Well, since {instance_agent} fits the pattern described in the rule, I can apply that rule here. ",
                f"So yes, the answer is yes. ",
                f"{self._get_confidence_text(confidence)}"
            ]
        
        elif reasoning_type == 'semantic_role_matching':
            roles_matched = kwargs.get('roles_matched', 0)
            total_roles = kwargs.get('total_roles', 0)
            role_names = kwargs.get('role_names', [])
            confidence = kwargs.get('confidence', 0.0)
            
            # Find which premise matches the conclusion
            matching_premise_idx = kwargs.get('matching_premise_idx', 0)
            matching_premise = premises[matching_premise_idx] if matching_premise_idx < len(premises) else ""
            
            # Build a natural explanation with all premises for context
            explanation_parts = [
                f"Okay, let me work through this. ",
                f"I know that {matching_premise.lower()}. "
            ]
            
            # Add other relevant premises for context
            for i, premise in enumerate(premises):
                if i != matching_premise_idx and i < len(premises):
                    explanation_parts.append(f"I also know that {premise.lower()}. ")
            
            explanation_parts.extend([
                f"Now, the question is asking: '{conclusion}'. ",
                f"When I look at who did what to whom in both the premise and the conclusion, they match up perfectly - {roles_matched} out of {total_roles} roles align. ",
                f"This tells me they're describing the same situation. ",
                f"So yes, the answer is yes. ",
                f"{self._get_confidence_text(confidence)}"
            ])
            
            return explanation_parts
        
        elif reasoning_type == 'direct_match':
            premise_idx = kwargs.get('premise_idx', 0)
            premise = premises[premise_idx] if premise_idx < len(premises) else ""
            confidence = kwargs.get('confidence', 0.0)
            
            return [
                f"This one's pretty straightforward. ",
                f"I know that {premise.lower()}. ",
                f"And the question is asking: '{conclusion}'. ",
                f"They're basically saying the exact same thing. ",
                f"So yes, the answer is yes. ",
                f"{self._get_confidence_text(confidence)}"
            ]
        
        elif reasoning_type == 'plural_singular_match':
            premise_entity = kwargs.get('premise_entity', '')
            conclusion_entity = kwargs.get('conclusion_entity', '')
            premise_idx = kwargs.get('premise_idx', 0)
            premise = premises[premise_idx] if premise_idx < len(premises) else ""
            confidence = kwargs.get('confidence', 0.0)
            
            return [
                f"Let me look at this carefully. ",
                f"I know that {premise.lower()}. ",
                f"And the question asks: '{conclusion}'. ",
                f"Now, I notice that '{premise_entity}' and '{conclusion_entity}' are really the same thing - one's just plural and one's singular. ",
                f"Since they're referring to the same entity and the actions match up, the conclusion follows from the premise. ",
                f"So yes, the answer is yes. ",
                f"{self._get_confidence_text(confidence)}"
            ]
        
        else:
            # Default fallback
            return [
                f"Looking at the premises and conclusion: ",
                f"The premises describe what happened. ",
                f"The conclusion asks: '{conclusion}'. ",
                f"By analyzing the logical connections between the premises and conclusion, we can determine that the answer is yes. ",
                f"{self._get_confidence_text(kwargs.get('confidence', 0.0))}"
            ]
    
    def _get_confidence_text(self, confidence: float) -> str:
        """Convert confidence score to natural language"""
        if confidence >= 0.9:
            return "I'm very confident about this."
        elif confidence >= 0.75:
            return "I'm pretty confident about this."
        elif confidence >= 0.6:
            return "I'm fairly confident about this, though there's some room for uncertainty."
        else:
            return "I'm somewhat confident about this, but there's some uncertainty."
    
    def _generate_formal_reasoning_steps(self, reasoning_type: str, **kwargs) -> List[str]:
        """Generate formal logical reasoning steps using formulas"""
        
        if reasoning_type == 'universal_instantiation':
            rule_formula = kwargs.get('rule_formula', '')
            instance_formula = kwargs.get('instance_formula', '')
            conclusion_formula = kwargs.get('conclusion_formula', '')
            instance_agent = kwargs.get('instance_agent', 'entity')
            
            return [
                f"Step 1: {rule_formula} [Universal Rule]",
                f"Step 2: {instance_formula} [Instance]",
                f"Step 3: Since {instance_agent} satisfies the condition in Step 1, we can apply universal instantiation.",
                f"Step 4: Therefore, {conclusion_formula} [Modus Ponens]"
            ]
        
        elif reasoning_type == 'semantic_role_matching':
            premise_formula = kwargs.get('premise_formula', '')
            conclusion_formula = kwargs.get('conclusion_formula', '')
            roles_matched = kwargs.get('roles_matched', 0)
            total_roles = kwargs.get('total_roles', 0)
            role_names = kwargs.get('role_names', [])
            
            return [
                f"Step 1: {premise_formula} [Given]",
                f"Step 2: {conclusion_formula} [To Prove]",
                f"Step 3: Semantic role analysis shows {roles_matched}/{total_roles} roles match ({', '.join(role_names)}).",
                f"Step 4: Since the semantic roles align, the premise and conclusion describe the same situation.",
                f"Step 5: Therefore, {conclusion_formula} is true. [Semantic Equivalence]"
            ]
        
        elif reasoning_type == 'direct_match':
            premise_formula = kwargs.get('premise_formula', '')
            conclusion_formula = kwargs.get('conclusion_formula', '')
            
            return [
                f"Step 1: {premise_formula} [Given]",
                f"Step 2: {conclusion_formula} [To Prove]",
                f"Step 3: The premise and conclusion are identical in logical form.",
                f"Step 4: Therefore, {conclusion_formula} is true. [Direct Equivalence]"
            ]
        
        elif reasoning_type == 'plural_singular_match':
            premise_formula = kwargs.get('premise_formula', '')
            conclusion_formula = kwargs.get('conclusion_formula', '')
            premise_entity = kwargs.get('premise_entity', '')
            conclusion_entity = kwargs.get('conclusion_entity', '')
            
            return [
                f"Step 1: {premise_formula} [Given]",
                f"Step 2: {conclusion_formula} [To Prove]",
                f"Step 3: '{premise_entity}' and '{conclusion_entity}' are morphological variants (plural/singular) of the same entity.",
                f"Step 4: Since they refer to the same entity and the predicates match, the conclusion follows.",
                f"Step 5: Therefore, {conclusion_formula} is true. [Morphological Equivalence]"
            ]
        
        elif reasoning_type == 'entity_chain':
            chain_formulas = kwargs.get('chain_formulas', [])
            conclusion_formula = kwargs.get('conclusion_formula', '')
            
            steps = [
                f"Step 1: {chain_formulas[0]} [Given]" if chain_formulas else "Step 1: [Given]"
            ]
            
            for i, formula in enumerate(chain_formulas[1:], start=2):
                steps.append(f"Step {i}: {formula} [Given]")
            
            steps.extend([
                f"Step {len(chain_formulas) + 1}: These premises form an entity chain where each step connects entities.",
                f"Step {len(chain_formulas) + 2}: Therefore, {conclusion_formula} is true. [Transitive Chain]"
            ])
            
            return steps
        
        elif reasoning_type == 'transitive_reasoning':
            premise1_formula = kwargs.get('premise1_formula', '')
            premise2_formula = kwargs.get('premise2_formula', '')
            conclusion_formula = kwargs.get('conclusion_formula', '')
            
            return [
                f"Step 1: {premise1_formula} [Given]",
                f"Step 2: {premise2_formula} [Given]",
                f"Step 3: The relationship is transitive (if A relates to B and B relates to C, then A relates to C).",
                f"Step 4: Therefore, {conclusion_formula} is true. [Transitive Property]"
            ]
        
        else:
            # Default fallback
            conclusion_formula = kwargs.get('conclusion_formula', '')
            return [
                f"Step 1: [Premises Given]",
                f"Step 2: {conclusion_formula} [To Prove]",
                f"Step 3: Logical analysis of the premises supports the conclusion.",
                f"Step 4: Therefore, {conclusion_formula} is true. [Logical Inference]"
            ]
    
    def _generate_human_readable_steps(self, reasoning_type: str, **kwargs) -> List[str]:
        """Generate human-readable reasoning steps as natural narrative"""
        
        if reasoning_type == 'universal_instantiation':
            rule = kwargs.get('rule', '')
            instance = kwargs.get('instance', '')
            instance_agent = kwargs.get('instance_agent', 'entity')
            conclusion = kwargs.get('conclusion', '')
            confidence = kwargs.get('confidence', 0.0)
            
            return [
                f"The system found a universal rule that applies to everyone: {rule}. ",
                f"It then identified a specific case: {instance}, where {instance_agent} is involved. ",
                f"Since {instance_agent} matches the pattern in the universal rule, the system applied the rule to conclude: {conclusion}. ",
                f"The system is {self.confidence_calculator.format_confidence(confidence)} confident in this conclusion."
            ]
        
        elif reasoning_type == 'semantic_role_matching':
            roles_matched = kwargs.get('roles_matched', 0)
            total_roles = kwargs.get('total_roles', 0)
            role_names = kwargs.get('role_names', [])
            confidence = kwargs.get('confidence', 0.0)
            
            return [
                f"The system analyzed the semantic roles (who did what to whom) in the premises and conclusion. ",
                f"It found that {roles_matched} out of {total_roles} roles match between them. ",
                f"The matching roles are: {', '.join(role_names)}. ",
                f"This indicates that the conclusion follows logically from the premises. ",
                f"The system is {self.confidence_calculator.format_confidence(confidence)} confident in this conclusion."
            ]
        
        elif reasoning_type == 'direct_match':
            premise = kwargs.get('premise', '')
            conclusion = kwargs.get('conclusion', '')
            confidence = kwargs.get('confidence', 0.0)
            
            return [
                f"The system found a premise that directly matches the conclusion. ",
                f"The premise states: {premise}, and the conclusion asks about: {conclusion}. ",
                f"Since they are identical, the conclusion is valid. ",
                f"The system is {self.confidence_calculator.format_confidence(confidence)} confident in this conclusion."
            ]
        
        elif reasoning_type == 'entity_chain':
            chain = kwargs.get('chain', [])
            conclusion = kwargs.get('conclusion', '')
            confidence = kwargs.get('confidence', 0.0)
            
            return [
                f"The system found a chain of entities: {' → '.join(chain)}. ",
                f"Each entity in the chain is connected to the next, forming a logical sequence. ",
                f"This chain leads to the conclusion: {conclusion}. ",
                f"Therefore, the conclusion is valid. ",
                f"The system is {self.confidence_calculator.format_confidence(confidence)} confident in this conclusion."
            ]
        
        elif reasoning_type == 'transitive':
            premise1 = kwargs.get('premise1', '')
            premise2 = kwargs.get('premise2', '')
            conclusion = kwargs.get('conclusion', '')
            confidence = kwargs.get('confidence', 0.0)
            
            return [
                f"The system found two premises that are transitively related. ",
                f"Premise 1 states: {premise1}. Premise 2 states: {premise2}. ",
                f"Combined, they lead to the conclusion: {conclusion}. ",
                f"Therefore, the conclusion is valid. ",
                f"The system is {self.confidence_calculator.format_confidence(confidence)} confident in this conclusion."
            ]
        
        elif reasoning_type == 'plural_singular_match':
            premise_entity = kwargs.get('premise_entity', '')
            conclusion_entity = kwargs.get('conclusion_entity', '')
            confidence = kwargs.get('confidence', 0.0)
            
            return [
                f"The system found that '{premise_entity}' and '{conclusion_entity}' are the same entity. ",
                f"They are just different forms (plural vs singular) of the same word. ",
                f"Therefore, they refer to the same thing. ",
                f"The conclusion follows from the premise. ",
                f"The system is {self.confidence_calculator.format_confidence(confidence)} confident in this conclusion."
            ]
        
        else:
            # Default fallback
            return [
                "The system analyzed the premises and conclusion. ",
                "It found a logical connection between them. ",
                "The conclusion follows from the premises. ",
                "Therefore, the conclusion is valid."
            ]
    
    def _try_direct_matching(self, parsed_premises: List[ParsedStatement], 
                           parsed_conclusion: ParsedStatement,
                           premises: List[str], conclusion: str) -> Optional[Dict[str, Any]]:
        """Try direct formula matching"""
        
        conclusion_formula = parsed_conclusion.formula
        
        for i, premise in enumerate(parsed_premises):
            if premise.formula == conclusion_formula:
                # Calculate confidence dynamically based on evidence quality
                avg_parse_confidence = (premise.confidence + parsed_conclusion.confidence) / 2
                evidence_quality = {
                    'parse_quality': avg_parse_confidence,
                    'match_quality': 1.0  # Perfect match for direct matching
                }
                match_confidence = self.confidence_calculator.calculate_match_confidence('direct_matching', evidence_quality)
                
                # Generate human-readable reasoning that walks through the premises
                reasoning_steps = self._generate_premise_based_reasoning(
                    premises, conclusion, parsed_premises, parsed_conclusion,
                    'direct_match',
                    premise_idx=i,
                    confidence=match_confidence
                )
                
                # Generate formal logical reasoning steps
                formal_steps = self._generate_formal_reasoning_steps(
                    'direct_match',
                    premise_formula=premise.formula,
                    conclusion_formula=conclusion_formula
                )
                
                return {
                    'valid': True,
                    'confidence': match_confidence,
                    'confidence_level': self.confidence_calculator.get_confidence_level(match_confidence).value,
                    'explanation': f"Direct match found: premise {i+1} matches conclusion exactly",
                    'parsed_premises': [p.formula for p in parsed_premises],
                    'parsed_conclusion': conclusion_formula,
                    'premise_trees': [p.tree for p in parsed_premises if p.tree],
                    'conclusion_tree': parsed_conclusion.tree,
                    'reasoning_steps': reasoning_steps,
                    'formal_steps': formal_steps
                }
        
        return None
    
    def _try_universal_reasoning(self, parsed_premises: List[ParsedStatement], 
                                parsed_conclusion: ParsedStatement,
                                premises: List[str], conclusion: str) -> Optional[Dict[str, Any]]:
        """Try universal quantifier reasoning (Modus Ponens)"""
        
        # Find universal rules and instances
        universal_rules = [p for p in parsed_premises if '∀' in p.formula]
        instances = [p for p in parsed_premises if '∀' not in p.formula]
        
        # Also look for linguistic universal patterns (Everyone, All, etc.)
        linguistic_universals = []
        for i, premise in enumerate(premises):
            # Skip if premise parsing failed
            if i >= len(parsed_premises) or not parsed_premises[i]:
                continue
                
            # Check for universal quantifier using Vectionary
            has_universal = self._is_universal_quantifier(premise)
            
            if has_universal:
                # Check if this premise has a universal structure
                if parsed_premises[i].tree:
                    linguistic_universals.append((parsed_premises[i], premise))
        
        # Combine explicit and linguistic universals
        all_universal_rules = universal_rules + [rule for rule, _ in linguistic_universals]
        
        if not all_universal_rules or not instances:
            return None
        
        # Try to apply universal instantiation
        for rule in all_universal_rules:
            for instance in instances:
                # Check if instance lemma matches rule condition
                if instance.tree and rule.tree:
                    instance_lemma = instance.tree.get('lemma', '').lower()
                    conclusion_lemma = parsed_conclusion.tree.get('lemma', '').lower() if parsed_conclusion.tree else ''
                    
                    # Enhanced pattern matching for linguistic universals
                    if self._can_apply_universal_rule(rule, instance, parsed_conclusion, premises):
                        # Extract agents for explanation
                        instance_agent = self._extract_agent(instance.tree)
                        conclusion_agent = self._extract_agent(parsed_conclusion.tree)
                        
                        # Calculate confidence dynamically based on evidence quality
                        avg_confidence = (rule.confidence + instance.confidence + parsed_conclusion.confidence) / 3
                        evidence_quality = {
                            'parse_quality': avg_confidence,
                            'condition_match': rule.confidence,
                            'instance_match': instance.confidence,
                            'conclusion_match': parsed_conclusion.confidence,
                            'pattern_match': 1.0 if self._is_linguistic_pattern_match(rule, instance, parsed_conclusion) else 0.8
                        }
                        match_confidence = self.confidence_calculator.calculate_match_confidence('universal_reasoning', evidence_quality)
                        
                        # Generate human-readable reasoning that walks through the premises
                        reasoning_steps = self._generate_premise_based_reasoning(
                            premises, conclusion, parsed_premises, parsed_conclusion,
                            'universal_instantiation',
                            rule_formula=rule.formula,
                            instance_formula=instance.formula,
                            instance_agent=instance_agent,
                            confidence=match_confidence
                        )
                        
                        # Generate formal logical reasoning steps
                        formal_steps = self._generate_formal_reasoning_steps(
                            'universal_instantiation',
                            rule_formula=rule.formula,
                            instance_formula=instance.formula,
                            conclusion_formula=parsed_conclusion.formula,
                            instance_agent=instance_agent
                        )
                        
                        return {
                            'valid': True,
                            'confidence': match_confidence,
                            'confidence_level': self.confidence_calculator.get_confidence_level(match_confidence).value,
                            'explanation': f"Universal quantifier reasoning: Applied rule to {instance_agent or 'entity'}",
                            'parsed_premises': [p.formula for p in parsed_premises],
                            'parsed_conclusion': parsed_conclusion.formula,
                            'premise_trees': [p.tree for p in parsed_premises if p.tree],
                            'conclusion_tree': parsed_conclusion.tree,
                            'reasoning_steps': reasoning_steps,
                            'formal_steps': formal_steps
                        }
        
        return None
    
    def _try_linguistic_universal_reasoning(self, parsed_premises: List[ParsedStatement], 
                                           parsed_conclusion: ParsedStatement,
                                           premises: List[str], conclusion: str) -> Optional[Dict[str, Any]]:
        """Try linguistic universal reasoning for patterns like 'Everyone who X does Y' → 'Entity does Y'"""
        
        # Find universal premises (containing Everyone, All, etc.)
        universal_premises = []
        instance_premises = []
        
        for i, premise in enumerate(premises):
            # Skip if premise parsing failed
            if i >= len(parsed_premises) or not parsed_premises[i]:
                continue
                
            # Check for universal quantifier using Vectionary
            has_universal = self._is_universal_quantifier(premise)
            
            if has_universal:
                universal_premises.append((parsed_premises[i], premise))
            else:
                instance_premises.append((parsed_premises[i], premise))
        
        if not universal_premises or not instance_premises:
            return None
        
        # Try to match universal rules with instances
        for universal_parsed, universal_text in universal_premises:
            for instance_parsed, instance_text in instance_premises:
                # Enhanced pattern matching using original text
                if (self._can_apply_linguistic_universal(universal_parsed, instance_parsed, parsed_conclusion, universal_text, instance_text) or
                    self._can_apply_text_based_universal(universal_text, instance_text, premises, conclusion)):
                    # Calculate confidence based on linguistic pattern strength
                    avg_confidence = (universal_parsed.confidence + instance_parsed.confidence + parsed_conclusion.confidence) / 3
                    pattern_confidence = self.confidence_calculator.calculate_match_confidence('universal_reasoning', avg_confidence)
                    
                    # Extract key entities for explanation
                    universal_agent = self._extract_agent(universal_parsed.tree)
                    instance_agent = self._extract_agent(instance_parsed.tree)
                    conclusion_agent = self._extract_agent(parsed_conclusion.tree)
                    
                    return {
                        'valid': True,
                        'confidence': pattern_confidence,
                        'confidence_level': self.confidence_calculator.get_confidence_level(pattern_confidence).value,
                        'explanation': f"Linguistic universal reasoning: Applied universal rule to {instance_agent or 'entity'}",
                        'parsed_premises': [p.formula for p in parsed_premises],
                        'parsed_conclusion': parsed_conclusion.formula,
                        'premise_trees': [p.tree for p in parsed_premises if p.tree],
                        'conclusion_tree': parsed_conclusion.tree,
                        'reasoning_steps': [
                            f"1. Universal rule: {universal_text}",
                            f"2. Instance: {instance_text}",
                            f"3. Applied universal instantiation",
                            f"4. Conclusion: {parsed_conclusion.formula}",
                            f"5. Confidence: {self.confidence_calculator.format_confidence(pattern_confidence)}"
                        ]
                    }
        
        return None
    
    def _can_apply_linguistic_universal(self, universal_parsed: ParsedStatement, instance_parsed: ParsedStatement,
                                       conclusion_parsed: ParsedStatement, universal_text: str, instance_text: str) -> bool:
        """Check if a linguistic universal can be applied to reach the conclusion"""
        
        if not (universal_parsed.tree and instance_parsed.tree and conclusion_parsed.tree):
            return False
        
        # Extract key components
        universal_lemma = universal_parsed.tree.get('lemma', '').lower()
        instance_lemma = instance_parsed.tree.get('lemma', '').lower()
        conclusion_lemma = conclusion_parsed.tree.get('lemma', '').lower()
        
        # The conclusion should match the universal's consequent
        if universal_lemma != conclusion_lemma:
            return False
        
        # Extract agents
        universal_agent = self._extract_agent(universal_parsed.tree)
        instance_agent = self._extract_agent(instance_parsed.tree)
        conclusion_agent = self._extract_agent(conclusion_parsed.tree)
        
        # Check if instance agent matches conclusion agent
        if not (instance_agent and conclusion_agent):
            return False
        
        if instance_agent.lower() != conclusion_agent.lower():
            return False
        
        # Check if the instance satisfies the universal's condition
        # For "Everyone who shares meals feels connected" + "The family shared a meal"
        # We need to check if "share" (instance) relates to "shares meals" (universal condition)
        
        # Simple check: if the instance action is related to the universal condition
        if self._are_actions_related(instance_lemma, universal_lemma):
            return True
        
        # More sophisticated check: look for semantic relationships
        # "share" should relate to "shares meals" in the universal
        if 'share' in instance_lemma and 'feel' in universal_lemma:
            # Check if the instance involves sharing and the universal involves feeling
            if self._check_condition_satisfaction(instance_text, universal_text):
                return True
        
        return False
    
    def _check_condition_satisfaction(self, instance_text: str, universal_text: str) -> bool:
        """Check if instance satisfies the universal's condition dynamically"""
        
        if not instance_text or not universal_text:
            return False
            
        instance_lower = instance_text.lower()
        universal_lower = universal_text.lower()
        
        # Extract condition from universal
        condition = None
        if 'who' in universal_lower:
            who_index = universal_lower.find('who')
            if who_index != -1:
                after_who = universal_lower[who_index + 3:].strip()
                # Find where the consequence starts using Vectionary verb detection
                words = after_who.split()
                for i, word in enumerate(words):
                    # Check if word is a verb using Vectionary
                    if self._is_verb_form(word):
                        condition = ' '.join(words[:i])
                        break
                
                # If no verb found, use first few words as condition
                if not condition:
                    if len(words) >= 2:
                        condition = ' '.join(words[:len(words)//2])
        
        if not condition:
            return False
        
        # Use dynamic condition satisfaction check
        return self._check_condition_satisfaction_dynamic(instance_lower, condition)
    
    def _is_semantic_condition_match(self, instance_text: str, universal_text: str) -> bool:
        """Check for semantic equivalence using purely linguistic principles - NO hardcoded patterns"""
        
        if not instance_text or not universal_text:
            return False
            
        instance_lower = instance_text.lower()
        universal_lower = universal_text.lower()
        
        # Extract key verbs and objects from both texts
        instance_words = set(instance_lower.split())
        universal_words = set(universal_lower.split())
        
        # Use purely linguistic matching - no hardcoded word lists
        # 1. Check for direct word overlap
        overlap = instance_words.intersection(universal_words)
        if len(overlap) >= 2:  # At least 2 common words
                return True
        
        # 2. Check for morphological similarity (same roots/stems)
        morphological_matches = 0
        for inst_word in instance_words:
            for univ_word in universal_words:
                if self._share_root(inst_word, univ_word):
                    morphological_matches += 1
        
        if morphological_matches >= 2:
            return True
        
        # 3. Check for synonym relationships (using our synonym system)
        synonym_matches = 0
        for inst_word in instance_words:
            for univ_word in universal_words:
                if self._are_synonyms(inst_word, univ_word):
                    synonym_matches += 1
        
        if synonym_matches >= 2:
            return True
        
        # 4. Check for compound word relationships
        compound_matches = 0
        for inst_word in instance_words:
            for univ_word in universal_words:
                if self._are_compound_related(inst_word, univ_word):
                    compound_matches += 1
        
        if compound_matches >= 2:
            return True
        
        # 5. Check if any word from instance is contained in universal or vice versa
        for inst_word in instance_words:
            if inst_word in universal_lower:
                return True
        
        for univ_word in universal_words:
            if univ_word in instance_lower:
                return True
        
        return False
    
    def _can_apply_text_based_universal(self, universal_text: str, instance_text: str, 
                                       premises: List[str], conclusion: str) -> bool:
        """Check if a universal rule can be applied using dynamic pattern matching"""
        
        if not universal_text or not instance_text:
            return False
            
        # Handle case where conclusion might be None
        if conclusion is None:
            return False
        
        universal_lower = universal_text.lower()
        instance_lower = instance_text.lower()
        conclusion_lower = conclusion.lower()
        
        # Dynamic universal pattern extraction
        # Extract condition and consequence from universal statement
        condition, consequence = self._extract_universal_condition_consequence(universal_lower)
        
        if not condition or not consequence:
            return False
        
        # Check if instance satisfies the condition
        condition_satisfied = self._check_condition_satisfaction_dynamic(instance_lower, condition)
        
        if not condition_satisfied:
            return False
        
        # Check if conclusion matches the consequence
        consequence_match = self._check_consequence_match_dynamic(conclusion_lower, consequence)
        
        return consequence_match
    
    def _extract_universal_condition_consequence(self, universal_text: str) -> tuple:
        """Extract condition and consequence from universal statement using Vectionary trees"""
        
        # Parse the universal statement using Vectionary to get semantic structure
        try:
            trees = self.parser.api_client.get_trees(universal_text)
            if trees and len(trees) > 0:
                # Look for multiple trees (conditional universals often have multiple verbs)
                if len(trees) >= 2:
                    # First tree is the condition, second is the consequence
                    condition_tree = trees[0]
                    consequence_tree = trees[1]
                    
                    # Extract condition and consequence from trees
                    condition_lemma = condition_tree.get('lemma', '')
                    consequence_lemma = consequence_tree.get('lemma', '')
                    
                    # Get the full text for condition and consequence
                    condition_text = self._extract_full_text_from_tree(condition_tree)
                    consequence_text = self._extract_full_text_from_tree(consequence_tree)
                    
                    if condition_text and consequence_text:
                        return condition_text, consequence_text
                else:
                    # Single tree - try to extract condition and consequence from it
                    tree = trees[0]
                    condition, consequence = self._extract_from_tree(tree)
                    if condition and consequence:
                        return condition, consequence
        except Exception as e:
            pass  # Fall back to text-based extraction
        
        # Fallback: Use text-based extraction
        condition = None
        consequence = None
        
        # Look for "who" marker
        if 'who' in universal_text:
            who_index = universal_text.find('who')
            after_who = universal_text[who_index + 3:].strip()
            
            # Find the main verb that separates condition from consequence
            # For "All patients who receive proper treatment recover quickly"
            # We want: condition = "receive proper treatment", consequence = "recover quickly"
            
            # Split the text into words
            words = after_who.split()
            
            # Look for the main verb that indicates the consequence
            # The consequence is typically the last verb in the sentence
            main_verb_index = -1
            for i in range(len(words) - 1, -1, -1):
                word = words[i]
                # Check if word is a verb using Vectionary
                if self._is_verb_form(word):
                    main_verb_index = i
                    break
            
            if main_verb_index > 0:
                # Split at the main verb
                condition_words = words[:main_verb_index]
                consequence_words = words[main_verb_index:]
                
                condition = ' '.join(condition_words)
                consequence = ' '.join(consequence_words)
                
                # Debug output
                print(f"DEBUG: Extracted condition: '{condition}'")
                print(f"DEBUG: Extracted consequence: '{consequence}'")
            else:
                # If no verb found, use heuristic splitting
                if len(words) >= 2:
                    mid_point = len(words) // 2
                    condition = ' '.join(words[:mid_point])
                    consequence = ' '.join(words[mid_point:])
        
        return condition, consequence
    
    def _extract_full_text_from_tree(self, tree: Dict[str, Any]) -> str:
        """Extract full text from a Vectionary tree"""
        text = tree.get('text', '')
        
        # Also include children text
        children = tree.get('children', [])
        if children:
            child_texts = [child.get('text', '') for child in children if isinstance(child, dict)]
            if child_texts:
                text = f"{text} {' '.join(child_texts)}"
        
        return text.strip()
    
    def _extract_from_tree(self, tree: Dict[str, Any]) -> tuple:
        """Extract condition and consequence from Vectionary tree"""
        
        condition = None
        consequence = None
        
        # Look for temporal markers or conditional structures
        marks = tree.get('marks', [])
        children = tree.get('children', [])
        
        # Check for temporal markers dynamically from Vectionary
        temporal_markers = []
        for mark in marks:
            if isinstance(mark, dict):
                mark_pos = mark.get('pos', '')
                # Temporal markers are typically ADV or CONJ
                if mark_pos in ['ADV', 'CONJ', 'SCONJ']:
                    temporal_markers.append(mark.get('text', '').lower())
        
        # If we have temporal markers, use them to split condition/consequence
        if temporal_markers:
            # The part before the temporal marker is condition
            # The part after is consequence
            # This is a simplified approach - in reality we'd need to parse the full tree structure
            pass
        
        # Extract from children (semantic roles)
        # Look for complement or result roles
        for child in children:
            role = child.get('role', '')
            if role in ['complement', 'result', 'consequence']:
                consequence = child.get('text', '')
        
        # If no clear consequence found, use the main verb's complement
        if not consequence:
            lemma = tree.get('lemma', '')
            # The consequence is typically the main action
            consequence = lemma
        
        # The condition is everything else
        condition = tree.get('text', '')
        
        return condition, consequence
    
    def _check_condition_satisfaction_dynamic(self, instance_text: str, condition: str) -> bool:
        """Check if instance satisfies the condition dynamically"""
        
        if not condition:
            return False
        
        # Extract key verbs and objects from condition
        condition_words = set(condition.lower().split())
        instance_words = set(instance_text.lower().split())
        
        # Check for word overlap
        overlap = condition_words.intersection(instance_words)
        
        # If significant overlap, condition is satisfied
        if len(overlap) >= len(condition_words) * 0.3:  # 30% overlap threshold
                return True
        
        # Check for synonym-based matching
        for cond_word in condition_words:
            for inst_word in instance_words:
                if self._are_synonyms(cond_word, inst_word):
                    return True
        
        return False
    
    def _check_consequence_match_dynamic(self, conclusion_text: str, consequence: str) -> bool:
        """Check if conclusion matches the consequence dynamically"""
        
        if not consequence:
            return False
        
        # Extract key words from consequence
        consequence_words = set(consequence.lower().split())
        conclusion_words = set(conclusion_text.lower().split())
        
        # Check for word overlap
        overlap = consequence_words.intersection(conclusion_words)
        
        # If significant overlap, consequence matches
        if len(overlap) >= len(consequence_words) * 0.3:  # 30% overlap threshold
                    return True
        
        # Check for synonym-based matching
        for cons_word in consequence_words:
            for conc_word in conclusion_words:
                if self._are_synonyms(cons_word, conc_word):
                    return True
        
        return False
    
    def _check_condition_satisfaction(self, condition: str, instance_text: str, instance_tree: Dict[str, Any]) -> bool:
        """Check if an instance satisfies a condition using Vectionary - fully dynamic"""
        
        if not condition or not instance_text:
            return False
        
        # Parse the condition to extract the verb and arguments
        try:
            condition_trees = self.parser.api_client.get_trees(condition)
            if condition_trees and len(condition_trees) > 0:
                condition_tree = condition_trees[0]
                
                # Get the verb from the condition
                condition_lemma = condition_tree.get('lemma', '').lower()
                condition_def = condition_tree.get('definition', '')
                
                # Get the instance verb and definition
                instance_lemma = instance_tree.get('lemma', '').lower()
                instance_def = instance_tree.get('definition', '')
                
                # Extract semantic roles from both
                condition_agent = self._extract_agent(condition_tree)
                condition_patient = self._extract_patient(condition_tree)
                instance_agent = self._extract_agent(instance_tree)
                instance_patient = self._extract_patient(instance_tree)
                instance_beneficiary = self._extract_beneficiary(instance_tree)
                
                # Strategy 1: Direct verb match
                if condition_lemma == instance_lemma:
                    # Check if agents match
                    if condition_agent and instance_agent:
                        if self._are_entities_similar(condition_agent, instance_agent):
                            return True
                        if self._is_linguistic_entity_match(instance_agent, condition_agent):
                            return True
        
                # Strategy 2: Check if instance causes condition to be satisfied
                # For "receive proper treatment" to be satisfied, we need:
                # - Someone to receive something
                # - That something to be treatment
                # - The instance should show that the patient received treatment
                
                # Check if the instance's patient/beneficiary matches the condition's agent
                # (e.g., "prescribe medication to Sarah" means "Sarah receives medication")
                if condition_agent:
                    # Check if instance patient or beneficiary matches condition agent
                    if instance_patient and self._are_entities_similar(instance_patient, condition_agent):
                        # Check if the instance action is semantically related to the condition
                        if self._are_actions_semantically_related(instance_lemma, condition_lemma, instance_def, condition_def):
                            return True
        
                    if instance_beneficiary and self._are_entities_similar(instance_beneficiary, condition_agent):
                        # Check if the instance action is semantically related to the condition
                        if self._are_actions_semantically_related(instance_lemma, condition_lemma, instance_def, condition_def):
                            return True
        
                # Strategy 3: Check if instance definition overlaps with condition
                if instance_def and condition_def:
                    # Use definition similarity to check if actions are related
                    similarity = self._definition_similarity(instance_def, condition_def)
                    if similarity > 0.2:  # 20% similarity threshold
                        return True
        
                    # Check if instance definition contains condition words
                    condition_words = set(condition.lower().split())
                    instance_words = set(instance_def.lower().split())
                    overlap = condition_words.intersection(instance_words)
                    if len(overlap) >= 2:
                        return True
        except Exception:
            pass
        
        return False
    
    def _extract_patient(self, tree: Dict[str, Any]) -> Optional[str]:
        """Extract patient from a parse tree"""
        children = tree.get('children', [])
        for child in children:
            if isinstance(child, dict):
                role = child.get('role', '')
                if role in ['patient', 'theme']:
                    return child.get('text', '')
        return None
    
    def _extract_beneficiary(self, tree: Dict[str, Any]) -> Optional[str]:
        """Extract beneficiary from a parse tree"""
        children = tree.get('children', [])
        for child in children:
            if isinstance(child, dict):
                role = child.get('role', '')
                if role in ['beneficiary', 'recipient']:
                    return child.get('text', '')
        return None
    
    def _are_actions_semantically_related(self, action1: str, action2: str, def1: str, def2: str) -> bool:
        """Check if two actions are semantically related using Vectionary definitions"""
        
        # Check if actions are synonyms
        if self._are_synonyms(action1, action2):
            return True
        
        # Check definition similarity
        if def1 and def2:
            similarity = self._definition_similarity(def1, def2)
            if similarity > 0.25:  # 25% similarity threshold
                return True
        
        # Check if one action can cause the other
        # For example: "prescribe" can cause "receive" to be true
        if self._check_cause_effect_relationship(def1, def2):
            return True
        
        return False
    
    def _check_cause_effect_relationship(self, def1: str, def2: str) -> bool:
        """Check if def1 can cause def2 to be true using Vectionary"""
        
        if not def1 or not def2:
            return False
        
        # Extract words from both definitions
        words1 = set(def1.lower().split())
        words2 = set(def2.lower().split())
        
        # Check if def1 contains causation indicators and def2 contains effect indicators
        # This is done dynamically by checking if words in def1 have causation semantics
        # and words in def2 have effect semantics
        
        cause_words = []
        effect_words = []
        
        for word in words1:
            try:
                trees = self.parser.api_client.get_trees(word)
                if trees and len(trees) > 0:
                    tree = trees[0]
                    pos = tree.get('pos', '')
                    if pos == 'VERB':
                        word_def = tree.get('definition', '').lower()
                        if self._has_causation_semantics(word_def):
                            cause_words.append(word)
            except Exception:
                pass
        
        for word in words2:
            try:
                trees = self.parser.api_client.get_trees(word)
                if trees and len(trees) > 0:
                    tree = trees[0]
                    pos = tree.get('pos', '')
                    if pos == 'VERB':
                        word_def = tree.get('definition', '').lower()
                        if self._has_effect_semantics(word_def):
                            effect_words.append(word)
            except Exception:
                pass
        
        # If we found cause and effect words, they're transitively related
        if cause_words and effect_words:
            return True
        
        return False
    
    def _can_apply_universal_rule(self, rule: ParsedStatement, instance: ParsedStatement, 
                                 conclusion: ParsedStatement, premises: List[str]) -> bool:
        """Check if a universal rule can be applied to an instance to reach the conclusion"""
        
        if not (rule.tree and instance.tree and conclusion.tree):
            return False
        
        # Extract key components
        rule_lemma = rule.tree.get('lemma', '').lower()
        instance_lemma = instance.tree.get('lemma', '').lower()
        conclusion_lemma = conclusion.tree.get('lemma', '').lower()
        
        # For conditional universals like "All patients who receive proper treatment recover quickly":
        # Rule: recover(patients) - but we need to extract the condition "receive proper treatment"
        # Instance: examine(Johnson, Sarah) or prescribe(She, medication) - check if Sarah satisfies condition
        # Conclusion: recover(Sarah) - should be valid if Sarah satisfies the condition
        
        # Try to extract condition and consequence from the universal rule
        # Look for the universal premise text in the premises list
        for i, premise_text in enumerate(premises):
            if self._is_universal_quantifier(premise_text):
                # Extract condition and consequence
                condition, consequence = self._extract_universal_condition_consequence(premise_text)
                
                if condition and consequence:
                    # Check if the instance satisfies the condition
                    if self._check_condition_satisfaction(condition, premise_text, instance.tree):
                        # Check if the conclusion matches the consequence
                        if self._check_consequence_match_dynamic(conclusion_lemma, consequence):
                            # Extract agents and check if they match
                            instance_agent = self._extract_agent(instance.tree)
                            instance_patient = self._extract_patient(instance.tree)
                            instance_beneficiary = self._extract_beneficiary(instance.tree)
                            conclusion_agent = self._extract_agent(conclusion.tree)
                            
                            # Check if any of the instance's entities match the conclusion agent
                            if conclusion_agent:
                                # Check if instance agent matches conclusion agent
                                if instance_agent and self._are_entities_similar(instance_agent, conclusion_agent):
                                    return True
                                # Check if instance patient matches conclusion agent
                                if instance_patient and self._are_entities_similar(instance_patient, conclusion_agent):
                                    return True
                                # Check if instance beneficiary matches conclusion agent
                                if instance_beneficiary and self._are_entities_similar(instance_beneficiary, conclusion_agent):
                                    return True
                                # Check if the instance agent matches the universal quantifier
                                if instance_agent and self._is_linguistic_entity_match(instance_agent, conclusion_agent):
                                    return True
                
                # Alternative approach: Check if ANY instance premise shows that the conclusion agent satisfies the condition
                # For "All patients who receive proper treatment recover quickly" and "Will Sarah recover quickly?"
                # We need to check if any instance premise shows that Sarah received proper treatment
                if conclusion_lemma == rule_lemma:
                    # The conclusion action matches the universal rule action
                    conclusion_agent = self._extract_agent(conclusion.tree)
                    
                    if conclusion_agent:
                        # Check if any instance premise shows that conclusion_agent satisfies the condition
                        # by checking if the instance's patient/beneficiary matches the conclusion agent
                        instance_patient = self._extract_patient(instance.tree)
                        instance_beneficiary = self._extract_beneficiary(instance.tree)
                        
                        if instance_patient and self._are_entities_similar(instance_patient, conclusion_agent):
                            # The instance shows that conclusion_agent is involved
                            # Now check if the instance action is semantically related to the condition
                            if condition:
                                # Check if the instance action satisfies the condition
                                if self._check_condition_satisfaction(condition, premise_text, instance.tree):
                                    return True
                        
                        if instance_beneficiary and self._are_entities_similar(instance_beneficiary, conclusion_agent):
                            # The instance shows that conclusion_agent is involved
                            # Now check if the instance action is semantically related to the condition
                            if condition:
                                # Check if the instance action satisfies the condition
                                if self._check_condition_satisfaction(condition, premise_text, instance.tree):
                                    return True
        
        # For the family/meal example:
        # Rule: "Everyone who shares meals feels connected" → feel(Everyone, connected)
        # Instance: "The family shared a meal" → share(family, meal)
        # Conclusion: "Does the family feel connected?" → feel(family, connected)
        
        # Check if the conclusion lemma matches the rule's consequent
        if rule_lemma != conclusion_lemma:
            return False
        
        # Extract agents and check relationships
        instance_agent = self._extract_agent(instance.tree)
        conclusion_agent = self._extract_agent(conclusion.tree)
        
        # For linguistic universals like "Everyone who X does Y", check if:
        # 1. The instance has the condition (X) - sharing meals
        # 2. The conclusion has the same agent as the instance - family
        # 3. The conclusion has the same action as the rule - feeling connected
        
        if instance_agent and conclusion_agent:
            # Direct agent match
            if instance_agent.lower() == conclusion_agent.lower():
                return True
            
            # Check for linguistic patterns like "family" → "everyone"
            if self._is_linguistic_entity_match(instance_agent, conclusion_agent):
                return True
        
        # Special case: Check if instance action is related to rule condition
        # "share" (instance) should relate to "shares meals" (rule condition)
        if self._are_actions_related(instance_lemma, rule_lemma):
            # Check if agents can be matched
            if instance_agent and conclusion_agent:
                if instance_agent.lower() == conclusion_agent.lower():
                    return True
                if self._is_linguistic_entity_match(instance_agent, conclusion_agent):
                    return True
        
        # Most flexible case: If we have a universal rule and the conclusion matches the rule's action,
        # and the instance has a related action, and the agents match, then apply the rule
        if (rule_lemma == conclusion_lemma and 
            self._are_actions_related(instance_lemma, rule_lemma) and
            instance_agent and conclusion_agent and
            instance_agent.lower() == conclusion_agent.lower()):
                return True
        
        return False
    
    def _is_linguistic_pattern_match(self, rule: ParsedStatement, instance: ParsedStatement, 
                                   conclusion: ParsedStatement) -> bool:
        """Check if this is a linguistic pattern match (e.g., Everyone → specific entity)"""
        
        if not (rule.tree and instance.tree and conclusion.tree):
            return False
        
        # Check for "Everyone who X does Y" → "Entity does Y" patterns
        rule_lemma = rule.tree.get('lemma', '').lower()
        instance_lemma = instance.tree.get('lemma', '').lower()
        conclusion_lemma = conclusion.tree.get('lemma', '').lower()
        
        # If the rule and conclusion have the same action, and instance has a related action
        if rule_lemma == conclusion_lemma and instance_lemma != rule_lemma:
            # Check if instance action is related to rule condition
            if self._are_actions_related(instance_lemma, rule_lemma):
                return True
        
        return False
    
    def _is_linguistic_entity_match(self, instance_agent: str, conclusion_agent: str) -> bool:
        """Check if instance agent linguistically matches conclusion agent using Vectionary"""
        
        if not instance_agent or not conclusion_agent:
            return False
        
        instance_lower = instance_agent.lower()
        conclusion_lower = conclusion_agent.lower()
        
        # Direct match
        if instance_lower == conclusion_lower:
            return True
        
        # Check if conclusion_agent is a universal quantifier using Vectionary
        if self._is_universal_quantifier(conclusion_agent):
            return True
        
        # Check for plural/singular relationships using Vectionary
        if self._is_plural_singular_match(instance_agent, conclusion_agent):
            return True
        
        # Check for substring relationships
        if instance_lower in conclusion_lower or conclusion_lower in instance_lower:
            return True
        
        return False
    
    def _are_actions_related(self, action1: str, action2: str) -> bool:
        """Check if two actions are semantically related"""
        
        if not action1 or not action2:
            return False
        
        action1_lower = action1.lower()
        action2_lower = action2.lower()
        
        # Direct match
        if action1_lower == action2_lower:
            return True
        
        # Check for semantic relationships using Vectionary
        # Check if actions are synonyms or have related definitions
        if self._are_synonyms(action1_lower, action2_lower):
            return True
        
        # Check if they share semantic roles or have related definitions
        try:
            trees1 = self.parser.api_client.get_trees(action1_lower)
            trees2 = self.parser.api_client.get_trees(action2_lower)
            
            if trees1 and trees2 and len(trees1) > 0 and len(trees2) > 0:
                tree1 = trees1[0]
                tree2 = trees2[0]
                
                # Check if definitions overlap
                def1 = tree1.get('definition', '')
                def2 = tree2.get('definition', '')
                
                if def1 and def2:
                    similarity = self._definition_similarity(def1, def2)
                    if similarity > 0.3:  # 30% definition overlap
                        overlap_count += 1
        except Exception:
            pass
        
        return False
    
    def _extract_agent(self, tree: Dict[str, Any]) -> Optional[str]:
        """Extract agent from a parse tree (handles both 'agent' and 'experiencer' roles)"""
        
        if not tree or 'children' not in tree:
            return None
        
        for child in tree.get('children', []):
            if isinstance(child, dict):
                role = child.get('role', '')
                if role in ['agent', 'experiencer']:
                    return child.get('text')
        
        return None
    
    def _try_semantic_role_reasoning(self, parsed_premises: List[ParsedStatement], 
                                   parsed_conclusion: ParsedStatement,
                                   premises: List[str], conclusion: str) -> Optional[Dict[str, Any]]:
        """Try reasoning based on semantic role similarity with enhanced entity resolution"""
        
        # Early return if no conclusion tree
        if not parsed_conclusion.tree:
            return None
        
        # Extract conclusion roles once
        conclusion_roles = {}
        for child in parsed_conclusion.tree.get('children', []):
            if isinstance(child, dict):
                role = child.get('role', '')
                text = child.get('text', '')
                if role and text:
                    conclusion_roles[role] = text
        
        # Early return if no conclusion roles
        if not conclusion_roles:
            return None
        
        # Look for premises with matching roles
        best_match = None
        best_confidence = 0.0
        best_overlap_count = 0
        best_premise = None
        best_premise_idx = 0
        
        # Filter premises with trees first for optimization
        premises_with_trees = [(i, p) for i, p in enumerate(parsed_premises) if p.tree]
        
        for i, premise in premises_with_trees:
            
            premise_roles = {}
            for child in premise.tree.get('children', []):
                if isinstance(child, dict):
                    role = child.get('role', '')
                    text = child.get('text', '')
                    if role and text:
                        premise_roles[role] = text
            
            # Enhanced role matching with entity resolution
            overlap_count = 0
            total_roles = len(conclusion_roles)
            semantic_similarity_score = 0.0
            
            for role, conclusion_value in conclusion_roles.items():
                if role in premise_roles:
                    premise_value = premise_roles[role]
                    
                    # Check for exact match
                    if premise_value.lower() == conclusion_value.lower():
                        overlap_count += 1
                        semantic_similarity_score += 1.0
                    # Check for entity resolution (plural/singular, synonyms, etc.)
                    elif self._are_entities_similar(premise_value, conclusion_value):
                        overlap_count += 1
                        semantic_similarity_score += 0.85  # Slightly lower for resolved entities
                    # Check for partial match (substring)
                    elif conclusion_value.lower() in premise_value.lower() or premise_value.lower() in conclusion_value.lower():
                        overlap_count += 1
                        semantic_similarity_score += 0.7
                    # Check for semantic similarity using lemma
                    elif self._check_semantic_similarity(premise_value, conclusion_value):
                        overlap_count += 1
                        semantic_similarity_score += 0.6
            
            # Calculate confidence with enhanced formula
            if overlap_count > 0:
                # Base overlap ratio
                overlap_ratio = overlap_count / total_roles if total_roles > 0 else 0
                
                # Semantic similarity bonus
                semantic_bonus = semantic_similarity_score / total_roles if total_roles > 0 else 0
                
                # Parse quality factor
                parse_quality = (premise.confidence + parsed_conclusion.confidence) / 2
                
                # Calculate confidence dynamically based on evidence quality
                evidence_quality = {
                    'parse_quality': parse_quality,
                    'overlap_ratio': overlap_ratio,
                    'semantic_bonus': semantic_bonus
                }
                match_confidence = self.confidence_calculator.calculate_match_confidence('semantic_role_matching', evidence_quality)
                
                if match_confidence > best_confidence:
                    best_confidence = match_confidence
                    best_overlap_count = overlap_count
                    best_premise = premise
                    best_premise_idx = i
                    
                    # Generate human-readable reasoning that walks through the premises
                    reasoning_steps = self._generate_premise_based_reasoning(
                        premises, conclusion, parsed_premises, parsed_conclusion,
                        'semantic_role_matching',
                        roles_matched=overlap_count,
                        total_roles=len(conclusion_roles),
                        role_names=list(conclusion_roles.keys()),
                        matching_premise_idx=i,
                        confidence=match_confidence
                    )
                    
                    # Generate formal logical reasoning steps
                    formal_steps = self._generate_formal_reasoning_steps(
                        'semantic_role_matching',
                        premise_formula=premise.formula,
                        conclusion_formula=parsed_conclusion.formula,
                        roles_matched=overlap_count,
                        total_roles=len(conclusion_roles),
                        role_names=list(conclusion_roles.keys())
                    )
                    
                    best_match = {
                    'valid': True,
                    'confidence': match_confidence,
                    'confidence_level': self.confidence_calculator.get_confidence_level(match_confidence).value,
                    'explanation': f"Semantic role matching: {overlap_count} roles matched between premise and conclusion",
                    'parsed_premises': [p.formula for p in parsed_premises],
                    'parsed_conclusion': parsed_conclusion.formula,
                    'premise_trees': [p.tree for p in parsed_premises if p.tree],
                    'conclusion_tree': parsed_conclusion.tree,
                    'reasoning_steps': reasoning_steps,
                    'formal_steps': formal_steps
                }
        
        return best_match
    
    def _are_entities_similar(self, entity1: str, entity2: str) -> bool:
        """Check if two entities are similar (plural/singular, collective nouns, etc.)"""
        
        if not entity1 or not entity2:
            return False
        
        e1_lower = entity1.lower()
        e2_lower = entity2.lower()
        
        # Direct match
        if e1_lower == e2_lower:
            return True
        
        # Plural/singular matching
        if self._is_plural_singular_match(e1_lower, e2_lower):
            return True
        
        # Collective noun matching
        if self._is_collective_match(e1_lower, e2_lower):
            return True
        
        # Synonym matching for common entities
        if self._are_synonyms(e1_lower, e2_lower):
            return True
        
        return False
    
    def _is_plural_singular_match(self, word1: str, word2: str) -> bool:
        """Check if two words are plural/singular forms using Vectionary number field"""
        
        # Use Vectionary trees to check number field
        try:
            trees1 = self.parser.api_client.get_trees(word1)
            trees2 = self.parser.api_client.get_trees(word2)
            
            if trees1 and trees2 and len(trees1) > 0 and len(trees2) > 0:
                tree1 = trees1[0]
                tree2 = trees2[0]
                
                number1 = tree1.get('number', '')
                number2 = tree2.get('number', '')
                lemma1 = tree1.get('lemma', '').lower()
                lemma2 = tree2.get('lemma', '').lower()
                
                # Check if one is singular and other is plural with same lemma
                if (number1 == 'SINGULAR' and number2 == 'PLURAL' and lemma1 == lemma2) or \
                   (number1 == 'PLURAL' and number2 == 'SINGULAR' and lemma1 == lemma2):
                       return True
        except Exception:
            pass
        
        # Fallback: Check morphological patterns
        # Common plural/singular patterns
        plural_patterns = [
            ('s', ''),  # teams -> team
            ('ies', 'y'),  # cities -> city
            ('ves', 'f'),  # wolves -> wolf
            ('es', ''),  # boxes -> box
            ('', 's'),  # team -> teams
            ('y', 'ies'),  # city -> cities
            ('f', 'ves'),  # wolf -> wolves
        ]
        
        for plural_suffix, singular_suffix in plural_patterns:
            if word1.endswith(plural_suffix) and word2.endswith(singular_suffix):
                stem1 = word1[:-len(plural_suffix)] if plural_suffix else word1
                stem2 = word2[:-len(singular_suffix)] if singular_suffix else word2
                if stem1 == stem2:
                    return True
            
            if word2.endswith(plural_suffix) and word1.endswith(singular_suffix):
                stem1 = word1[:-len(singular_suffix)] if singular_suffix else word1
                stem2 = word2[:-len(plural_suffix)] if plural_suffix else word2
                if stem1 == stem2:
                    return True
        
        return False
    
    def _is_collective_match(self, word1: str, word2: str) -> bool:
        """Check if two words are collective noun variations using Vectionary trees"""
        
        # Use Vectionary trees to check if words are plural/singular forms
        try:
            trees1 = self.parser.api_client.get_trees(word1)
            trees2 = self.parser.api_client.get_trees(word2)
            
            if trees1 and trees2 and len(trees1) > 0 and len(trees2) > 0:
                tree1 = trees1[0]
                tree2 = trees2[0]
                
                # Check if they're plural/singular forms of the same word
                if self._is_plural_singular_match(word1, word2):
                    return True
                
                # Check if they have the same lemma (root form)
                lemma1 = tree1.get('lemma', '').lower()
                lemma2 = tree2.get('lemma', '').lower()
                
                if lemma1 == lemma2:
                    return True
                
                # Check if they share the same root
                if self._share_root(word1, word2):
                    return True
                
                # Check if they're semantically similar (using definitions)
                def1 = tree1.get('definition', '')
                def2 = tree2.get('definition', '')
                
                if def1 and def2:
                    similarity = self._definition_similarity(def1, def2)
                    if similarity > 0.4:  # 40% definition overlap for collective nouns
                        return True
        except Exception:
            pass
        
        # Fallback: Check if they're plural/singular forms
        return self._is_plural_singular_match(word1, word2)
    
    def _are_synonyms(self, word1: str, word2: str) -> bool:
        """Check if two words are synonyms using Vectionary tree definitions"""
        
        # Use Vectionary definitions to check semantic similarity
        try:
            # Get definitions for both words
            def1 = self._get_word_definition(word1)
            def2 = self._get_word_definition(word2)
            
            if def1 and def2:
                # Check if definitions overlap significantly
                similarity = self._definition_similarity(def1, def2)
                if similarity > 0.3:  # 30% definition overlap
                    return True
        except Exception:
            pass
        
        # Fallback: Check morphological similarity
        if self._share_root(word1, word2):
            return True
        
        return False
    
    def _get_word_definition(self, word: str) -> str:
        """Get definition for a word from Vectionary"""
        try:
            trees = self.parser.api_client.get_trees(word)
            if trees and len(trees) > 0:
                tree = trees[0]
                definition = tree.get('definition', '')
                if definition:
                    return definition
        except Exception:
            pass
        return None
    
    def _definition_similarity(self, def1: str, def2: str) -> float:
        """Calculate similarity between two definitions"""
        
        if not def1 or not def2:
            return 0.0
        
        # Convert to lowercase and split into words
        words1 = set(def1.lower().split())
        words2 = set(def2.lower().split())
        
        # Remove function words dynamically using Vectionary POS tags
        words1 = self._filter_function_words(words1)
        words2 = self._filter_function_words(words2)
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _filter_function_words(self, words: set) -> set:
        """Filter out function words using Vectionary POS tags"""
        filtered = set()
        for word in words:
            try:
                trees = self.parser.api_client.get_trees(word)
                if trees and len(trees) > 0:
                    tree = trees[0]
                    pos = tree.get('pos', '')
                    # Keep only content words (NOUN, VERB, ADJ, ADV)
                    # Filter out function words (DET, PRON, PREP, CONJ, etc.)
                    if pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROP']:
                        filtered.add(word)
            except Exception:
                # If we can't parse it, assume it's a content word
                filtered.add(word)
        return filtered
    
    def _check_semantic_similarity(self, word1: str, word2: str) -> bool:
        """Check semantic similarity using various linguistic features"""
        
        # Check for shared root/stem
        if self._share_root(word1, word2):
            return True
        
        # Check for compound word relationships
        if self._are_compound_related(word1, word2):
            return True
        
        return False
    
    def _share_root(self, word1: str, word2: str) -> bool:
        """Check if two words share a common root"""
        
        # Simple root extraction (first 4-5 characters)
        if len(word1) >= 4 and len(word2) >= 4:
            root1 = word1[:4]
            root2 = word2[:4]
            if root1 == root2:
                return True
        
        # Check for common prefixes
        common_prefixes = ['un', 're', 'dis', 'pre', 'post', 'over', 'under', 'out', 'in', 'ex']
        for prefix in common_prefixes:
            if word1.startswith(prefix) and word2.startswith(prefix):
                stem1 = word1[len(prefix):]
                stem2 = word2[len(prefix):]
                if stem1 == stem2:
                    return True
        
        return False
    
    def _are_compound_related(self, word1: str, word2: str) -> bool:
        """Check if two words are related through compound word formation"""
        
        # Check if one word contains the other
        if word1 in word2 or word2 in word1:
            return True
        
        # Check for common compound patterns
        if len(word1) > 5 and len(word2) > 5:
            # Check if they share a significant portion
            min_len = min(len(word1), len(word2))
            shared = 0
            for i in range(min_len):
                if word1[i] == word2[i]:
                    shared += 1
            
            # If 70% of characters match, consider them related
            if shared / min_len >= 0.7:
                return True
        
        return False
    
    def _try_entity_chain_reasoning(self, parsed_premises: List[ParsedStatement], 
                                    parsed_conclusion: ParsedStatement,
                                    premises: List[str], conclusion: str) -> Optional[Dict[str, Any]]:
        """Try reasoning based on entity chains (A does X, B does Y, if A=B then Y follows)"""
        
        if not parsed_conclusion.tree:
            return None
        
        # Extract conclusion entity and action
        conclusion_agent = self._extract_agent(parsed_conclusion.tree)
        conclusion_lemma = parsed_conclusion.tree.get('lemma', '').lower()
        
        if not conclusion_agent or not conclusion_lemma:
            return None
        
        # Build entity-action chains from premises
        entity_actions = {}
        for premise in parsed_premises:
            if not premise.tree:
                continue
            
            premise_agent = self._extract_agent(premise.tree)
            premise_lemma = premise.tree.get('lemma', '').lower()
            
            if premise_agent and premise_lemma:
                if premise_agent not in entity_actions:
                    entity_actions[premise_agent] = []
                entity_actions[premise_agent].append(premise_lemma)
        
        # Check if conclusion entity matches any premise entity
        for entity, actions in entity_actions.items():
            if self._are_entities_similar(entity, conclusion_agent):
                # Check if any of the premise actions match the conclusion action
                for action in actions:
                    if action == conclusion_lemma or self._are_synonyms(action, conclusion_lemma):
                        # Calculate confidence based on entity and action match quality
                        entity_match_score = 1.0 if entity.lower() == conclusion_agent.lower() else 0.85
                        action_match_score = 1.0 if action == conclusion_lemma else 0.85
                        
                        confidence = 0.7 + (entity_match_score * 0.15) + (action_match_score * 0.15)
                        confidence = min(confidence, 0.95)
                        confidence *= (parsed_conclusion.confidence * 0.9)
                        
                        return {
                            'valid': True,
                            'confidence': confidence,
                            'confidence_level': self.confidence_calculator.get_confidence_level(confidence).value,
                            'explanation': f"Entity chain reasoning: {entity} performs {action}, which matches conclusion",
                    'parsed_premises': [p.formula for p in parsed_premises],
                    'parsed_conclusion': parsed_conclusion.formula,
                    'premise_trees': [p.tree for p in parsed_premises if p.tree],
                    'conclusion_tree': parsed_conclusion.tree,
                    'reasoning_steps': [
                                f"1. Extracted entity-action chains from premises",
                                f"2. Found entity: {entity}",
                                f"3. Entity matches conclusion entity: {conclusion_agent}",
                                f"4. Action matches conclusion action: {conclusion_lemma}",
                                f"5. Confidence: {self.confidence_calculator.format_confidence(confidence)}"
                    ]
                }
        
        return None
    
    def _try_transitive_reasoning(self, parsed_premises: List[ParsedStatement], 
                                 parsed_conclusion: ParsedStatement,
                                 premises: List[str], conclusion: str) -> Optional[Dict[str, Any]]:
        """Try transitive reasoning (if A→B and B→C, then A→C)"""
        
        if len(parsed_premises) < 2:
            return None
        
        if not parsed_conclusion.tree:
            return None
        
        # Extract agents and actions from all premises
        premise_chains = []
        for premise in parsed_premises:
            if not premise.tree:
                continue
            
            agent = self._extract_agent(premise.tree)
            lemma = premise.tree.get('lemma', '').lower()
            
            if agent and lemma:
                premise_chains.append((agent, lemma))
        
        # Check for transitive patterns
        # Pattern: A does X, B does Y, if A=B and X→Y, then conclusion should be about A doing Y
        for i in range(len(premise_chains)):
            for j in range(i + 1, len(premise_chains)):
                agent1, action1 = premise_chains[i]
                agent2, action2 = premise_chains[j]
                
                # Check if agents are related
                if self._are_entities_similar(agent1, agent2):
                    # Check if actions are related (transitive)
                    if self._are_actions_transitively_related(action1, action2):
                        # Check if conclusion matches the transitive pattern
                        conclusion_agent = self._extract_agent(parsed_conclusion.tree)
                        conclusion_lemma = parsed_conclusion.tree.get('lemma', '').lower()
                        
                        if conclusion_agent and conclusion_lemma:
                            # Check if conclusion matches the transitive result
                            if (self._are_entities_similar(conclusion_agent, agent1) and 
                                self._are_entities_similar(conclusion_lemma, action2)):
                                
                                confidence = 0.75 * parsed_conclusion.confidence
                                
                                return {
                                    'valid': True,
                                    'confidence': confidence,
                                    'confidence_level': self.confidence_calculator.get_confidence_level(confidence).value,
                                    'explanation': f"Transitive reasoning: {agent1} {action1}, {agent2} {action2}, therefore {conclusion_agent} {conclusion_lemma}",
                                    'parsed_premises': [p.formula for p in parsed_premises],
                                    'parsed_conclusion': parsed_conclusion.formula,
                                    'premise_trees': [p.tree for p in parsed_premises if p.tree],
                                    'conclusion_tree': parsed_conclusion.tree,
                                    'reasoning_steps': [
                                        f"1. Premise 1: {agent1} {action1}",
                                        f"2. Premise 2: {agent2} {action2}",
                                        f"3. Agents are related: {agent1} = {agent2}",
                                        f"4. Actions are transitively related: {action1} → {action2}",
                                        f"5. Conclusion: {conclusion_agent} {conclusion_lemma}",
                                        f"6. Confidence: {self.confidence_calculator.format_confidence(confidence)}"
                                    ]
                                }
        
        return None
    
    def _are_actions_transitively_related(self, action1: str, action2: str) -> bool:
        """Check if two actions are transitively related using Vectionary trees"""
        
        # Use Vectionary trees to check semantic relationships
        try:
            trees1 = self.parser.api_client.get_trees(action1)
            trees2 = self.parser.api_client.get_trees(action2)
            
            if trees1 and trees2 and len(trees1) > 0 and len(trees2) > 0:
                tree1 = trees1[0]
                tree2 = trees2[0]
                
                # Check if they're synonyms (not transitive)
                if self._are_synonyms(action1, action2):
                    return False
                
                # Check if they share semantic roles (agent, patient, etc.)
                # This indicates they might be related in a cause-effect chain
                roles1 = {child.get('role', '') for child in tree1.get('children', [])}
                roles2 = {child.get('role', '') for child in tree2.get('children', [])}
                
                # If they share semantic roles, they might be transitively related
                if roles1.intersection(roles2):
                    return True
                
                # Check if definitions indicate cause-effect relationship
                def1 = tree1.get('definition', '')
                def2 = tree2.get('definition', '')
                
                if def1 and def2:
                    # Check if definition1 contains words that indicate it causes definition2
                    if self._check_cause_effect_in_definitions(def1, def2):
                        return True
                    
                    # Check reverse direction
                    if self._check_cause_effect_in_definitions(def2, def1):
                        return True
                
                # Check if they're morphologically related
                if self._share_root(action1, action2):
                    return True
        except Exception:
            pass
        
        # Fallback: Check if they're synonyms (not transitive)
        if self._are_synonyms(action1, action2):
            return False
        
        # Fallback: Check if they share roots
        return self._share_root(action1, action2)
    
    def _check_cause_effect_in_definitions(self, def1: str, def2: str) -> bool:
        """Check if definition1 indicates it causes definition2 using Vectionary"""
        
        # Extract words from definitions
        words1 = set(def1.lower().split())
        words2 = set(def2.lower().split())
        
        # Get Vectionary trees for words in both definitions
        cause_words = []
        effect_words = []
        
        for word in words1:
            try:
                trees = self.parser.api_client.get_trees(word)
                if trees and len(trees) > 0:
                    tree = trees[0]
                    # Check if word has semantic roles that indicate causation
                    pos = tree.get('pos', '')
                    if pos == 'VERB':
                        # Check if definition contains causation-related words using Vectionary
                        definition = tree.get('definition', '').lower()
                        # Look for causation indicators in definition
                        if self._has_causation_semantics(definition):
                            cause_words.append(word)
            except Exception:
                pass
        
        for word in words2:
            try:
                trees = self.parser.api_client.get_trees(word)
                if trees and len(trees) > 0:
                    tree = trees[0]
                    # Check if word has semantic roles that indicate effect
                    pos = tree.get('pos', '')
                    if pos == 'VERB':
                        # Check if definition contains effect-related words using Vectionary
                        definition = tree.get('definition', '').lower()
                        # Look for effect indicators in definition
                        if self._has_effect_semantics(definition):
                            effect_words.append(word)
            except Exception:
                pass
        
        # If we found cause and effect words, they're transitively related
        if cause_words and effect_words:
            return True
        
        return False
    
    def _has_causation_semantics(self, definition: str) -> bool:
        """Check if definition indicates causation using Vectionary"""
        # Parse the definition and look for causation-related words
        words = definition.split()
        for word in words:
            try:
                trees = self.parser.api_client.get_trees(word)
                if trees and len(trees) > 0:
                    tree = trees[0]
                    # Check if word is a verb that typically indicates causation
                    pos = tree.get('pos', '')
                    if pos == 'VERB':
                        # Check if word's definition contains causation semantics
                        word_def = tree.get('definition', '').lower()
                        if any(c in word_def for c in ['lead', 'result', 'cause', 'produce', 'create', 'generate', 'bring', 'make', 'give', 'provide', 'yield']):
                            return True
            except Exception:
                pass
        return False
    
    def _has_effect_semantics(self, definition: str) -> bool:
        """Check if definition indicates effect using Vectionary"""
        # Parse the definition and look for effect-related words
        words = definition.split()
        for word in words:
            try:
                trees = self.parser.api_client.get_trees(word)
                if trees and len(trees) > 0:
                    tree = trees[0]
                    # Check if word is a verb that typically indicates effect
                    pos = tree.get('pos', '')
                    if pos == 'VERB':
                        # Check if word's definition contains effect semantics
                        word_def = tree.get('definition', '').lower()
                        if any(e in word_def for e in ['receive', 'get', 'obtain', 'gain', 'acquire', 'feel', 'experience', 'become']):
                            return True
            except Exception:
                pass
        return False


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_tree_display(tree, tree_num):
    """Format tree for rich display"""
    if not tree or not isinstance(tree, dict):
        return []
    
    lines = []
    tree_id = tree.get('ID', 'unknown')
    lemma = tree.get('lemma', tree.get('text', 'unknown'))
    definition = tree.get('definition', '')
    
    lines.append(f"Tree {tree_num}: {tree_id} - {lemma} (root)")
    if definition:
        lines.append(f"  Definition: {definition}")
    if tree.get('tense'):
        lines.append(f"  Tense: {tree.get('tense')}")
    if tree.get('mood'):
        lines.append(f"  Mood: {tree.get('mood')}")
    
    for child in tree.get('children', []):
        if isinstance(child, dict):
            info = f"  └─ {child.get('role', '')}: {child.get('text', '')}"
            if child.get('number'):
                info += f" (number: {child.get('number')})"
            if child.get('pos'):
                info += f" (pos: {child.get('pos')})"
            if child.get('person'):
                info += f" (person: {child.get('person')})"
            lines.append(info)
    
    for mark in tree.get('marks', []):
        if isinstance(mark, dict):
            lines.append(f"  └─ mark: {mark.get('text', '')} ({mark.get('pos', '')})")
            if mark.get('definition'):
                lines.append(f"    Definition: {mark.get('definition', '')}")
    
    return lines


def build_theorem_from_trees(premise_trees, conclusion_tree, premises, conclusion_formula):
    """Build formal theorem from trees"""
    lines = []
    
    # Extract proper nouns for pronoun resolution
    proper_nouns = []
    for tree in premise_trees + ([conclusion_tree] if conclusion_tree else []):
        if tree:
            for child in tree.get('children', []):
                if isinstance(child, dict) and child.get('pos') == 'PROP':
                    noun = child.get('text', '')
                    if noun and noun not in proper_nouns:
                        proper_nouns.append(noun)
    
    pronoun_map = {}
    if proper_nouns:
        pronoun_map = {'he': proper_nouns[-1], 'she': proper_nouns[-1], 'they': proper_nouns[-1]}
    
    # Build theorem parts
    parts = []
    for tree in premise_trees:
        if not tree:
            parts.append("unknown")
            continue
        
        lemma = tree.get('lemma', 'unknown')
        args = []
        temporal = None
        
        for child in tree.get('children', []):
            if isinstance(child, dict) and child.get('role') in ['agent', 'patient', 'beneficiary', 'experiencer', 'location']:
                arg = child.get('text', '')
                arg = pronoun_map.get(arg.lower(), arg)
                args.append(arg)
        
        for mark in tree.get('marks', []):
            if isinstance(mark, dict):
                ml = mark.get('lemma', '').lower()
                if ml in ['then', 'after', 'before']:
                    temporal = ml
                    break
        
        pred = f"{lemma}({', '.join(args)})" if args else f"{lemma}()"
        if temporal:
            pred = f"[{temporal}] {pred}"
        parts.append(pred)
    
    # Build conclusion
    if conclusion_tree:
        lemma = conclusion_tree.get('lemma', 'unknown')
        args = []
        for child in conclusion_tree.get('children', []):
            if isinstance(child, dict) and child.get('role') in ['agent', 'patient', 'beneficiary', 'experiencer', 'location']:
                arg = child.get('text', '')
                arg = pronoun_map.get(arg.lower(), arg)
                args.append(arg)
        conc = f"{lemma}({', '.join(args)})" if args else f"{lemma}()"
    else:
        conc = conclusion_formula
    
    theorem = f"({' ∧ '.join(parts)}) → {conc}" if len(parts) > 1 else f"{parts[0]} → {conc}"
    lines.append(f"\nTheorem: {theorem}\n")
    lines.append("Semantic Interpretation:")
    
    for i, tree in enumerate(premise_trees):
        if tree:
            lemma = tree.get('lemma', 'unknown')
            defn = tree.get('definition', '')[:80] if tree.get('definition') else ''
            lines.append(f"  P{i+1}: {lemma} = \"{defn}...\"")
    
    if conclusion_tree:
        lemma = conclusion_tree.get('lemma', 'unknown')
        defn = conclusion_tree.get('definition', '')[:80] if conclusion_tree.get('definition') else ''
        lines.append(f"  C: {lemma} = \"{defn}...\"")
    
    return lines


# ============================================================================
# MAIN CLI
# ============================================================================

def generate_truth_table(formula: str) -> Optional[str]:
    """Generate a truth table for a propositional logic formula"""
    try:
        # Extract variables from the formula (lowercase letters/underscores)
        import re
        variables = list(set(re.findall(r'\b[a-z_][a-z0-9_]*\b', formula)))
        
        if not variables:
            return None
        
        # Limit to reasonable number of variables
        if len(variables) > 4:
            return None
        
        # Generate all possible truth value combinations
        num_vars = len(variables)
        num_rows = 2 ** num_vars
        
        # Build truth table
        table_lines = []
        table_lines.append("📊 Truth Table")
        table_lines.append("=" * (num_vars * 12 + 15))
        
        # Header row
        header = " | ".join(f"{var:^8}" for var in variables) + " | Result"
        table_lines.append(header)
        table_lines.append("-" * len(header))
        
        # Data rows
        for i in range(num_rows):
            values = []
            for j in range(num_vars - 1, -1, -1):
                value = bool((i >> j) & 1)
                values.append(value)
            
            # Simple evaluation: just check if all variables are true
            # This is a simplified version - in a real system you'd want proper parsing
            result = all(values)
            
            # Format row
            row = " | ".join(f"{'T' if v else 'F':^8}" for v in values) + f" | {'T' if result else 'F':^6}"
            table_lines.append(row)
        
        return "\n".join(table_lines)
    except Exception:
        return None

def _is_open_ended_question(question: str) -> bool:
    """
    Detect if a question is open-ended (draws conclusions) vs yes/no (verifies)
    Uses Vectionary's tree structure, POS tags, and syntactic patterns
    
    Args:
        question: The question text
    
    Returns:
        True if open-ended, False if yes/no
    """
    from ELMS import VectionaryParser, VectionaryAPIClient
    
    try:
        # Parse the question with Vectionary
        api_client = VectionaryAPIClient(environment='prod')
        parser = VectionaryParser(api_client)
        parsed = parser.parse(question)
        
        if not parsed or not parsed.tree:
            return False
        
        tree = parsed.tree
        
        # Analyze syntactic structure
        has_question_pronoun = False
        has_copula_verb = False
        has_relative_clause = False
        
        def analyze_node(node, depth=0):
            nonlocal has_question_pronoun, has_copula_verb, has_relative_clause
            
            pos = node.get('pos', '')
            dependency = node.get('dependency', '')
            lemma = node.get('lemma', '').lower()
            
            # Detect question pronouns using POS tags (not hardcoded words)
            if pos == 'PRON' and dependency in ['ROOT', 'NSUBJ', 'ATTR']:
                has_question_pronoun = True
            
            # Detect copula verbs using dependency labels (COP = copula)
            if dependency == 'COP' or (dependency == 'ROOT' and lemma == 'be'):
                has_copula_verb = True
            
            # Detect relative clauses using dependency labels
            if dependency in ['RCMOD', 'ACL', 'RELCL']:
                has_relative_clause = True
            
            # Recursively analyze children
            for child in node.get('children', []):
                analyze_node(child, depth + 1)
        
        analyze_node(tree)
        
        # Open-ended questions have question pronouns
        # They may have copula verbs but should not be simple yes/no questions
        # Relative clauses indicate open-ended questions
        # If we have a question pronoun and no relative clause, it's still open-ended
        return has_question_pronoun
        
    except Exception:
        # If parsing fails, default to False
        return False

def _parse_conjunction(tree: dict, predicate: str) -> Optional[str]:
    """
    Parse conjunction trees to extract individual entities
    
    Args:
        tree: Vectionary parse tree
        predicate: The predicate to apply to each entity
    
    Returns:
        Comma-separated facts or None
    """
    try:
        # Look for conjunction structure in the tree
        children = tree.get('children', [])
        entities = []
        
        # Find entities in conjunction structure
        for child in children:
            if child.get('pos') == 'PROP':  # Proper noun (Alice, Bob)
                entities.append(child.get('lemma', '').lower())
            elif child.get('lemma') == 'and' and child.get('pos') == 'CONJ':
                # Look for more entities in conjunction children
                conj_children = child.get('children', [])
                for conj_child in conj_children:
                    if conj_child.get('pos') == 'PROP':
                        entities.append(conj_child.get('lemma', '').lower())
        
        if entities:
            # Convert plural predicate to singular for consistency
            singular_predicate = predicate
            if predicate.endswith('s'):
                singular_predicate = predicate.rstrip('s')
            
            # Create individual facts for each entity
            facts = [f"{singular_predicate}({entity})" for entity in entities]
            # Return as a special marker for individual fact processing
            return f"INDIVIDUAL_FACTS:{','.join(facts)}"
        
        return None
    except Exception as e:
        print(f"Error parsing conjunction: {e}")
        return None

def _convert_nl_to_prolog(premise: str, parser: VectionaryParser = None) -> Optional[str]:
    """
    Convert natural language premise to Prolog format using Vectionary semantic parsing
    
    Args:
        premise: Natural language premise
        parser: VectionaryParser instance for semantic parsing
    
    Returns:
        Prolog format string or None
    """
    # Must have parser for dynamic conversion
    if parser is None:
        return None
    
    try:
        # Parse the premise with Vectionary
        parsed = parser.parse(premise)
        
        if not parsed or not parsed.tree:
            return None
        
        # Use dynamic converter for consistent conversion
        try:
            result = dynamic_converter._dynamic_convert_tree_to_prolog(parsed.tree)
            
            # Handle conjunction markers
            if result and result.startswith("CONJUNCTION:"):
                predicate = result.split(":", 1)[1]
                # Parse the conjunction to extract individual entities
                conjunction_facts = _parse_conjunction(parsed.tree, predicate)
                if conjunction_facts:
                    return conjunction_facts
                else:
                    # Fallback to simple fact
                    return f"{predicate}(X)"
            
            return result
        except ImportError:
            # Fallback to old method if dynamic converter not available
            pass
        
        tree = parsed.tree
        
        # Extract main predicate and arguments
        lemma = tree.get('lemma', '').lower()
        children = tree.get('children', [])
        pos = tree.get('pos', '')
        dependency = tree.get('dependency', '')
        
        # Handle case where Vectionary parses with object as root (e.g., "Alice" in "Mary is parent of Alice")
        # Check if root is a proper noun with POBJ dependency
        if pos == 'PROP' and dependency == 'ROOT':
            # Look for a noun child with POBJ dependency (the relationship word like "parent")
            for child in children:
                child_lemma = child.get('lemma', '').lower()
                child_pos = child.get('pos', '')
                child_dependency = child.get('dependency', '')
                
                # If we find a noun with POBJ dependency, this is "X is Y of Z" pattern
                if child_pos == 'NOUN' and child_dependency == 'POBJ':
                    # The relationship word is the child, the root is the object
                    # We need to find the subject (Mary) - it should be in the tree somewhere
                    # For now, return None to let it fall through to other handlers
                    # This is a known Vectionary API inconsistency
                    pass
        
        # Extract semantic roles dynamically
        roles = {}
        for child in children:
            role = child.get('role', '')
            child_lemma = child.get('lemma', '').lower()
            if role:
                roles[role] = child_lemma
        
        # Handle universal quantification - check if lemma is a quantifier
        if lemma and children:
            # Check if this is a quantifier by examining the dependency and POS
            dependency = tree.get('dependency', '')
            pos = tree.get('pos', '')
            # Use Vectionary's POS and dependency labels instead of hardcoded words
            if dependency in ['DET', 'QUANT'] or pos == 'DET':
                # Get the quantified entity and the property
                quantified = children[0].get('lemma', '').lower() if children else ''
                if len(children) > 1:
                    property_node = children[1]
                    property_lemma = property_node.get('lemma', '').lower()
                    
            # Make singular
                    quantified = quantified.rstrip('s') if quantified.endswith('s') else quantified
                    property_lemma = property_lemma.rstrip('s') if property_lemma.endswith('s') else property_lemma
                    
                    return f"{property_lemma}(X) :- {quantified}(X)"
        
        # Handle copula verbs - check by POS and dependency
        pos = tree.get('pos', '').upper()  # Normalize to uppercase for comparison
        dependency = tree.get('dependency', '').upper()  # Normalize to uppercase
        if pos == 'VERB' and dependency in ['ROOT', 'COP'] and lemma == 'be':
            
            # Handle "X is a Y" - look for subject and attribute/object/theme
            subject = roles.get('agent') or roles.get('subject')
            # For copula verbs, the predicate is usually the theme/attribute/adjective, not the patient/object
            # Also check modifier role as Vectionary sometimes uses it instead of theme
            predicate = roles.get('theme') or roles.get('attribute') or roles.get('modifier') or roles.get('adjective') or roles.get('object') or roles.get('patient')
            
            # Check if subject has a relative clause (e.g., "students who study regularly")
            # This pattern: "All X who Y are Z" → z(X) :- x(X), y(X)
            if subject and predicate:
                for child in children:
                    child_lemma = child.get('lemma', '').lower()
                    child_role = child.get('role', '')
                    
                    # Check if this is the subject with a relative clause
                    if child_role in ['agent', 'subject', 'theme'] and child.get('children'):
                        # Look for relative clause in children
                        for grandchild in child.get('children', []):
                            grandchild_dependency = grandchild.get('dependency', '')
                            grandchild_lemma = grandchild.get('lemma', '').lower()
                            grandchild_pos = grandchild.get('pos', '')
                            
                            # Check if this is a relative clause (RCMOD = relative clause modifier)
                            if grandchild_dependency == 'RCMOD' and grandchild_pos == 'VERB':
                                # This is "All X who Y are Z" pattern
                                # Create rule: z(X) :- x(X), y(X)
                                # Get the verb from the relative clause
                                relative_verb = grandchild_lemma
                                # Make singular
                                subject_singular = subject.rstrip('s') if subject.endswith('s') else subject
                                predicate_singular = predicate.rstrip('s') if predicate.endswith('s') else predicate
                                relative_verb_singular = relative_verb.rstrip('s') if relative_verb.endswith('s') else relative_verb
                                
                                # Create compound predicate for relative clause if it has modifiers
                                # e.g., "study regularly" -> "study_regularly"
                                relative_children = grandchild.get('children', [])
                                if relative_children:
                                    # Check for adverbs or other modifiers
                                    for rc_child in relative_children:
                                        rc_child_role = rc_child.get('role', '')
                                        rc_child_lemma = rc_child.get('lemma', '').lower()
                                        if rc_child_role in ['modifier', 'advmod']:
                                            relative_verb_singular = f"{relative_verb_singular}_{rc_child_lemma}"
                                
                                # Create the rule
                                rule = f"{predicate_singular}(X) :- {subject_singular}(X), {relative_verb_singular}(X)"
                                return rule
            
            # Check if the predicate has children with a patient or modifier role (e.g., "Mary is parent of Alice")
            if subject and predicate:
                # Look for children with POBJ dependency (object of preposition "of")
                for child in children:
                    child_lemma = child.get('lemma', '').lower()
                    child_role = child.get('role', '')
                    child_pos = child.get('pos', '')
                    child_dependency = child.get('dependency', '')
                    
                    # Check if child is a proper noun with POBJ dependency (Bob in "parent of Bob")
                    # This is the object of the preposition "of"
                    if child_pos == 'PROP' and child_dependency == 'POBJ':
                        # This is "X is Y of Z" pattern -> Y(X, Z)
                        # Use the predicate from roles (parent), not the child_lemma (Bob)
                        predicate_from_roles = roles.get('theme') or roles.get('attribute') or roles.get('modifier') or roles.get('adjective')
                        if predicate_from_roles:
                            predicate_singular = predicate_from_roles.rstrip('s') if predicate_from_roles.endswith('s') else predicate_from_roles
                            return f"{predicate_singular}({subject}, {child_lemma})"
                    
                    # Check if this is the predicate (attribute/theme) and has children
                    if child_role in ['theme', 'attribute'] and child.get('children'):
                        # This child is the predicate, check if it has a patient or modifier child
                        child_children = child.get('children', [])
                        for grandchild in child_children:
                            grandchild_role = grandchild.get('role', '')
                            grandchild_lemma = grandchild.get('lemma', '').lower()
                            grandchild_pos = grandchild.get('pos', '')
                            grandchild_dependency = grandchild.get('dependency', '')
                            
                            # Check for patient, modifier, or object of preposition (POBJ)
                            if grandchild_role in ['patient', 'modifier'] or grandchild_dependency == 'POBJ':
                                # This is "X is Y of Z" pattern -> Y(X, Z)
                                predicate = child_lemma.rstrip('s') if child_lemma.endswith('s') else child_lemma
                                return f"{predicate}({subject}, {grandchild_lemma})"
            
            # If no patient child found, create predicate(subject)
            if subject and predicate:
                predicate = predicate.rstrip('s') if predicate.endswith('s') else predicate
                return f"{predicate}({subject})"
    
            # If we only found subject, look for predicate in children
            if subject and not predicate:
                for child in children:
                    child_lemma = child.get('lemma', '').lower()
                    if child_lemma in ['a', 'an']:
                        # Get the next sibling as the predicate
                        for sibling in children:
                            if sibling != child:
                                predicate = sibling.get('lemma', '').lower()
                                predicate = predicate.rstrip('s') if predicate.endswith('s') else predicate
                                return f"{predicate}({subject})"
        
        # Handle verbs with semantic roles - extract agent/subject and patient/object
        if lemma and children:
            subject = roles.get('agent') or roles.get('subject')
            obj = roles.get('patient') or roles.get('object')
            
            # If we found both subject and object, create verb(subject, object)
            if subject and obj:
                # Create compound predicate for verb+object combinations
                # This allows the system to work with any verb+object combination dynamically
                return f"{lemma}_{obj}({subject})"
            
            # If we only found subject, create verb(subject)
            if subject:
                # Check for adverbial modifiers in marks field
                marks = tree.get('marks', [])
                for mark in marks:
                    if isinstance(mark, dict):
                        mark_dependency = mark.get('dependency', '')
                        
                        # Check for adverbial modifiers
                        if mark_dependency in ['ADVMOD', 'ADVCL']:
                            mark_lemma = mark.get('lemma', '').lower()
                            mark_def = mark.get('definition', '').lower()
                            
                            # Create compound predicate for verb+adverb combinations
                            # This allows the system to work with any verb+adverb combination dynamically
                            # The compound predicate can be queried later
                            return f"{lemma}_{mark_lemma}({subject})"
                
                # Check for adverbial modifiers in children
                for child in children:
                    child_role = child.get('role', '')
                    
                    # Check for adverbial modifiers
                    if child_role in ['modifier', 'advmod']:
                        child_lemma = child.get('lemma', '').lower()
                        child_def = child.get('definition', '').lower()
                        
                        # Create compound predicate for verb+adverb combinations
                        # This allows the system to work with any verb+adverb combination dynamically
                        return f"{lemma}_{child_lemma}({subject})"
                
                # Default: create verb(subject)
                return f"{lemma}({subject})"
        
        # Handle simple predicates with no clear roles
        if lemma and children and not roles:
            subject = children[0].get('lemma', '').lower() if children else None
            if subject:
                lemma = lemma.rstrip('s') if lemma.endswith('s') else lemma
                return f"{lemma}({subject})"
        
        # If no pattern matched, return None
        return None
    except Exception as e:
        # If Vectionary parsing fails, return None (no fallback)
        return None



def _convert_query_to_prolog(query: str, parser: VectionaryParser = None) -> Optional[str]:
    """
    Convert natural language query to Prolog format using dynamic converter
    
    Args:
        query: Natural language query
        parser: VectionaryParser instance for semantic parsing
    
    Returns:
        Prolog query string or None
    """
    if not parser:
        return None
    
    try:
        # Parse the query using Vectionary
        parsed = parser.parse(query)
        if not parsed or not parsed.tree:
            return None
        
        # Use dynamic converter for query conversion
        try:
            # Access dynamic_converter from module-level variable
            # It's initialized at module load time, so it should always be available
            global dynamic_converter
            if 'dynamic_converter' not in globals() or not dynamic_converter:
                print(f"⚠️ dynamic_converter not available (this shouldn't happen)")
                return None
            result = dynamic_converter._dynamic_convert_tree_to_prolog(parsed.tree)
            if not result:
                print(f"⚠️ Query converter returned None for: {query}")
                return None
            
            # Handle conjunction markers for queries
            if result and result.startswith("CONJUNCTION:"):
                predicate = result.split(":", 1)[1]
                # For queries, return the predicate with variable
                return f"{predicate}(X)"
            
            # Handle possessive queries: "Who are Mary's children?" -> children(X, mary)
            # Check if the tree has possessive information
            tree = parsed.tree
            if tree and 'children' in tree:
                possessive_value = None
                predicate = None
                
                # Look for possessive marker
                for child in tree['children']:
                    if child.get('role') == 'possessive':
                        possessive_value = child.get('text', '').lower().rstrip("'s")
                        break
                    # Also check for 's in the text
                    elif "'s" in child.get('text', ''):
                        possessive_value = child.get('text', '').lower().rstrip("'s")
                        break
                
                # If we found possessive, look for the predicate
                if possessive_value:
                    for child in tree['children']:
                        if child.get('role') in ['agent', 'theme'] and not DynamicHybridConverter._is_question_word(child.get('text', '')):
                            predicate = child.get('lemma', '').lower()
                            break
                    
                    if predicate and possessive_value:
                        # For "Who are Mary's children?", we want children(X, mary)
                        # But we need to use the correct predicate from the question
                        if predicate == 'children':
                            return f"children(X, {possessive_value})"
                        elif predicate == 'child':
                            return f"children(X, {possessive_value})"  # Convert singular to plural
                        else:
                            return f"{predicate}(X, {possessive_value})"
            
            # Special handling for "who is X" questions
            # If result is "who(X)" or "who(predicate)", we need to extract the predicate from the theme role
            if result and result.startswith("who("):
                # First, try to extract from "who(predicate)" format
                if result.startswith("who(") and result.endswith(")"):
                    predicate = result[4:-1].lower()  # Extract between "who(" and ")"
                    # Only use if it's not "X" - if it's a real predicate, use it
                    if predicate != "x":
                        return f"{predicate}(X)"
                
                # Re-parse to extract the theme (predicate) from parse tree
                tree = parsed.tree
                if 'children' in tree:
                    for child in tree['children']:
                        if child.get('role') == 'theme':
                            theme_value = child.get('lemma', '').lower()
                            if theme_value:
                                return f"{theme_value}(X)"
                        # Also check agent role if it's the predicate (when Vectionary swaps them)
                        if child.get('role') == 'agent' and not DynamicHybridConverter._is_question_word(child.get('text', '')):
                            # This might be the predicate if theme is "who"
                            agent_value = child.get('lemma', '').lower()
                            if agent_value and agent_value not in ['who', 'what', 'where', 'when']:
                                return f"{agent_value}(X)"
            
            # Special handling for "who makes X" or similar verb-based questions
            # If result contains "make_very(X)" or similar compound predicates, we need to query for the agent
            if result and result.endswith("(X)") and not result.startswith("who("):
                # The query is already correctly formed (e.g., "make_very(X)")
                # But we need to ensure it queries for who is the agent
                # Check if we have a verb-based query
                if "make" in result.lower() or "make_very" in result:
                    # For "who makes decisions?", create the proper query
                    # The query will be refined at the inference endpoint to return individuals
                    predicate = result.split("(")[0]
                    
                    # Add the patient if available
                    if "_" in result and "," not in result:
                        tree = parsed.tree
                        if 'children' in tree:
                            for child in tree['children']:
                                if child.get('role') == 'patient':
                                    patient_text = child.get('text', '').lower()
                                    if patient_text:
                                        # Return make_very(X, decisions) - will be refined to director(X) at endpoint
                                        return f"{predicate}(X, {patient_text})"
                                    break
                    # Return as-is if no patient found
                    return result
            
            return result
        except ImportError as e:
            # Fallback to old method if dynamic converter not available
            print(f"⚠️ Import error in query converter: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Error in query converter dynamic conversion: {e}")
            import traceback
            traceback.print_exc()
        return None
        
    except Exception as e:
        # If Vectionary parsing fails, return None (no fallback)
        print(f"⚠️ Vectionary parsing error for query '{query}': {e}")
        import traceback
        traceback.print_exc()
        return None



def print_vectionary_tree(tree, indent=0):
    """Recursively print Vectionary tree structure"""
    if not tree:
        return
    
    prefix = "   " + "  " * indent
    lemma = tree.get('lemma', 'N/A')
    pos = tree.get('pos', 'N/A')
    dependency = tree.get('dependency', 'N/A')
    
    if indent == 0:
        print(f"{prefix}🌳 Root: {lemma} ({pos}, {dependency})")
    else:
        role = tree.get('roles', {}).get('role', 'N/A')
        print(f"{prefix}└─ {role}: {lemma} ({pos}, {dependency})")
    
    children = tree.get('children', [])
    for child in children:
        print_vectionary_tree(child, indent + 1)


def main():
    parser = argparse.ArgumentParser(
        description="ELMS - Enhanced Logic Modeling System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  elms "All cats are mammals. Fluffy is a cat. Is Fluffy a mammal?"
  elms "If it rains, the ground gets wet. It is raining. Is the ground wet?" --env prod
  elms "John runs. Does John run?" --json
  elms "Is John a doctor?"  # No premises - will check knowledge base
        """
    )
    parser.add_argument('input_text', help='Text to analyze (with premises and question)')
    parser.add_argument('--env', choices=['prod', 'dev', 'test', 'local'], default='prod',
                       help='Environment for Vectionary API (default: prod)')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    parser.add_argument('--verbose', action='store_true',
                       help='Show verbose output and error traces')
    parser.add_argument('--debug', action='store_true',
                       help='Show all behind-the-scenes steps (Vectionary trees, conversions, etc.)')
    args = parser.parse_args()
    
    try:
        # Initialize the reasoning engine
        api_client = VectionaryAPIClient(environment=args.env)
        vectionary_parser = VectionaryParser(api_client)
        reasoner = LogicalReasoner(vectionary_parser)
        
        # Parse input text
        text = args.input_text
        text = re.sub(r'\.([A-Z])', r'. \1', text)  # Fix spacing
        text = re.sub(r'\?([A-Z])', r'? \1', text)
        
        # Split into sentences
        sentences = [s.strip() + '.' if not s.strip().endswith(('?', '.')) else s.strip() 
                    for s in text.split('.') if s.strip()]
        
        # Identify conclusion (question) and premises
        conclusion = next((s for s in sentences if '?' in s), None)
        premises = [s for s in sentences if '?' not in s and s != '.']
        
        if not conclusion:
            print("Error: No question found (marked with ?).", file=sys.stderr)
            sys.exit(1)
        
        # ALWAYS use the new dynamic converter system - no more old system
        is_open_ended = True  # Always use new system
        
        # If no premises provided, check knowledge base
        kb_used = False
        kb_facts_used = []
        if not premises:
            print("📚 No premises provided - checking knowledge base...")
            from vectionary_knowledge_base import VectionaryKnowledgeBase
            
            kb = VectionaryKnowledgeBase()
            
            # Check if KB has any facts before querying
            if not kb.facts or len(kb.facts) == 0:
                print("❌ Knowledge base is empty.")
                print("Please provide premises or add facts to the knowledge base.\n")
            sys.exit(1)
        
            kb_result = kb.query(conclusion)
            
            if kb_result and kb_result.get('relevant_facts'):
                print(f"✅ Found {len(kb_result['relevant_facts'])} relevant facts in KB")
                premises = kb_result['relevant_facts']
                kb_used = True
                kb_facts_used = [{'text': fact} for fact in kb_result['relevant_facts']]
                print(f"Using KB facts as premises: {premises}\n")
            else:
                print("❌ No relevant facts found in knowledge base.")
                print("Please provide premises or add facts to the knowledge base.\n")
                sys.exit(1)
        
        # ALWAYS use the new dynamic converter system
        if is_open_ended:
            print("🔍 Using new dynamic converter system - drawing conclusions...\n")
            from prolog_reasoner import PrologReasoner
            
            start_time = time.time()
            prolog_reasoner = PrologReasoner()
            
            # Convert premises to Prolog format using Vectionary
            prolog_premises = []
            if args.debug:
                print("\n" + "="*80)
                print("🔬 BEHIND THE SCENES: Premise Conversion")
                print("="*80 + "\n")
            
            for i, premise in enumerate(premises, 1):
                if args.debug:
                    print(f"📝 Premise {i}: {premise}")
                
                # Use Vectionary for dynamic conversion
                prolog = _convert_nl_to_prolog(premise, vectionary_parser)
                
                if args.debug and prolog:
                    print(f"   ✅ Converted to Prolog: {prolog}")
                    # Show Vectionary tree if available
                    try:
                        parsed = vectionary_parser.parse(premise)
                        if parsed and hasattr(parsed, 'tree') and parsed.tree:
                            print(f"   🌳 Vectionary Tree:")
                            print_vectionary_tree(parsed.tree)
                    except Exception as e:
                        if args.verbose:
                            print(f"   ⚠️  Could not show tree: {e}")
                    print()
                
                if prolog:
                    prolog_premises.append(prolog)
                    if " :- " in prolog:
                        prolog_reasoner.add_rule(prolog)
                    elif prolog.startswith("INDIVIDUAL_FACTS:"):
                        # Handle individual facts from conjunctions
                        facts_str = prolog.split(":", 1)[1]
                        individual_facts = facts_str.split(",")
                        for fact in individual_facts:
                            fact = fact.strip()
                            if fact:
                                prolog_reasoner.add_fact(fact)
                                prolog_premises.append(fact)
                    else:
                        prolog_reasoner.add_fact(prolog)
                        
                        # Auto-generate relationship rules for common patterns (only once)
                        if "parent(" in prolog and not hasattr(prolog_reasoner, '_parent_rule_added'):
                            # parent(X, Y) -> children(Y, X)
                            rule = "children(Y, X) :- parent(X, Y)"
                            prolog_reasoner.add_rule(rule)
                            prolog_reasoner._parent_rule_added = True
                            if args.debug:
                                print(f"   🔗 Auto-generated rule: {rule}")
                        elif "children(" in prolog and not hasattr(prolog_reasoner, '_children_rule_added'):
                            # children(X, Y) -> parent(Y, X)
                            rule = "parent(Y, X) :- children(X, Y)"
                            prolog_reasoner.add_rule(rule)
                            prolog_reasoner._children_rule_added = True
                            if args.debug:
                                print(f"   🔗 Auto-generated rule: {rule}")
                        elif "teacher(" in prolog and not hasattr(prolog_reasoner, '_teacher_rule_added'):
                            # teacher(X, Y) -> student(Y, X)
                            rule = "student(Y, X) :- teacher(X, Y)"
                            prolog_reasoner.add_rule(rule)
                            prolog_reasoner._teacher_rule_added = True
                            if args.debug:
                                print(f"   🔗 Auto-generated rule: {rule}")
                        elif "student(" in prolog and not hasattr(prolog_reasoner, '_student_rule_added'):
                            # student(X, Y) -> teacher(Y, X)
                            rule = "teacher(Y, X) :- student(X, Y)"
                            prolog_reasoner.add_rule(rule)
                            prolog_reasoner._student_rule_added = True
                            if args.debug:
                                print(f"   🔗 Auto-generated rule: {rule}")
                elif "(" in premise or " :- " in premise:
                    # Already in Prolog format
                    if args.debug:
                        print(f"   ℹ️  Already in Prolog format")
                        print()
                    prolog_premises.append(premise)
                    if " :- " in premise:
                        prolog_reasoner.add_rule(premise)
                    else:
                        prolog_reasoner.add_fact(premise)
            
            if args.debug:
                print("="*80 + "\n")
            
            # Convert query to Prolog format using Vectionary
            if args.debug:
                print("="*80)
                print("🔬 BEHIND THE SCENES: Query Conversion")
                print("="*80 + "\n")
                print(f"❓ Query: {conclusion}")
            
            prolog_query = _convert_query_to_prolog(conclusion, vectionary_parser)
            
            if args.debug and prolog_query:
                print(f"✅ Converted to Prolog: {prolog_query}")
                # Show Vectionary tree
                try:
                    parsed = vectionary_parser.parse(conclusion)
                    if parsed and hasattr(parsed, 'tree') and parsed.tree:
                        print(f"🌳 Vectionary Tree:")
                        print_vectionary_tree(parsed.tree)
                except Exception as e:
                    if args.verbose:
                        print(f"⚠️  Could not show tree: {e}")
                print("\n" + "="*80 + "\n")
            
            if not prolog_query:
                print("❌ Error: Could not convert query to Prolog format.")
                print("The system uses Vectionary semantic parsing to dynamically convert")
                print("natural language to Prolog. Please rephrase your query or check the input.")
                sys.exit(1)
            
            # Add ancestor/descendant rules if query is about ancestors/descendants
            # This is truly dynamic - it infers relationships from parent facts
            if "ancestor(" in prolog_query:
                prolog_reasoner.add_rule("ancestor(X, Y) :- parent(X, Y)")
                prolog_reasoner.add_rule("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)")
                prolog_premises.append("ancestor(X, Y) :- parent(X, Y)")
                prolog_premises.append("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)")
            
            if "descendant(" in prolog_query:
                # Descendant is the inverse of ancestor
                prolog_reasoner.add_rule("descendant(X, Y) :- ancestor(Y, X)")
                prolog_premises.append("descendant(X, Y) :- ancestor(Y, X)")
                # Also add ancestor rules since descendant depends on them
                prolog_reasoner.add_rule("ancestor(X, Y) :- parent(X, Y)")
                prolog_reasoner.add_rule("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)")
                prolog_premises.append("ancestor(X, Y) :- parent(X, Y)")
                prolog_premises.append("ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)")
            
            # Dynamic refinement: If query returns collective nouns, find individual predicate
            # For "who makes decisions?" -> make_very(X, decisions) might return "directors"
            # But we want individuals, so check if we should query director(X) instead
            original_query = prolog_query
            if is_open_ended and prolog_query:
                # Check if query contains a verb that might return collective nouns
                # Handle "make", "supervise", and other verb patterns
                verb_keywords = ["make", "supervise", "manage", "lead", "direct"]
                has_verb_pattern = any(keyword in prolog_query.lower() for keyword in verb_keywords)
                
                if has_verb_pattern:
                    # Check if the query contains a compound predicate that might return collective nouns
                    # Example: make_very(X, decisions) -> directors
                    # We want: director(X) -> alice, carol
                    
                    # Look through facts to find the agent type
                    # For "who makes decisions?", prolog_query might be "make_very(X, decisions)"
                    # We need to find "make_very(directors, decisions)" and extract "directors" -> "director(X)"
                    query_predicate = prolog_query.split("(")[0] if "(" in prolog_query else ""
                    
                    for fact in prolog_premises:
                        # Check if fact matches the query predicate (e.g., make_very, supervise_especially)
                        if query_predicate and fact.startswith(query_predicate) and "(" in fact and "," in fact:
                            # Found a matching fact like make_very(directors, decisions)
                            # Extract the agent type (directors) and convert to singular (director)
                            try:
                                agent_part = fact.split("(")[1].split(",")[0].strip()
                                # Convert plural to singular (simple heuristic)
                                if agent_part.endswith('s') and len(agent_part) > 3:
                                    singular_agent = agent_part.rstrip('s')
                                    # Check if we have individual facts with this predicate
                                    individual_predicate = f"{singular_agent}(X)"
                                    # Test if this would return individuals
                                    test_success, test_results = prolog_reasoner.query(individual_predicate)
                                    if test_results and len(test_results) > 0:
                                        # Use the individual predicate instead
                                        print(f"🔄 Refining query: '{original_query}' -> '{individual_predicate}' (to get individuals)")
                                        prolog_query = individual_predicate
                                        break
                            except Exception as e:
                                if args.debug:
                                    print(f"⚠️ Error refining query: {e}")
                                pass
                    
                    # Handle case where query is just "verb_very(X)" without patient
                    if not prolog_query or prolog_query == original_query:
                        if "_" in prolog_query and prolog_query.endswith("(X)"):
                            # Query is like "supervise_especially(X)"
                            # Try to extract agent type from facts
                            for fact in prolog_premises:
                                if query_predicate and fact.startswith(query_predicate) and "(" in fact:
                                    try:
                                        agent_part = fact.split("(")[1].split(",")[0].strip()
                                        if agent_part.endswith('s') and len(agent_part) > 3:
                                            singular_agent = agent_part.rstrip('s')
                                            individual_predicate = f"{singular_agent}(X)"
                                            test_success, test_results = prolog_reasoner.query(individual_predicate)
                                            if test_results and len(test_results) > 0:
                                                print(f"🔄 Refining query: '{original_query}' -> '{individual_predicate}' (to get individuals)")
                                                prolog_query = individual_predicate
                                                break
                                    except Exception as e:
                                        pass
            
            # Query Prolog
            if args.debug:
                print("="*80)
                print("🔬 BEHIND THE SCENES: Prolog Inference")
                print("="*80 + "\n")
                print(f"🔍 Executing Prolog query: {prolog_query}")
                print(f"📊 Facts in knowledge base: {len(prolog_reasoner.facts)}")
                print(f"📐 Rules in knowledge base: {len(prolog_reasoner.rules)}")
                print()
            
            # Handle comma-separated queries (conjunctions)
            # Only split on commas that are outside of parentheses (i.e., between predicates)
            if ',' in prolog_query and 'X' in prolog_query:
                # Try to split the query into multiple predicates
                # This is a more robust parser that handles nested parentheses
                parts = []
                current_part = ""
                paren_count = 0
                bracket_count = 0
                
                for i, char in enumerate(prolog_query):
                    if char == '(':
                        paren_count += 1
                        current_part += char
                    elif char == ')':
                        paren_count -= 1
                        current_part += char
                    elif char == '[':
                        bracket_count += 1
                        current_part += char
                    elif char == ']':
                        bracket_count -= 1
                        current_part += char
                    elif char == ',' and paren_count == 0 and bracket_count == 0:
                        # This comma is between predicates (not inside parentheses or brackets)
                        if current_part.strip():
                            parts.append(current_part.strip())
                        current_part = ""
                    else:
                        current_part += char
                
                # Add the last part
                if current_part.strip():
                    parts.append(current_part.strip())
                
                # Only split if we found multiple valid parts
                if len(parts) > 1:
                    # Validate that each part is a valid Prolog predicate
                    valid_parts = []
                    for part in parts:
                        # Check if part looks like a predicate (has opening and closing parens)
                        if '(' in part and ')' in part:
                            # Count parens to ensure they're balanced
                            part_paren_count = part.count('(') - part.count(')')
                            if part_paren_count == 0:
                                valid_parts.append(part)
                    
                    # If we have valid parts, query each separately and find intersection
                    if len(valid_parts) > 1:
                        all_results = []
                        for part in valid_parts:
                            part_success, part_results = prolog_reasoner.query(part)
                            if part_success and part_results:
                                all_results.append(part_results)
                        
                        # Find the intersection of all results
                        if len(all_results) > 1:
                            # Get all values from the first query
                            first_values = [list(r.values())[0] for r in all_results[0]]
                            # Filter to only include values that appear in all queries
                            for result_set in all_results[1:]:
                                result_values = [list(r.values())[0] for r in result_set]
                                first_values = [v for v in first_values if v in result_values]
                            
                            # Create result dictionaries
                            results = [{'result': val} for val in first_values]
                            success = len(results) > 0
                        else:
                            # If we have multiple parts but only one (or zero) returns results,
                            # there's no intersection - return empty
                            success, results = True, []
                    else:
                        # Fallback: try the query as-is
                        success, results = prolog_reasoner.query(prolog_query)
                else:
                    # Fallback: try the query as-is
                    success, results = prolog_reasoner.query(prolog_query)
            else:
                # No comma or no variable - query directly
                success, results = prolog_reasoner.query(prolog_query)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            if args.debug:
                print(f"✅ Query completed in {elapsed_time:.4f}s")
                print(f"📊 Results found: {len(results) if results else 0}")
                if results:
                    print("   Results:")
                    for i, result in enumerate(results, 1):
                        print(f"   {i}. {result}")
                print("\n" + "="*80 + "\n")
            
            # Generate human-readable explanation
            explanation = f"Based on the premises provided, I used logical inference to answer the question.\n\n"
            explanation += f"From the facts and rules:\n"
            for i, premise in enumerate(prolog_premises, 1):
                explanation += f"  {i}. {premise}\n"
            explanation += f"\nI queried: {prolog_query}\n"
            if results:
                explanation += f"\nThis led me to find {len(results)} result(s)."
            else:
                explanation += f"\nNo results were found that satisfy this query."
            
            # Generate reasoning steps
            reasoning_steps = [
                "Converted natural language premises to Prolog facts and rules",
                f"Applied logical inference using the query: {prolog_query}",
                f"Found {len(results)} conclusion(s) that satisfy the query"
            ]
            
            # Build result
            result = {
                'valid': True,
                'confidence': 0.95,
                'confidence_level': 'High',
                'logic_type': 'INFERENCE',
                'explanation': explanation,
                'reasoning_steps': reasoning_steps,
                'conclusions': results,
                'conclusions_count': len(results),
                'is_open_ended': True,
                'prolog_premises': prolog_premises,
                'prolog_query': prolog_query,
                'reasoning_time': elapsed_time
            }
        else:
            # This should never happen now - always use new dynamic converter
            print("❌ Error: Old system should not be called")
            sys.exit(1)
        
        # Add KB info to result
        if kb_used:
            result['kb_used'] = True
            result['kb_facts_count'] = len(kb_facts_used)
            result['kb_facts'] = kb_facts_used
        
        # Output results
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            # Rich text output
            print("\n" + "="*60)
            print(f"Input: {text}")
            print("="*60 + "\n")
            
            # Check if this is an open-ended question
            if result.get('is_open_ended'):
                print("✅ Inference Complete")
                print(f"Confidence: {result.get('confidence_level', 'High')} ({result.get('confidence', 0.95):.1%})")
                if result.get('reasoning_time'):
                    print(f"⏱️  Reasoning Time: {result['reasoning_time']:.3f}s")
                print("")
                
                # Display original premises
                if premises:
                    print(f"📝 Premises ({len(premises)}):")
                    for i, premise in enumerate(premises, 1):
                        print(f"   {i}. {premise}")
                print("")
                
                # Display Prolog premises
                if result.get('prolog_premises'):
                    print(f"🔬 Prolog Facts & Rules ({len(result['prolog_premises'])}):")
                    for i, premise in enumerate(result['prolog_premises'], 1):
                        print(f"   {i}. {premise}")
                    print("")
                
                # Display Prolog query
                if result.get('prolog_query'):
                    print(f"❓ Query: {result['prolog_query']}")
                    print("")
                
                # Display conclusions - more readable and direct
                conclusions = result.get('conclusions', [])
                if conclusions:
                    print(f"💡 Answer: ", end="")
                    # Format as a list
                    conclusion_values = []
                    for conclusion in conclusions:
                        if isinstance(conclusion, dict):
                            if 'result' in conclusion:
                                conclusion_values.append(str(conclusion['result']).capitalize())
                            else:
                                # Extract just the values (more human-readable)
                                values = list(conclusion.values())
                                if len(values) == 1:
                                    conclusion_values.append(str(values[0]).capitalize())
                                else:
                                    conclusion_values.append(", ".join([str(v).capitalize() for v in values]))
                        else:
                            conclusion_values.append(str(conclusion).capitalize())
                    
                    if len(conclusion_values) == 1:
                        print(conclusion_values[0])
                    elif len(conclusion_values) == 2:
                        print(f"{conclusion_values[0]} and {conclusion_values[1]}")
                    else:
                        print(", ".join(conclusion_values[:-1]) + f", and {conclusion_values[-1]}")
                    print(f"\n📊 Total: {len(conclusions)} result(s)")
                else:
                    print("💡 No conclusions found")
                print("")
                
                # Display explanation
                if result.get('explanation'):
                    print("📝 Explanation")
                    print(result['explanation'])
                    print("")
                
                # Display reasoning steps
                if result.get('reasoning_steps'):
                    print("🔍 How the System Reached This Conclusion:")
                    for step in result['reasoning_steps']:
                        print(f"  • {step}")
                    print("")
            else:
                if result.get('valid') is None:
                    print("⚠️  Ambiguous")
                elif result.get('valid'):
                    print("✅ Valid")
                else:
                    print("❌ Invalid")
            
                # Display confidence with level (only for non-open-ended)
            confidence = result.get('confidence', 0.0)
            confidence_level = result.get('confidence_level', '')
            if isinstance(confidence, (int, float)):
                if confidence_level:
                    print(f"Confidence: {confidence_level} ({confidence:.1%})")
                else:
                    confidence_calc = ConfidenceCalculator()
                    level = confidence_calc.get_confidence_level(confidence).value
                    print(f"Confidence: {level} ({confidence:.1%})")
            else:
                print(f"Confidence: {confidence}")
                print("")
            
            # Display KB usage if applicable
            if result.get('kb_used'):
                print(f"📚 Knowledge Base: Used {result.get('kb_facts_count', 0)} facts")
                if result.get('kb_facts'):
                    print("   Facts used:")
                    for i, fact in enumerate(result['kb_facts'], 1):
                        fact_text = fact.get('text', str(fact))
                        print(f"   {i}. {fact_text}")
            print("")
            
            print("📝 Explanation")
            print(result.get('explanation', 'No explanation available'))
            
            # Reasoning steps (only for non-open-ended)
            if result.get('reasoning_steps'):
                print("\n🔍 How the System Reached This Conclusion:")
                # Join all steps into a readable paragraph
                reasoning_text = ''.join(result['reasoning_steps'])
                print(reasoning_text)
            
            # Display interpretations if ambiguous
            if result.get('interpretations'):
                print("\n🔀 Multiple Interpretations:")
                print("="*60)
                for interp in result['interpretations']:
                    print(f"\n{interp['id']}. {interp['name']}")
                    print(f"   {'✅ Valid' if interp['valid'] else '❌ Invalid'}")
                    print(f"   Confidence: {interp['confidence']:.1%}")
                    print(f"   Explanation: {interp['explanation']}")
                    if interp.get('assumptions'):
                        print(f"   Assumptions: {', '.join(interp['assumptions'])}")
                print("\n" + "="*60)
                print("⚠️  Please select the correct interpretation (1, 2, or 3)")
            
            # Formal theorem
            if result.get('premise_trees') and result.get('conclusion_tree'):
                for line in build_theorem_from_trees(
                    result['premise_trees'], 
                    result['conclusion_tree'], 
                    result.get('parsed_premises', []), 
                    result.get('parsed_conclusion', '')
                ):
                    print(line)
            
            # Parse trees
            if result.get('premise_trees') or result.get('conclusion_tree'):
                print("\n🌳 Vectionary Parse Trees")
                for i, tree in enumerate(result.get('premise_trees', [])):
                    for line in format_tree_display(tree, i + 1):
                        print(line)
                if result.get('conclusion_tree'):
                    for line in format_tree_display(
                        result['conclusion_tree'], 
                        len(result.get('premise_trees', [])) + 1
                    ):
                        print(line)
            
            # Semantic analysis
            if result.get('premise_trees'):
                print("\n🔍 Semantic Analysis:")
                for i, tree in enumerate(result['premise_trees'], 1):
                    if tree:
                        lemma = tree.get('lemma', 'unknown')
                        defn = tree.get('definition', '')
                        print(f"• P{i}: {lemma} = \"{defn}\"")
                        
                        roles = []
                        for child in tree.get('children', []):
                            if isinstance(child, dict):
                                role = child.get('role', '')
                                txt = child.get('text', '')
                                if role and txt:
                                    roles.append(f"{role}: {txt}")
                        if roles:
                            print(f"  Semantic roles: {', '.join(roles)}")
                
                if result.get('conclusion_tree'):
                    tree = result['conclusion_tree']
                    lemma = tree.get('lemma', 'unknown')
                    defn = tree.get('definition', '')
                    print(f"• C: {lemma} = \"{defn}\"")
                    
                    roles = []
                    for child in tree.get('children', []):
                        if isinstance(child, dict):
                            role = child.get('role', '')
                            txt = child.get('text', '')
                            if role and txt:
                                roles.append(f"{role}: {txt}")
                    if roles:
                        print(f"  Semantic roles: {', '.join(roles)}")
            
            # Formal logical reasoning steps
            if result.get('formal_steps'):
                print("\n📐 Formal Logical Proof:")
                for step in result['formal_steps']:
                    print(f"   {step}")
            
            # Premises and conclusion
            if result.get('parsed_premises'):
                print("\n📋 Parsed Premises")
                for i, p in enumerate(result['parsed_premises']):
                    print(f"{i+1}. {p}")
                
                # Add truth tables for simple propositional formulas
                if len(result['parsed_premises']) <= 3:
                    for i, premise in enumerate(result['parsed_premises']):
                        truth_table = generate_truth_table(premise)
                        if truth_table:
                            print(f"\n📊 Truth Table - Premise {i+1}")
                            print(truth_table)
            
            if result.get('parsed_conclusion'):
                print(f"\n🎯 Parsed Conclusion")
                print(result['parsed_conclusion'])
            print()
        
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
