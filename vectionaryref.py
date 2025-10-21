#!/usr/bin/env python3
# for refrence on what vectionary can do
"""
Simple script to get mod2 trees from the Parsimony API and output as JSON.
Usage: python get_mod2_tree.py "Your text here"
       python get_mod2_tree.py "Your text here" --tree  # For tree visualization
"""

import sys
import json
import argparse
import requests
from typing import Dict, Any, List

# API endpoints
ENDPOINTS = {
    'local': 'http://localhost:8001/arborize/mod1',
    'dev': 'https://us-central1-parsimony-server.cloudfunctions.net/arborize-dev/arborize/mod1',
    'test': 'https://us-central1-parsimony-server.cloudfunctions.net/arborize-test/arborize/mod1',
    'prod': 'https://us-central1-parsimony-server.cloudfunctions.net/arborize/arborize/mod1'
}

def pretty_print_tree(tree: Dict[str, Any], indent: int = 0, show_details: bool = False) -> None:
    """Pretty print a tree structure with indentation."""
    prefix = "  " * indent

    # Get the word and POS - can be directly in the tree or in vectionary_element
    if 'vectionary_element' in tree:
        word = tree['vectionary_element'].get('word', tree['vectionary_element'].get('text', '?'))
        pos = tree['vectionary_element'].get('pos', '?')
        dependency = tree['vectionary_element'].get('dependency', '')
        lemma = tree['vectionary_element'].get('lemma', '')
    else:
        # Direct format (mod2 style)
        word = tree.get('text', tree.get('word', '?'))
        pos = tree.get('pos', '?')
        dependency = tree.get('dependency', '')
        lemma = tree.get('lemma', '')

    role = tree.get('role', '')
    marks = tree.get('marks', [])
    degree = tree.get('degree', '')

    # Build the display string
    display_parts = [f"{word} ({pos})"]

    if role and role != 'none':
        display_parts.append(f"role:{role}")

    if marks:
        # Handle marks which can be strings or dicts
        mark_strs = []
        for mark in marks:
            if isinstance(mark, str):
                mark_strs.append(mark)
            elif isinstance(mark, dict):
                # For dict marks, use the text or ID
                mark_strs.append(mark.get('text', mark.get('ID', str(mark))))
        if mark_strs:
            display_parts.append(f"marks:[{', '.join(mark_strs)}]")

    if degree and degree != 'none':
        display_parts.append(f"degree:{degree}")

    # Print the node
    print(f"{prefix}{'└─ ' if indent > 0 else ''}{' '.join(display_parts)}")

    # Print additional details if requested
    if show_details:
        if dependency:
            print(f"{prefix}   dependency: {dependency}")
        if lemma and lemma != word:
            print(f"{prefix}   lemma: {lemma}")
        if 'definition' in tree:
            print(f"{prefix}   definition: {tree['definition'][:80]}...")

    # Print children
    children = tree.get('children', [])
    for child in children:
        pretty_print_tree(child, indent + 1, show_details)

def get_mod2_trees(text: str, environment: str = 'local', provider: str = None, model: str = None) -> List[Dict[str, Any]]:
    """Get mod2 trees from the API."""
    endpoint = ENDPOINTS[environment]

    # Prepare request payload
    payload = {'text': text}
    if provider:
        payload['llm_provider'] = provider
    if model:
        payload['llm_model'] = model

    # Make API request
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        response.raise_for_status()
        result = response.json()

        # Extract trees from response
        trees = []
        if 'trees' in result:
            # Multiple trees response (most common for mod1/mod2)
            trees.extend(result['trees'])
        elif 'tree' in result:
            # Single tree response
            trees.append(result['tree'])
        elif 'sentences' in result:
            # Sentences with individual trees
            for sentence in result['sentences']:
                if 'tree' in sentence:
                    trees.append(sentence['tree'])

        return trees

    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"API Error: {json.dumps(error_detail, indent=2)}", file=sys.stderr)
            except:
                print(f"API Response: {e.response.text}", file=sys.stderr)
        return []

def main():
    parser = argparse.ArgumentParser(description='Get mod2 trees and pretty print them')
    parser.add_argument('text', help='Text to process (can be multiple sentences)')
    parser.add_argument('--env', choices=['local', 'dev', 'test', 'prod'],
                       default='local', help='Environment to use (default: local)')
    parser.add_argument('--provider', help='LLM provider (gemini, openai, anthropic, cerebras)')
    parser.add_argument('--model', help='Specific LLM model to use')
    parser.add_argument('--tree', action='store_true', help='Output tree visualization instead of JSON')
    parser.add_argument('--details', action='store_true', help='Show additional details (only for tree view)')

    args = parser.parse_args()

    # Only show header info if in tree mode
    if args.tree:
        print(f"\nProcessing: \"{args.text}\"")
        print(f"Environment: {args.env}")
        if args.provider:
            print(f"Provider: {args.provider}")
        if args.model:
            print(f"Model: {args.model}")
        print("-" * 50)

    # Get trees from API
    trees = get_mod2_trees(args.text, args.env, args.provider, args.model)

    if not trees:
        print("No trees returned from API", file=sys.stderr)
        return 1

    # Output results
    if args.tree:
        # Pretty print each tree visualization
        for i, tree in enumerate(trees):
            if len(trees) > 1:
                print(f"\nTree {i+1}:")
            pretty_print_tree(tree, show_details=args.details)
    else:
        # Default: Pretty printed JSON output with proper indentation
        if len(trees) > 1:
            output = {'trees': trees}
        else:
            output = {'tree': trees[0]}
        print(json.dumps(output, indent=2, ensure_ascii=False))

    return 0

if __name__ == '__main__':
    sys.exit(main())