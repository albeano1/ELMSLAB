#!/usr/bin/env python3
"""
Environment Setup Script

This script helps users set up their environment variables for the LLM integration.
"""

import os
import sys
from pathlib import Path


def setup_environment():
    """Set up environment variables for LLM integration."""
    print("🔧 Enhanced Logic Modeling System - Environment Setup")
    print("=" * 60)
    
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    env_example = project_root / "env.example"
    
    # Check if .env already exists
    if env_file.exists():
        print("✅ .env file already exists")
        response = input("Do you want to update it? (y/N): ").strip().lower()
        if response != 'y':
            print("Environment setup cancelled.")
            return
    
    # Check if env.example exists
    if not env_example.exists():
        print("❌ env.example file not found")
        print("Please make sure you're running this from the project root directory.")
        return
    
    # Copy env.example to .env
    try:
        with open(env_example, 'r') as src:
            content = src.read()
        
        with open(env_file, 'w') as dst:
            dst.write(content)
        
        print("✅ Created .env file from env.example")
        
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return
    
    # Get API key from user
    print("\n🔑 API Key Setup")
    print("Get your Anthropic API key from: https://console.anthropic.com/")
    
    api_key = input("Enter your Anthropic API key (or press Enter to skip): ").strip()
    
    if api_key:
        # Update the .env file with the actual API key
        try:
            with open(env_file, 'r') as f:
                content = f.read()
            
            # Replace the placeholder with the actual API key
            content = content.replace('your-anthropic-api-key-here', api_key)
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print("✅ API key added to .env file")
            
        except Exception as e:
            print(f"❌ Failed to update .env file: {e}")
            return
    
    # Show next steps
    print("\n🎉 Environment setup complete!")
    print("\nNext steps:")
    print("1. Run the setup script: python setup_llm_integration.py")
    print("2. Start the API server: python vectionary_98_api.py")
    print("3. Open the web interface: logic_ui_llm_enhanced.html")
    
    print("\n📝 Note: The .env file is automatically ignored by git for security.")
    print("   You can edit it anytime with your preferred text editor.")


if __name__ == "__main__":
    setup_environment()
