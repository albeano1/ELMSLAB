#!/usr/bin/env python3
"""
LLM Integration Setup Script

This script helps set up the LLM integration for the Enhanced Logic Modeling System.
It checks dependencies, validates API keys, and runs initial tests.
"""

import os
import sys
import subprocess
import json
import asyncio
from pathlib import Path


class LLMIntegrationSetup:
    """Setup and validation for LLM integration."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.requirements_file = self.project_root / "requirements.txt"
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        
    def print_header(self, title: str):
        """Print a formatted header."""
        print("\n" + "=" * 60)
        print(f"🔧 {title}")
        print("=" * 60)
    
    def print_step(self, step: str, status: str = ""):
        """Print a setup step."""
        if status:
            print(f"✅ {step} - {status}")
        else:
            print(f"🔄 {step}...")
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        self.print_header("Checking Python Version")
        
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            self.print_step(f"Python {version.major}.{version.minor}.{version.micro}", "Compatible")
            return True
        else:
            print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
            print("   Required: Python 3.8 or higher")
            return False
    
    def check_virtual_environment(self) -> bool:
        """Check if virtual environment exists and is activated."""
        self.print_header("Checking Virtual Environment")
        
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            self.print_step("Virtual environment", "Active")
            return True
        else:
            print("⚠️ Virtual environment not detected")
            print("   It's recommended to use a virtual environment")
            
            if self.venv_path.exists():
                print(f"   Found venv at: {self.venv_path}")
                print("   To activate: source venv/bin/activate")
                return True
            else:
                print("   No venv found. Consider creating one:")
                print("   python -m venv venv")
                return False
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        self.print_header("Checking Dependencies")
        
        required_packages = [
            'fastapi',
            'uvicorn',
            'pydantic',
            'aiohttp',
            'spacy'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                self.print_step(f"{package}", "Installed")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package} - Missing")
        
        if missing_packages:
            print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
            print("   Install with: pip install -r requirements.txt")
            return False
        
        return True
    
    def check_api_key(self) -> bool:
        """Check if Anthropic API key is available."""
        self.print_header("Checking API Key")
        
        if self.api_key:
            # Mask the API key for security
            masked_key = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else "***"
            self.print_step("ANTHROPIC_API_KEY", f"Set ({masked_key})")
            return True
        else:
            print("❌ ANTHROPIC_API_KEY not found")
            print("   Set it with: export ANTHROPIC_API_KEY='your-key-here'")
            print("   Or add to .env file: ANTHROPIC_API_KEY=your-key-here")
            return False
    
    def install_dependencies(self) -> bool:
        """Install missing dependencies."""
        self.print_header("Installing Dependencies")
        
        if not self.requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        try:
            self.print_step("Installing packages from requirements.txt")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                self.print_step("Dependencies", "Installed successfully")
                return True
            else:
                print(f"❌ Installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Installation error: {e}")
            return False
    
    async def test_claude_connection(self) -> bool:
        """Test connection to Claude API."""
        self.print_header("Testing Claude API Connection")
        
        if not self.api_key:
            print("❌ Cannot test Claude API - no API key")
            return False
        
        try:
            # Import and test Claude integration
            sys.path.append(str(self.project_root))
            from claude_integration import ClaudeIntegration
            
            claude = ClaudeIntegration()
            response = await claude.query_claude("Hello, this is a test message.")
            
            if response and response.content:
                self.print_step("Claude API", "Connection successful")
                return True
            else:
                print("❌ Claude API - No response received")
                return False
                
        except Exception as e:
            print(f"❌ Claude API - Connection failed: {e}")
            return False
    
    def test_vectionary_system(self) -> bool:
        """Test Vectionary system functionality."""
        self.print_header("Testing Vectionary System")
        
        try:
            sys.path.append(str(self.project_root))
            from vectionary_98_percent_solution import Vectionary98PercentSolution
            
            vectionary = Vectionary98PercentSolution()
            result = vectionary.parse_with_vectionary_98("All birds can fly")
            
            if result and result.formula:
                self.print_step("Vectionary parsing", "Working")
                return True
            else:
                print("❌ Vectionary parsing - Failed")
                return False
                
        except Exception as e:
            print(f"❌ Vectionary system - Error: {e}")
            return False
    
    def test_knowledge_base(self) -> bool:
        """Test knowledge base functionality."""
        self.print_header("Testing Knowledge Base")
        
        try:
            sys.path.append(str(self.project_root))
            from vectionary_knowledge_base import VectionaryKnowledgeBase
            
            kb = VectionaryKnowledgeBase()
            fact = kb.add_fact("Test fact for setup", confidence=0.9)
            
            if fact and fact.id:
                self.print_step("Knowledge base", "Working")
                # Clean up test fact
                kb.delete_fact(fact.id)
                return True
            else:
                print("❌ Knowledge base - Failed to add fact")
                return False
                
        except Exception as e:
            print(f"❌ Knowledge base - Error: {e}")
            return False
    
    async def run_comprehensive_test(self) -> bool:
        """Run comprehensive integration test."""
        self.print_header("Running Comprehensive Test")
        
        try:
            # Test the complete workflow
            sys.path.append(str(self.project_root))
            from claude_integration import ClaudeIntegration
            from vectionary_98_percent_solution import Vectionary98PercentSolution
            
            # Test Vectionary
            vectionary = Vectionary98PercentSolution()
            parsed = vectionary.parse_with_vectionary_98("All birds can fly")
            
            if not parsed or not parsed.formula:
                print("❌ Vectionary parsing failed")
                return False
            
            # Test Claude (if available)
            if self.api_key:
                claude = ClaudeIntegration()
                result = await claude.reason_with_claude(
                    ["All birds can fly", "Tweety is a bird"],
                    "Can Tweety fly?"
                )
                
                if not result or "claude_response" not in result:
                    print("❌ Claude reasoning failed")
                    return False
            
            self.print_step("Comprehensive test", "Passed")
            return True
            
        except Exception as e:
            print(f"❌ Comprehensive test failed: {e}")
            return False
    
    def create_env_file(self) -> bool:
        """Create .env file template."""
        self.print_header("Creating Environment File")
        
        env_file = self.project_root / ".env"
        env_example = self.project_root / "env.example"
        
        if env_file.exists():
            print("✅ .env file already exists")
            return True
        
        if not env_example.exists():
            print("❌ env.example file not found")
            return False
        
        try:
            # Copy env.example to .env
            with open(env_example, 'r') as src:
                content = src.read()
            
            with open(env_file, 'w') as dst:
                dst.write(content)
            
            self.print_step(".env file", "Created from env.example")
            print("   Edit .env file and add your actual API keys")
            print("   The .env file is already in .gitignore for security")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    
    def print_summary(self, results: dict):
        """Print setup summary."""
        self.print_header("Setup Summary")
        
        total_checks = len(results)
        passed_checks = sum(1 for result in results.values() if result)
        
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Success Rate: {(passed_checks/total_checks)*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        for check, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} {check}")
        
        if passed_checks == total_checks:
            print("\n🎉 Setup Complete! Your LLM integration is ready to use.")
            print("\nNext steps:")
            print("1. Start the API server: python vectionary_98_api.py")
            print("2. Open the web interface: logic_ui_llm_enhanced.html")
            print("3. Run tests: python test_llm_integration.py")
        else:
            print("\n⚠️ Setup incomplete. Please address the failed checks above.")
            print("\nCommon solutions:")
            print("- Install dependencies: pip install -r requirements.txt")
            print("- Set API key: export ANTHROPIC_API_KEY='your-key'")
            print("- Activate virtual environment: source venv/bin/activate")
    
    async def run_setup(self):
        """Run the complete setup process."""
        print("🚀 Enhanced Logic Modeling System - LLM Integration Setup")
        print("This script will check and configure your system for LLM integration.")
        
        results = {}
        
        # Basic checks
        results["Python Version"] = self.check_python_version()
        results["Virtual Environment"] = self.check_virtual_environment()
        results["Dependencies"] = self.check_dependencies()
        results["API Key"] = self.check_api_key()
        
        # Install dependencies if needed
        if not results["Dependencies"]:
            results["Dependencies"] = self.install_dependencies()
        
        # Create .env file
        results["Environment File"] = self.create_env_file()
        
        # System tests
        results["Vectionary System"] = self.test_vectionary_system()
        results["Knowledge Base"] = self.test_knowledge_base()
        
        # Claude tests (only if API key is available)
        if results["API Key"]:
            results["Claude Connection"] = await self.test_claude_connection()
            results["Comprehensive Test"] = await self.run_comprehensive_test()
        else:
            results["Claude Connection"] = False
            results["Comprehensive Test"] = False
        
        # Print summary
        self.print_summary(results)
        
        return results


async def main():
    """Main function."""
    setup = LLMIntegrationSetup()
    results = await setup.run_setup()
    
    # Exit with appropriate code
    total_checks = len(results)
    passed_checks = sum(1 for result in results.values() if result)
    
    if passed_checks >= total_checks * 0.8:  # 80% success rate
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    asyncio.run(main())
