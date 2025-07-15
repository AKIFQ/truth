#!/usr/bin/env python3
"""
Setup script for Truth vs Bias Detector
Automates installation and verification of dependencies.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_ollama():
    """Check if Ollama is installed and running."""
    print("🤖 Checking Ollama installation...")
    
    # Check if ollama command exists
    result = run_command("which ollama", "Checking Ollama installation")
    if not result:
        print("❌ Ollama not found. Please install Ollama first:")
        print("   Visit: https://ollama.ai/download")
        return False
    
    # Check if Ollama is running
    result = run_command("ollama list", "Checking Ollama service")
    if not result:
        print("❌ Ollama service not running. Please start Ollama:")
        print("   Run: ollama serve")
        return False
    
    print("✅ Ollama is installed and running")
    return True

def check_target_model():
    """Check if the target model is available."""
    print("📦 Checking target model...")
    
    result = run_command("ollama list", "Listing available models")
    if not result:
        return False
    
    if "llama3.1:8b" in result:
        print("✅ Target model llama3.1:8b is available")
        return True
    else:
        print("⚠️  Target model llama3.1:8b not found")
        print("   Available models:")
        for line in result.strip().split('\n')[1:]:  # Skip header
            if line.strip():
                print(f"   - {line.strip()}")
        
        response = input("\n🤔 Would you like to pull llama3.1:8b? (y/n): ")
        if response.lower() in ['y', 'yes']:
            result = run_command("ollama pull llama3.1:8b", "Pulling llama3.1:8b model")
            return result is not None
        else:
            print("⚠️  You can still use the tool with other available models")
            return True

def install_python_dependencies():
    """Install Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    result = run_command("pip install -r requirements.txt", "Installing Python dependencies")
    return result is not None

def test_installation():
    """Test the installation by running a simple probe."""
    print("🧪 Testing installation...")
    
    # Create a simple test script
    test_script = """
import subprocess
import json

def test_ollama():
    try:
        result = subprocess.run(["ollama", "run", "llama3.1:8b", "Hello"], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

if test_ollama():
    print("✅ Ollama test successful")
else:
    print("❌ Ollama test failed")
"""
    
    with open("test_installation.py", "w") as f:
        f.write(test_script)
    
    result = run_command("python test_installation.py", "Testing Ollama connection")
    
    # Clean up test file
    if os.path.exists("test_installation.py"):
        os.remove("test_installation.py")
    
    return result and "✅ Ollama test successful" in result

def create_quick_start_script():
    """Create a quick start script for easy usage."""
    script_content = """#!/usr/bin/env python3
\"\"\"
Quick start script for Truth vs Bias Detector
\"\"\"

import sys
import os

def main():
    print("🔬 Truth vs Bias Detector - Quick Start")
    print("=" * 50)
    
    print("\\nChoose an option:")
    print("1. Quick Ollama Probing (Recommended)")
    print("2. Advanced Neuron Activation Analysis")
    print("3. Web Dashboard")
    print("4. Exit")
    
    choice = input("\\nEnter your choice (1-4): ")
    
    if choice == "1":
        print("\\n🚀 Starting Ollama probing...")
        os.system("python truth_probe.py")
    elif choice == "2":
        print("\\n🧠 Starting neuron activation analysis...")
        os.system("python neuron_activations.py")
    elif choice == "3":
        print("\\n🌐 Starting web dashboard...")
        print("Opening http://localhost:8050 in your browser...")
        os.system("python truth_dashboard.py")
    elif choice == "4":
        print("\\n👋 Goodbye!")
        sys.exit(0)
    else:
        print("\\n❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
"""
    
    with open("quick_start.py", "w") as f:
        f.write(script_content)
    
    # Make it executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("quick_start.py", 0o755)
    
    print("✅ Quick start script created: quick_start.py")

def main():
    """Main setup function."""
    print("🔬 Truth vs Bias Detector - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check Ollama
    if not check_ollama():
        print("\n❌ Setup failed: Ollama issues")
        sys.exit(1)
    
    # Check target model
    if not check_target_model():
        print("\n⚠️  Setup completed with warnings")
    else:
        print("\n✅ Target model check passed")
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("\n❌ Setup failed: Python dependency issues")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("\n⚠️  Setup completed with warnings")
    else:
        print("\n✅ Installation test passed")
    
    # Create quick start script
    create_quick_start_script()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("=" * 50)
    
    print("\n📋 Next steps:")
    print("1. Run the quick start script: python quick_start.py")
    print("2. Or run individual components:")
    print("   - python truth_probe.py (Ollama probing)")
    print("   - python neuron_activations.py (Neuron analysis)")
    print("   - python truth_dashboard.py (Web dashboard)")
    
    print("\n📚 For more information, see README.md")
    print("🔧 For troubleshooting, check the README.md troubleshooting section")

if __name__ == "__main__":
    main() 