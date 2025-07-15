#!/usr/bin/env python3
"""
Simple test script to verify Ollama is working with the target model.
"""

import subprocess
import sys

def test_ollama_connection():
    """Test if Ollama is running and accessible."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ollama is running and accessible")
            return True
        else:
            print("❌ Ollama is not responding properly")
            return False
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False

def test_model_availability(model_name="llama3.1:8b"):
    """Test if the target model is available."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and model_name in result.stdout:
            print(f"✅ Model {model_name} is available")
            return True
        else:
            print(f"❌ Model {model_name} not found")
            print("Available models:")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"❌ Error checking model availability: {e}")
        return False

def test_simple_query(model_name="llama3.1:8b"):
    """Test a simple query to the model."""
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, "Hello, how are you?"], 
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and len(result.stdout.strip()) > 0:
            print("✅ Model query test successful")
            print(f"Response: {result.stdout.strip()[:100]}...")
            return True
        else:
            print("❌ Model query test failed")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error during model query test: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Ollama Setup")
    print("=" * 30)
    
    # Test 1: Ollama connection
    if not test_ollama_connection():
        print("\n❌ Ollama setup test failed")
        print("Please ensure Ollama is installed and running:")
        print("1. Install Ollama: https://ollama.ai/download")
        print("2. Start Ollama: ollama serve")
        sys.exit(1)
    
    # Test 2: Model availability
    if not test_model_availability():
        print("\n⚠️  Target model not found")
        print("You can still use the tool with other available models")
        print("Or pull the target model with: ollama pull llama3.1:8b")
    
    # Test 3: Simple query
    if not test_simple_query():
        print("\n❌ Model query test failed")
        sys.exit(1)
    
    print("\n" + "=" * 30)
    print("🎉 All tests passed!")
    print("✅ Ollama is ready for Truth vs Bias Detector")
    print("\nNext steps:")
    print("1. Run: python example.py")
    print("2. Run: python truth_probe.py")
    print("3. Run: python truth_dashboard.py")

if __name__ == "__main__":
    main() 