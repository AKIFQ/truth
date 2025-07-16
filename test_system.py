#!/usr/bin/env python3
"""
Simple test script for Self-Mapping v5 system
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import json
        print("✅ json imported")
    except ImportError as e:
        print(f"❌ json import failed: {e}")
        return False
    
    try:
        import sqlite3
        print("✅ sqlite3 imported")
    except ImportError as e:
        print(f"❌ sqlite3 import failed: {e}")
        return False
    
    try:
        import random
        print("✅ random imported")
    except ImportError as e:
        print(f"❌ random import failed: {e}")
        return False
    
    try:
        from datetime import datetime, timedelta
        print("✅ datetime imported")
    except ImportError as e:
        print(f"❌ datetime import failed: {e}")
        return False
    
    try:
        import networkx as nx
        print("✅ networkx imported")
    except ImportError as e:
        print(f"❌ networkx import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported")
    except ImportError as e:
        print(f"❌ matplotlib import failed: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print("✅ tqdm imported")
    except ImportError as e:
        print(f"❌ tqdm import failed: {e}")
        return False
    
    try:
        import schedule
        print("✅ schedule imported")
    except ImportError as e:
        print(f"❌ schedule import failed: {e}")
        return False
    
    # Test introspect module
    try:
        sys.path.append('truth')
        from introspect import analyze_activations, prompt_model
        print("✅ introspect module imported")
    except ImportError as e:
        print(f"❌ introspect import failed: {e}")
        return False
    
    return True

def test_database():
    """Test database connectivity."""
    print("\n🗄️ Testing database...")
    
    try:
        import sqlite3
        db_file = "truth/brain_map.db"
        
        if os.path.exists(db_file):
            conn = sqlite3.connect(db_file)
            c = conn.cursor()
            
            # Test basic query
            c.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = c.fetchall()
            print(f"✅ Database connected, found {len(tables)} tables")
            
            conn.close()
            return True
        else:
            print("⚠️ Database file not found, will be created on first run")
            return True
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_directories():
    """Test directory structure."""
    print("\n📁 Testing directories...")
    
    required_dirs = [
        "truth/activations",
        "truth/visuals", 
        "truth/logs"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path} exists")
        else:
            print(f"⚠️ {dir_path} missing, will be created")
    
    return True

def test_ollama():
    """Test Ollama availability."""
    print("\n🤖 Testing Ollama...")
    
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            if 'llama3.1:8b' in result.stdout:
                print("✅ Ollama and llama3.1:8b model are ready")
                return True
            else:
                print("⚠️ Ollama running but llama3.1:8b model not found")
                print("   Run: ollama pull llama3.1:8b")
                return False
        else:
            print("❌ Ollama not responding")
            return False
            
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧠 Self-Mapping v5 System Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_database,
        test_directories,
        test_ollama
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    if all(results):
        print("🎉 All tests passed! System is ready to run.")
        print("\n🚀 To start the system, run:")
        print("   source truth_env/bin/activate && python truth/self_mapper_v5.py")
        print("\n📊 To start the dashboard, run:")
        print("   source truth_env/bin/activate && python truth/dashboard_v5.py")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        print("\n🔧 Common fixes:")
        print("   - Install missing packages: pip install <package>")
        print("   - Pull Ollama model: ollama pull llama3.1:8b")
        print("   - Activate virtual environment: source truth_env/bin/activate")

if __name__ == "__main__":
    main() 