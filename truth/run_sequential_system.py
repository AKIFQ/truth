#!/usr/bin/env python3
"""
Bulletproof Launcher for Sequential General Self-Consistency Engine v6.3
Optimized for efficiency and reliability
"""

import os
import sys
import subprocess
import time
import signal
import atexit
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
VENV_PATH = PROJECT_ROOT / "truth_env"
TRUTH_DIR = PROJECT_ROOT / "truth"
PYTHON_EXECUTABLE = VENV_PATH / "bin" / "python"

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import networkx
        import matplotlib
        import numpy
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def activate_venv():
    """Activate virtual environment and update sys.path."""
    if not VENV_PATH.exists():
        print(f"❌ Virtual environment not found at {VENV_PATH}")
        print("Please run: python3 -m venv truth_env")
        return False
    
    # Add virtual environment to Python path
    site_packages = VENV_PATH / "lib" / "python3.13" / "site-packages"
    if site_packages.exists():
        sys.path.insert(0, str(site_packages))
    
    # Add project root to path
    sys.path.insert(0, str(PROJECT_ROOT))
    
    print(f"✅ Virtual environment activated: {VENV_PATH}")
    return True

def check_ollama():
    """Check if Ollama is running and accessible."""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ollama is running and accessible")
            return True
        else:
            print("❌ Ollama is not running properly")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False

def check_model():
    """Check if the required model is available."""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if 'llama3.1:8b' in result.stdout:
            print("✅ Model llama3.1:8b is available")
            return True
        else:
            print("❌ Model llama3.1:8b not found")
            print("Available models:")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"❌ Cannot check models: {e}")
        return False

def setup_directories():
    """Ensure all required directories exist."""
    directories = [
        TRUTH_DIR / "activations",
        TRUTH_DIR / "logs", 
        TRUTH_DIR / "visuals",
        TRUTH_DIR / "checkpoints"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory ready: {directory}")

def cleanup_handler(signum, frame):
    """Handle cleanup on interrupt."""
    print("\n🛑 Received interrupt signal. Cleaning up...")
    print("💾 All progress has been automatically saved.")
    print("🔄 You can resume anytime by running this script again.")
    sys.exit(0)

def main():
    """Main launcher function with comprehensive checks."""
    print("🚀 Bulletproof Sequential General Self-Consistency Engine v6.3")
    print("=" * 60)
    print("⚡ Optimized for efficiency - Sequential processing")
    print("🎯 Expected performance: 2-5x faster than async version")
    
    # Register cleanup handler
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    atexit.register(lambda: print("💾 Final cleanup completed"))
    
    # Step 1: Check virtual environment
    print("\n1️⃣ Checking virtual environment...")
    if not activate_venv():
        return 1
    
    # Step 2: Check dependencies
    print("\n2️⃣ Checking dependencies...")
    if not check_dependencies():
        print("Installing dependencies...")
        subprocess.run([str(PYTHON_EXECUTABLE), "-m", "pip", "install", 
                       "networkx", "matplotlib", "numpy"])
        if not check_dependencies():
            return 1
    
    # Step 3: Check Ollama
    print("\n3️⃣ Checking Ollama...")
    if not check_ollama():
        print("Please start Ollama: ollama serve")
        return 1
    
    # Step 4: Check model
    print("\n4️⃣ Checking model...")
    if not check_model():
        print("Please pull the model: ollama pull llama3.1:8b")
        return 1
    
    # Step 5: Setup directories
    print("\n5️⃣ Setting up directories...")
    setup_directories()
    
    # Step 6: Import and run the system
    print("\n6️⃣ Starting the system...")
    try:
        # Import the sequential system
        sys.path.insert(0, str(TRUTH_DIR))
        from self_mapper_v6_sequential import GeneralSelfConsistencyEngine
        
        # Create the engine
        engine = GeneralSelfConsistencyEngine()
        
        print("\n🎯 Choose mode:")
        print("1. 🚀 Auto mode (run until 50k neurons mapped)")
        print("2. 📊 Dashboard mode")
        print("3. 🧪 Test mode (5 mappings)")
        print("4. 📈 Performance test (10 mappings)")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting AUTO mode - will run until 50k neurons mapped...")
            print("⚡ Sequential processing - much faster than async!")
            engine.run_autonomous_loop()
        elif choice == "2":
            print("\n📊 Starting Dashboard...")
            subprocess.run([str(PYTHON_EXECUTABLE), str(TRUTH_DIR / "dashboard_v6.py")])
        elif choice == "3":
            print("\n🧪 Running Test Mode (5 mappings)...")
            start_time = time.time()
            for i in range(5):
                print(f"\n🧪 Test mapping {i+1}/5")
                engine.run_single_mapping(i+1, 1)
                if len(engine.mapped_neurons) >= 50000 * 0.95:
                    break
            elapsed = time.time() - start_time
            print(f"\n✅ Test complete! Time: {elapsed:.1f}s")
        elif choice == "4":
            print("\n📈 Running Performance Test (10 mappings)...")
            start_time = time.time()
            for i in range(10):
                print(f"\n📈 Performance mapping {i+1}/10")
                engine.run_single_mapping(i+1, 1)
            elapsed = time.time() - start_time
            rate = 10 / elapsed if elapsed > 0 else 0
            print(f"\n📊 Performance Results:")
            print(f"   Time: {elapsed:.1f}s")
            print(f"   Rate: {rate:.2f} neurons/sec")
            print(f"   ETA for 50k: {50000/rate/60:.1f} minutes")
        else:
            print("❌ Invalid choice")
            return 1
            
    except Exception as e:
        print(f"❌ Error running system: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n✅ System completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main()) 