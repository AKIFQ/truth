#!/usr/bin/env python3
"""
Launcher for Self-Mapping Loop v5: Full Autonomous Mapping System
"""

import os
import sys
import subprocess
import time
import threading

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'dash', 'dash-bootstrap-components', 'plotly', 'pandas', 
        'networkx', 'matplotlib', 'tqdm', 'schedule'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_ollama():
    """Check if Ollama is running and model is available."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and 'llama3.1:8b' in result.stdout:
            print("✅ Ollama and llama3.1:8b model are ready")
            return True
        else:
            print("❌ llama3.1:8b model not found. Run: ollama pull llama3.1:8b")
            return False
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        return False

def run_dashboard():
    """Run the dashboard in a separate thread."""
    print("🚀 Starting Dashboard v5...")
    try:
        subprocess.run([sys.executable, 'truth/dashboard_v5.py'], check=True)
    except KeyboardInterrupt:
        print("📊 Dashboard stopped")

def main():
    """Main launcher function."""
    print("🧠 Self-Mapping Loop v5: Full Autonomous Mapping System")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check Ollama
    if not check_ollama():
        return
    
    print("\n🎯 System ready! Choose an option:")
    print("1. 🚀 Run Autonomous Mapping (6-8 hours)")
    print("2. 📊 Start Dashboard Only")
    print("3. 🔄 Run Both (Mapping + Dashboard)")
    print("4. 🧪 Test Mode (5 mappings)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting Autonomous Mapping...")
        print("💡 Press Ctrl+C to stop and save progress")
        subprocess.run([sys.executable, 'truth/self_mapper_v5.py'])
        
    elif choice == "2":
        print("\n📊 Starting Dashboard...")
        subprocess.run([sys.executable, 'truth/dashboard_v5.py'])
        
    elif choice == "3":
        print("\n🔄 Starting both systems...")
        print("📊 Dashboard will be available at: http://127.0.0.1:8050")
        
        # Start dashboard in background
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        
        # Wait a moment for dashboard to start
        time.sleep(3)
        
        # Start mapping
        print("🚀 Starting Autonomous Mapping...")
        subprocess.run([sys.executable, 'truth/self_mapper_v5.py'])
        
    elif choice == "4":
        print("\n🧪 Running Test Mode (5 mappings)...")
        # Create a test version that only runs 5 mappings
        test_code = '''
import sys
sys.path.append('truth')
from self_mapper_v5 import AutonomousMapper

mapper = AutonomousMapper()
for i in range(5):
    print(f"\\n🧪 Test mapping {i+1}/5")
    mapper.run_single_mapping()
    if not mapper.coverage_tracker.should_continue():
        break

print("\\n✅ Test complete!")
'''
        with open('test_v5.py', 'w') as f:
            f.write(test_code)
        
        subprocess.run([sys.executable, 'test_v5.py'])
        os.remove('test_v5.py')
        
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main() 