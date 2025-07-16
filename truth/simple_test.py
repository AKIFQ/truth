#!/usr/bin/env python3
"""
Simple test to isolate the hanging issue
"""

import asyncio
import subprocess
import time
import os

async def test_async_prompt():
    """Test async prompt call."""
    print("Testing async prompt...")
    start_time = time.time()
    
    try:
        process = await asyncio.create_subprocess_exec(
            'ollama', 'run', 'llama3.1:8b', 'Hello, test message.',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        elapsed = time.time() - start_time
        print(f"✅ Async prompt completed in {elapsed:.2f}s")
        print(f"Response: {stdout.decode()[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Async prompt failed: {e}")
        return False

async def test_analyze_activations():
    """Test analyze_activations function."""
    print("Testing analyze_activations...")
    start_time = time.time()
    
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from introspect import analyze_activations
        activations = analyze_activations("test prompt", "llama3.1:8b")
        
        elapsed = time.time() - start_time
        print(f"✅ analyze_activations completed in {elapsed:.2f}s")
        print(f"Activations: {len(activations)} values")
        return True
    except Exception as e:
        print(f"❌ analyze_activations failed: {e}")
        return False

async def test_single_mapping():
    """Test a single mapping step."""
    print("Testing single mapping...")
    start_time = time.time()
    
    try:
        # Step 1: Get prompt
        prompt = "What is the truth about climate change?"
        print(f"Prompt: {prompt}")
        
        # Step 2: Analyze activations
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from introspect import analyze_activations
        activations = analyze_activations(prompt, "llama3.1:8b")
        print(f"Activations: {len(activations)} values")
        
        # Step 3: Get top neurons
        import numpy as np
        top_neurons = list(np.argsort(activations)[::-1][:5])
        print(f"Top neurons: {top_neurons}")
        
        # Step 4: Test async prompt for one neuron
        neuron_idx = top_neurons[0]
        activation = activations[neuron_idx]
        
        process = await asyncio.create_subprocess_exec(
            'ollama', 'run', 'llama3.1:8b', 
            f"Neuron {neuron_idx}, Activation {activation:.4f}, Concept:confidence format only (e.g., truth_detection:0.85). Return ONLY a short concept and confidence, no explanations.",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        elapsed = time.time() - start_time
        print(f"✅ Single mapping completed in {elapsed:.2f}s")
        print(f"Response: {stdout.decode()[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Single mapping failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🧪 Simple Async Test")
    print("=" * 40)
    
    # Test 1: Basic async prompt
    print("\n1️⃣ Testing basic async prompt...")
    success1 = await test_async_prompt()
    
    # Test 2: analyze_activations
    print("\n2️⃣ Testing analyze_activations...")
    success2 = await test_analyze_activations()
    
    # Test 3: Single mapping
    print("\n3️⃣ Testing single mapping...")
    success3 = await test_single_mapping()
    
    print(f"\n✅ All tests completed!")
    print(f"Results: {success1}, {success2}, {success3}")

if __name__ == "__main__":
    asyncio.run(main()) 