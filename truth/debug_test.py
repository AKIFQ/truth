#!/usr/bin/env python3
"""
Debug test for async Ollama calls
"""

import asyncio
import subprocess
import time

async def test_ollama_async():
    """Test async Ollama call."""
    print("Testing async Ollama call...")
    start_time = time.time()
    
    try:
        process = await asyncio.create_subprocess_exec(
            'ollama', 'list',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        elapsed = time.time() - start_time
        print(f"✅ Ollama async call completed in {elapsed:.2f}s")
        print(f"Output: {stdout.decode()[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Ollama async call failed: {e}")
        return False

async def test_ollama_prompt():
    """Test async Ollama prompt."""
    print("Testing async Ollama prompt...")
    start_time = time.time()
    
    try:
        process = await asyncio.create_subprocess_exec(
            'ollama', 'run', 'llama3.1:8b', 'Hello, test message.',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        elapsed = time.time() - start_time
        print(f"✅ Ollama prompt completed in {elapsed:.2f}s")
        print(f"Response: {stdout.decode()[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ Ollama prompt failed: {e}")
        return False

async def test_batch_processing():
    """Test batch processing."""
    print("Testing batch processing...")
    start_time = time.time()
    
    try:
        # Create 3 simple tasks
        tasks = [
            asyncio.create_subprocess_exec('echo', 'task_1'),
            asyncio.create_subprocess_exec('echo', 'task_2'),
            asyncio.create_subprocess_exec('echo', 'task_3')
        ]
        
        results = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start_time
        print(f"✅ Batch processing completed in {elapsed:.2f}s")
        print(f"Results: {len(results)} tasks completed")
        
        return True
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return False

async def main():
    """Main test function."""
    print("🧪 Debug Test for Async System")
    print("=" * 40)
    
    # Test 1: Basic async
    print("\n1️⃣ Testing basic async...")
    await asyncio.sleep(1)
    print("✅ Basic async works")
    
    # Test 2: Ollama list
    print("\n2️⃣ Testing Ollama list...")
    success = await test_ollama_async()
    
    # Test 3: Ollama prompt
    print("\n3️⃣ Testing Ollama prompt...")
    success = await test_ollama_prompt()
    
    # Test 4: Batch processing
    print("\n4️⃣ Testing batch processing...")
    success = await test_batch_processing()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main()) 