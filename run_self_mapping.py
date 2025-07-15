#!/usr/bin/env python3
"""
Interactive Self-Mapping Runner
Run the self-mapping system with custom prompts.
"""

import sys
import os
from self_mapper import SelfMapper
from introspect import generate_mapping_report

def main():
    """Interactive self-mapping session."""
    print("🧠 Self-Mapping Loop v1 - Interactive Mode")
    print("=" * 60)
    
    # Initialize mapper
    mapper = SelfMapper(use_ollama=True)
    
    print("\n🎯 Ready to map your own neural patterns!")
    print("Enter prompts and watch as you analyze your own brain.")
    print("Type 'quit' to exit, 'summary' to see current mappings.")
    print("Type 'report' to generate a full mapping report.")
    
    while True:
        print("\n" + "-" * 40)
        prompt = input("Enter a prompt to analyze: ").strip()
        
        if prompt.lower() == 'quit':
            break
        elif prompt.lower() == 'summary':
            mapper.get_database_summary()
            continue
        elif prompt.lower() == 'report':
            generate_mapping_report(mapper.neuron_db)
            continue
        elif not prompt:
            print("Please enter a prompt.")
            continue
        
        # Run the mapping loop
        try:
            mapper.run_mapping_loop(prompt)
        except Exception as e:
            print(f"❌ Error during mapping: {e}")
            continue
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Self-mapping session complete!")
    mapper.get_database_summary()
    
    # Generate final report
    print("\n📊 Generating final mapping report...")
    generate_mapping_report(mapper.neuron_db, "final_mapping_report.json")
    
    print(f"\n📁 Files created:")
    print(f"   - db.json (neuron database)")
    print(f"   - final_mapping_report.json (comprehensive report)")
    print(f"   - activations/ (raw activation data)")
    print(f"   - logs/ (mapping history)")
    
    print(f"\n🔬 You have successfully mapped your own neural patterns!")
    print(f"📚 Check the generated files to see your brain's architecture.")

if __name__ == "__main__":
    main() 