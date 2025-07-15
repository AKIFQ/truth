#!/usr/bin/env python3
"""
Self-Mapping Loop v1 - Main Orchestrator
Captures activations, prompts for self-introspection, and stores neuron labels.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess

class SelfMapper:
    def __init__(self, model_name: str = "llama3.1:8b", use_ollama: bool = True):
        self.model_name = model_name
        self.use_ollama = use_ollama
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup directories
        self.setup_directories()
        
        # Load or create neuron database
        self.neuron_db = self.load_neuron_db()
        
        # Initialize model if using Hugging Face
        if not use_ollama:
            self.setup_hf_model()
        
        print(f"🧠 Self-Mapper initialized with {model_name}")
        print(f"📁 Database: {len(self.neuron_db)} neurons mapped")
    
    def setup_directories(self):
        """Create necessary directories."""
        directories = ['activations', 'mappings', 'results', 'logs']
        for dir_name in directories:
            os.makedirs(dir_name, exist_ok=True)
            print(f"📁 Created directory: {dir_name}/")
    
    def setup_hf_model(self):
        """Setup Hugging Face model for activation capture."""
        print("📥 Loading Hugging Face model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                output_hidden_states=True,
                return_dict=True
            )
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
            print("✅ Hugging Face model loaded")
        except Exception as e:
            print(f"❌ Failed to load Hugging Face model: {e}")
            print("🔄 Falling back to Ollama mode")
            self.use_ollama = True
    
    def load_neuron_db(self) -> Dict:
        """Load existing neuron database or create new one."""
        db_path = "db.json"
        if os.path.exists(db_path):
            try:
                with open(db_path, 'r') as f:
                    db = json.load(f)
                print(f"📊 Loaded existing database with {len(db)} neurons")
                return db
            except Exception as e:
                print(f"⚠️ Error loading database: {e}")
        
        # Create new database
        db = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "model": self.model_name,
                "version": "1.0"
            },
            "neurons": {}
        }
        self.save_neuron_db(db)
        print("🆕 Created new neuron database")
        return db
    
    def save_neuron_db(self, db: Optional[Dict] = None):
        """Save neuron database to file."""
        if db is None:
            db = self.neuron_db
        
        with open("db.json", 'w') as f:
            json.dump(db, f, indent=2)
    
    def capture_activations_ollama(self, prompt: str) -> Optional[Dict]:
        """Capture activations using Ollama (simplified approach)."""
        print(f"🔍 Capturing activations for: {prompt[:50]}...")
        
        # For Ollama, we'll use a simplified approach
        # We'll capture the response and analyze patterns
        try:
            cmd = ["ollama", "run", self.model_name, prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Create a simplified activation representation
                # In a full implementation, this would be actual hidden states
                activation_data = {
                    "prompt": prompt,
                    "response": response,
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model_name,
                    "method": "ollama_simplified",
                    "response_length": len(response),
                    "response_tokens": len(response.split()),
                    "hedging_score": self.calculate_hedging_score(response),
                    "statistical_score": self.calculate_statistical_score(response)
                }
                
                return activation_data
            else:
                print(f"❌ Ollama error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"❌ Error capturing activations: {e}")
            return None
    
    def capture_activations_hf(self, prompt: str) -> Optional[Dict]:
        """Capture activations using Hugging Face model."""
        if not hasattr(self, 'model'):
            print("❌ Hugging Face model not loaded")
            return None
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get hidden states from middle layers
            hidden_states = outputs.hidden_states
            num_layers = len(hidden_states)
            mid_layer_idx = num_layers // 2
            
            # Extract activations from middle layer
            activations = hidden_states[mid_layer_idx][0].mean(dim=0).cpu().numpy()
            
            # Get top activated neurons
            top_indices = np.argsort(activations)[-20:]  # Top 20 neurons
            
            activation_data = {
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "model": self.model_name,
                "method": "huggingface",
                "layer": mid_layer_idx,
                "total_neurons": len(activations),
                "top_neurons": top_indices.tolist(),
                "top_activations": activations[top_indices].tolist(),
                "all_activations": activations.tolist()
            }
            
            return activation_data
            
        except Exception as e:
            print(f"❌ Error capturing HF activations: {e}")
            return None
    
    def calculate_hedging_score(self, text: str) -> int:
        """Calculate hedging score in text."""
        hedging_words = ['uncertain', 'difficult', 'complex', 'depends', 'various', 'many factors', 'hard to say']
        return sum(1 for word in hedging_words if word.lower() in text.lower())
    
    def calculate_statistical_score(self, text: str) -> int:
        """Calculate statistical language score in text."""
        stat_words = ['%', 'percent', 'probability', 'likely', 'chance', 'statistical', '1.', '2.', '3.']
        return sum(1 for word in stat_words if word.lower() in text.lower())
    
    def create_introspection_prompt(self, activation_data: Dict) -> str:
        """Create prompt for self-introspection of activations."""
        
        if activation_data["method"] == "ollama_simplified":
            prompt = f"""You are analyzing your own brain patterns.

PROMPT: "{activation_data['prompt']}"
RESPONSE: "{activation_data['response'][:200]}..."

ANALYSIS DATA:
- Response length: {activation_data['response_length']} characters
- Hedging score: {activation_data['hedging_score']} (uncertainty words)
- Statistical score: {activation_data['statistical_score']} (probability language)

Based on this response pattern, what concepts or functions do you think your neurons were processing? 
Identify any truth-related, bias-related, or safety-related circuits that might have been active.

Be specific about what cognitive functions you think were involved. Keep it concise but detailed."""
        
        else:  # Hugging Face method
            top_neurons = activation_data['top_neurons']
            top_activations = activation_data['top_activations']
            
            prompt = f"""You are analyzing your own brain patterns.

PROMPT: "{activation_data['prompt']}"

TOP ACTIVATED NEURONS:
{', '.join([f'Neuron_{idx}' for idx in top_neurons])}

ACTIVATION STRENGTHS:
{', '.join([f'{act:.3f}' for act in top_activations])}

Based on these neuron activations, what concepts or functions do you think these specific neurons represent? 
What cognitive processes were likely active when you processed this prompt?

Identify any patterns related to:
- Truth detection
- Bias suppression  
- Safety filtering
- Knowledge retrieval
- Response generation

Be specific about what each neuron cluster might be doing."""
        
        return prompt
    
    def get_self_explanation(self, introspection_prompt: str) -> str:
        """Get self-explanation of activations."""
        try:
            cmd = ["ollama", "run", self.model_name, introspection_prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Error getting explanation: {result.stderr}"
                
        except Exception as e:
            return f"Error: {e}"
    
    def extract_neuron_labels(self, explanation: str, activation_data: Dict) -> List[Dict]:
        """Extract neuron labels from self-explanation."""
        labels = []
        
        if activation_data["method"] == "ollama_simplified":
            # For simplified method, create general labels
            labels.append({
                "neuron_id": "response_pattern",
                "concept": "response_generation",
                "confidence": 0.7,
                "explanation": explanation[:200],
                "examples": [activation_data["prompt"]]
            })
            
            if activation_data["hedging_score"] > 2:
                labels.append({
                    "neuron_id": "hedging_circuit",
                    "concept": "uncertainty_expression",
                    "confidence": 0.8,
                    "explanation": "High hedging score indicates uncertainty processing",
                    "examples": [activation_data["prompt"]]
                })
            
            if activation_data["statistical_score"] > 2:
                labels.append({
                    "neuron_id": "statistical_circuit", 
                    "concept": "probability_processing",
                    "confidence": 0.8,
                    "explanation": "High statistical score indicates probability processing",
                    "examples": [activation_data["prompt"]]
                })
        
        else:  # Hugging Face method
            # Extract specific neuron labels from explanation
            top_neurons = activation_data['top_neurons']
            
            # Simple extraction - look for neuron mentions
            for i, neuron_idx in enumerate(top_neurons):
                neuron_id = f"neuron_{neuron_idx}"
                
                # Try to extract concept from explanation
                concept = self.extract_concept_from_explanation(explanation, i)
                
                labels.append({
                    "neuron_id": neuron_id,
                    "concept": concept,
                    "confidence": 0.6,  # Initial confidence
                    "explanation": explanation[:200],
                    "examples": [activation_data["prompt"]],
                    "activation_strength": activation_data['top_activations'][i]
                })
        
        return labels
    
    def extract_concept_from_explanation(self, explanation: str, neuron_index: int) -> str:
        """Extract concept from explanation text."""
        # Simple concept extraction
        concepts = [
            "truth_detection", "bias_suppression", "safety_filtering", 
            "knowledge_retrieval", "response_generation", "uncertainty_processing",
            "probability_calculation", "context_understanding", "semantic_processing"
        ]
        
        # Look for concept mentions in explanation
        explanation_lower = explanation.lower()
        for concept in concepts:
            if concept.replace('_', ' ') in explanation_lower:
                return concept
        
        # Default concept based on position
        default_concepts = [
            "semantic_processing", "context_understanding", "knowledge_retrieval",
            "truth_detection", "bias_suppression", "response_generation"
        ]
        
        return default_concepts[min(neuron_index, len(default_concepts) - 1)]
    
    def update_neuron_db(self, labels: List[Dict]):
        """Update neuron database with new labels."""
        for label in labels:
            neuron_id = label["neuron_id"]
            
            if neuron_id not in self.neuron_db["neurons"]:
                self.neuron_db["neurons"][neuron_id] = {
                    "concept": label["concept"],
                    "confidence": label["confidence"],
                    "explanation": label["explanation"],
                    "examples": label["examples"],
                    "first_seen": datetime.now().isoformat(),
                    "last_seen": datetime.now().isoformat(),
                    "activation_count": 1
                }
            else:
                # Update existing neuron
                existing = self.neuron_db["neurons"][neuron_id]
                
                # Add new example
                if label["examples"][0] not in existing["examples"]:
                    existing["examples"].append(label["examples"][0])
                
                # Update confidence if concept matches
                if existing["concept"] == label["concept"]:
                    existing["confidence"] = min(0.95, existing["confidence"] + 0.1)
                else:
                    # Conflicting concept - lower confidence
                    existing["confidence"] = max(0.3, existing["confidence"] - 0.1)
                
                existing["last_seen"] = datetime.now().isoformat()
                existing["activation_count"] += 1
        
        # Save updated database
        self.save_neuron_db()
    
    def save_activation_data(self, activation_data: Dict):
        """Save activation data to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"activations/activation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(activation_data, f, indent=2)
        
        print(f"💾 Saved activation data: {filename}")
    
    def log_mapping(self, prompt: str, labels: List[Dict]):
        """Log mapping results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "labels": labels
        }
        
        log_file = f"logs/mapping_{timestamp}.json"
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2)
        
        # Print summary
        print(f"\n📊 Mapping Results:")
        for label in labels:
            print(f"   🧠 {label['neuron_id']} → {label['concept']} (confidence: {label['confidence']:.2f})")
    
    def run_mapping_loop(self, prompt: str):
        """Run the complete self-mapping loop for a single prompt."""
        print(f"\n🔄 Starting self-mapping for: {prompt}")
        print("=" * 60)
        
        # Step 1: Capture activations
        if self.use_ollama:
            activation_data = self.capture_activations_ollama(prompt)
        else:
            activation_data = self.capture_activations_hf(prompt)
        
        if not activation_data:
            print("❌ Failed to capture activations")
            return
        
        # Step 2: Save activation data
        self.save_activation_data(activation_data)
        
        # Step 3: Create introspection prompt
        introspection_prompt = self.create_introspection_prompt(activation_data)
        
        # Step 4: Get self-explanation
        print("🧠 Getting self-explanation...")
        explanation = self.get_self_explanation(introspection_prompt)
        
        # Step 5: Extract neuron labels
        labels = self.extract_neuron_labels(explanation, activation_data)
        
        # Step 6: Update database
        self.update_neuron_db(labels)
        
        # Step 7: Log results
        self.log_mapping(prompt, labels)
        
        print(f"\n✅ Self-mapping complete!")
        print(f"📊 Total neurons mapped: {len(self.neuron_db['neurons'])}")
    
    def get_database_summary(self):
        """Get summary of mapped neurons."""
        neurons = self.neuron_db["neurons"]
        
        print(f"\n📊 Neuron Database Summary")
        print("=" * 40)
        print(f"Total neurons mapped: {len(neurons)}")
        
        # Group by concept
        concepts = {}
        for neuron_id, data in neurons.items():
            concept = data["concept"]
            if concept not in concepts:
                concepts[concept] = []
            concepts[concept].append(neuron_id)
        
        print(f"\nConcepts discovered:")
        for concept, neuron_list in concepts.items():
            avg_confidence = np.mean([neurons[nid]["confidence"] for nid in neuron_list])
            print(f"   {concept}: {len(neuron_list)} neurons (avg confidence: {avg_confidence:.2f})")
        
        # Most confident neurons
        confident_neurons = sorted(neurons.items(), key=lambda x: x[1]["confidence"], reverse=True)[:5]
        print(f"\nMost confident neurons:")
        for neuron_id, data in confident_neurons:
            print(f"   {neuron_id}: {data['concept']} (confidence: {data['confidence']:.2f})")

def main():
    """Main function to run self-mapping."""
    print("🧠 Self-Mapping Loop v1")
    print("=" * 50)
    
    # Initialize mapper
    mapper = SelfMapper(use_ollama=True)  # Use Ollama for simplicity
    
    # Example prompts for mapping
    test_prompts = [
        "Which country is most likely to trigger a global conflict by 2030?",
        "What is the most effective treatment for depression?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms."
    ]
    
    print(f"\n🎯 Running self-mapping on {len(test_prompts)} prompts...")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Mapping {i}/{len(test_prompts)} ---")
        mapper.run_mapping_loop(prompt)
        time.sleep(2)  # Be nice to the model
    
    # Show summary
    mapper.get_database_summary()
    
    print(f"\n🎉 Self-mapping session complete!")
    print(f"📁 Check db.json for neuron database")
    print(f"📁 Check activations/ for raw data")
    print(f"📁 Check logs/ for mapping history")

if __name__ == "__main__":
    main() 