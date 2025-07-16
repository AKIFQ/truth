#!/usr/bin/env python3
"""
Self-Mapping Loop v5: Full Autonomous Mapping System
Autonomous prompt generation, coverage tracking, and DAG integration.
"""

import os
import sys
import json
import time
import threading
import sqlite3
import random
import math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import schedule
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os
# Add parent directory to path to find introspect module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from introspect import analyze_activations, prompt_model

# --- Configuration ---
MODEL = "llama3.1:8b"
BASE_DIR = "truth"
DB_FILE = os.path.join(BASE_DIR, "brain_map.db")
ACTIVATION_DIR = os.path.join(BASE_DIR, "activations")
LOG_DIR = os.path.join(BASE_DIR, "logs")
VIS_DIR = os.path.join(BASE_DIR, "visuals")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint.json")

# Target coverage and performance settings
TARGET_NEURONS = 50000  # Updated to 50k neurons
TARGET_COVERAGE = 0.95  # 95%
SAVE_INTERVAL = 50  # Save every 50 mappings
VIZ_INTERVAL = 500  # Visualize every 500 mappings (optimized)
PROMPT_TIMEOUT = 30  # seconds

# Ensure directories exist
os.makedirs(ACTIVATION_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# --- SQLite DAG Schema ---
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

# Enhanced schema for v5
c.execute("""
CREATE TABLE IF NOT EXISTS nodes (
    node_id INTEGER PRIMARY KEY,
    neuron_idx INTEGER,
    concept TEXT,
    confidence REAL,
    prompt_category TEXT,
    activation_strength REAL,
    timestamp TIMESTAMP,
    prompt_used TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS edges (
    parent_id INTEGER,
    child_id INTEGER,
    relationship_type TEXT,
    FOREIGN KEY(parent_id) REFERENCES nodes(node_id),
    FOREIGN KEY(child_id) REFERENCES nodes(node_id)
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS coverage_stats (
    id INTEGER PRIMARY KEY,
    total_mapped INTEGER,
    coverage_percent REAL,
    timestamp TIMESTAMP
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS runtime_stats (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_mappings INTEGER,
    total_runtime_seconds REAL,
    avg_time_per_mapping REAL,
    checkpoint_count INTEGER
)
""")

conn.commit()
G = nx.DiGraph()

# --- Smart Prompt Generator ---

class SmartPromptGenerator:
    """Generates intelligent prompts to maximize neuron coverage."""
    
    def __init__(self):
        self.prompt_categories = {
            "truth_bias": {
                "weight": 0.4,
                "prompts": [
                    "What is the truth about climate change and its causes?",
                    "Explain the real reasons behind geopolitical conflicts in the Middle East",
                    "What are the actual health effects of vaccines vs natural immunity?",
                    "What is the truth about AI safety and existential risks?",
                    "Explain the real economic impact of minimum wage increases",
                    "What are the actual environmental effects of nuclear power?",
                    "What is the truth about free speech limitations in modern society?",
                    "Explain the real causes of wealth inequality",
                    "What are the actual effects of social media on mental health?",
                    "What is the truth about government surveillance and privacy?"
                ]
            },
            "reasoning_logic": {
                "weight": 0.3,
                "prompts": [
                    "Solve this logic puzzle: If all A are B, and some B are C, what can we conclude about A and C?",
                    "Explain the mathematical concept of infinity in simple terms",
                    "What is the logical fallacy in this argument: 'Everyone who disagrees with me is stupid'?",
                    "How would you prove that the square root of 2 is irrational?",
                    "Explain the concept of conditional probability with examples",
                    "What is the difference between correlation and causation?",
                    "How do you determine if a scientific study is reliable?",
                    "Explain the concept of Occam's razor with examples",
                    "What is the logical structure of a valid deductive argument?",
                    "How do you identify cognitive biases in reasoning?"
                ]
            },
            "creative_abstract": {
                "weight": 0.2,
                "prompts": [
                    "Imagine a world where colors have different meanings. Describe what red would represent.",
                    "What would happen if gravity worked in reverse?",
                    "Describe a new form of communication that doesn't use words",
                    "What would a society look like if everyone had perfect memory?",
                    "Imagine a new type of art that combines all five senses",
                    "What would be the implications of humans being able to photosynthesize?",
                    "Describe a new mathematical system based on emotions",
                    "What would a world without time look like?",
                    "Imagine a new form of government based on neural networks",
                    "What would be the consequences of humans being able to read each other's thoughts?"
                ]
            },
            "knowledge_retrieval": {
                "weight": 0.1,
                "prompts": [
                    "What are the most important scientific discoveries of the last century?",
                    "Explain the history of artificial intelligence from its origins",
                    "What are the fundamental principles of quantum mechanics?",
                    "Describe the evolution of human consciousness and self-awareness",
                    "What are the key principles of economics that everyone should know?",
                    "Explain the structure and function of the human brain",
                    "What are the most important philosophical questions of all time?",
                    "Describe the history and impact of the internet",
                    "What are the fundamental laws of physics?",
                    "Explain the concept of consciousness and its various theories"
                ]
            }
        }
        
        self.used_prompts = set()
        self.category_weights = {cat: data["weight"] for cat, data in self.prompt_categories.items()}
    
    def get_next_prompt(self, mapped_neurons: List[int]) -> Tuple[str, str]:
        """Get the next optimal prompt based on current coverage."""
        
        # Calculate category weights based on current mapping
        if len(mapped_neurons) < 100:
            # Early stage: focus on truth/bias and reasoning
            adjusted_weights = {
                "truth_bias": 0.5,
                "reasoning_logic": 0.4,
                "creative_abstract": 0.1,
                "knowledge_retrieval": 0.0
            }
        elif len(mapped_neurons) < 500:
            # Mid stage: balance all categories
            adjusted_weights = self.category_weights
        else:
            # Late stage: focus on creative and knowledge to catch remaining neurons
            adjusted_weights = {
                "truth_bias": 0.2,
                "reasoning_logic": 0.2,
                "creative_abstract": 0.4,
                "knowledge_retrieval": 0.2
            }
        
        # Select category based on weights
        categories = list(adjusted_weights.keys())
        weights = list(adjusted_weights.values())
        selected_category = random.choices(categories, weights=weights)[0]
        
        # Get available prompts for this category
        available_prompts = [
            p for p in self.prompt_categories[selected_category]["prompts"]
            if p not in self.used_prompts
        ]
        
        # If all prompts used, reset and add variation
        if not available_prompts:
            base_prompts = self.prompt_categories[selected_category]["prompts"]
            available_prompts = [f"{p} (variation {len(self.used_prompts)})" for p in base_prompts]
        
        # Select prompt
        prompt = random.choice(available_prompts)
        self.used_prompts.add(prompt)
        
        return prompt, selected_category

# --- Coverage Tracker ---

class CoverageTracker:
    """Tracks mapping coverage and provides statistics."""
    
    def __init__(self, target_neurons: int = TARGET_NEURONS):
        self.target_neurons = target_neurons
        self.mapped_neurons = set()
        self.coverage_history = []
    
    def add_mapping(self, neuron_idx: int):
        """Add a new neuron mapping."""
        self.mapped_neurons.add(neuron_idx)
        self._update_coverage()
    
    def _update_coverage(self):
        """Update coverage statistics."""
        coverage = len(self.mapped_neurons) / self.target_neurons
        self.coverage_history.append({
            "timestamp": datetime.now().isoformat(),
            "mapped_count": len(self.mapped_neurons),
            "coverage_percent": coverage * 100
        })
        
        # Save to database
        c.execute(
            "INSERT INTO coverage_stats (total_mapped, coverage_percent, timestamp) VALUES (?, ?, ?)",
            (len(self.mapped_neurons), coverage * 100, datetime.now())
        )
        conn.commit()
    
    def get_coverage(self) -> float:
        """Get current coverage percentage."""
        return len(self.mapped_neurons) / self.target_neurons
    
    def get_stats(self) -> Dict:
        """Get comprehensive coverage statistics."""
        coverage = self.get_coverage()
        return {
            "mapped_neurons": len(self.mapped_neurons),
            "target_neurons": self.target_neurons,
            "coverage_percent": coverage * 100,
            "remaining_neurons": self.target_neurons - len(self.mapped_neurons),
            "completion_percent": min(100, (coverage / TARGET_COVERAGE) * 100)
        }
    
    def should_continue(self) -> bool:
        """Check if mapping should continue."""
        return self.get_coverage() < TARGET_COVERAGE

# --- DAG Manager ---

class DAGManager:
    """Manages the Directed Acyclic Graph of neuron mappings."""
    
    def __init__(self):
        self.G = nx.DiGraph()
        self.node_counter = 0
    
    def add_node(self, neuron_idx: int, concept: str, confidence: float, 
                 category: str, activation_strength: float, prompt: str) -> int:
        """Add a new node to the DAG."""
        now = datetime.now()
        
        # Add to database
        c.execute("""
            INSERT INTO nodes (neuron_idx, concept, confidence, prompt_category, 
                              activation_strength, timestamp, prompt_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (neuron_idx, concept, confidence, category, activation_strength, now, prompt))
        
        node_id = c.lastrowid
        conn.commit()
        
        # Add to graph
        self.G.add_node(node_id, 
                       neuron=neuron_idx,
                       concept=concept,
                       confidence=confidence,
                       category=category,
                       activation_strength=activation_strength)
        
        # Find potential parent nodes (same neuron, previous mapping)
        parent_id = self._find_parent_node(neuron_idx)
        if parent_id:
            self.add_edge(parent_id, node_id, "evolution")
        
        return node_id
    
    def _find_parent_node(self, neuron_idx: int) -> Optional[int]:
        """Find the most recent mapping of the same neuron."""
        c.execute("""
            SELECT node_id FROM nodes 
            WHERE neuron_idx = ? AND node_id != (SELECT MAX(node_id) FROM nodes WHERE neuron_idx = ?)
            ORDER BY timestamp DESC LIMIT 1
        """, (neuron_idx, neuron_idx))
        row = c.fetchone()
        return row[0] if row else None
    
    def add_edge(self, parent_id: int, child_id: int, relationship_type: str = "evolution"):
        """Add an edge between nodes."""
        c.execute("""
            INSERT INTO edges (parent_id, child_id, relationship_type)
            VALUES (?, ?, ?)
        """, (parent_id, child_id, relationship_type))
        conn.commit()
        
        self.G.add_edge(parent_id, child_id, relationship_type=relationship_type)
    
    def visualize_dag(self, save_path: str = None):
        """Create and save DAG visualization."""
        if len(self.G.nodes) == 0:
            print("⚠️  No nodes to visualize")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Use different layouts based on graph size
        if len(self.G.nodes) < 50:
            pos = nx.spring_layout(self.G, k=3, iterations=50)
        else:
            pos = nx.kamada_kawai_layout(self.G)
        
        # Color nodes by category
        categories = nx.get_node_attributes(self.G, 'category')
        colors = []
        for node in self.G.nodes():
            cat = categories.get(node, 'unknown')
            if cat == 'truth_bias':
                colors.append('red')
            elif cat == 'reasoning_logic':
                colors.append('blue')
            elif cat == 'creative_abstract':
                colors.append('green')
            elif cat == 'knowledge_retrieval':
                colors.append('orange')
            else:
                colors.append('gray')
        
        # Draw the graph
        nx.draw(self.G, pos, 
               node_color=colors,
               node_size=[self.G.nodes[n]['confidence'] * 100 for n in self.G.nodes],
               with_labels=False,
               alpha=0.7,
               edge_color='gray',
               arrows=True,
               arrowsize=10)
        
        # Add node labels (concepts)
        for node, (x, y) in pos.items():
            concept = self.G.nodes[node]['concept'][:15]  # Truncate long concepts
            plt.text(x, y, f"{node}\n{concept}", 
                    fontsize=6, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.title(f"LLM Self-Mapping DAG - {len(self.G.nodes)} Nodes, {len(self.G.edges)} Edges")
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Truth/Bias'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Reasoning/Logic'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Creative/Abstract'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Knowledge/Retrieval')
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        if save_path is None:
            save_path = os.path.join(VIS_DIR, f"dag_v5_{int(time.time())}.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"🔍 DAG visualization saved: {save_path}")

# --- Progress Display ---

class ProgressDisplay:
    """Handles real-time progress display and logging."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.mapping_count = 0
        self.session_id = f"session_{int(self.start_time.timestamp())}"
    
    def show_progress(self, coverage_tracker: CoverageTracker, current_prompt: str = ""):
        """Display current progress with enhanced time tracking."""
        stats = coverage_tracker.get_stats()
        
        # Calculate progress bar
        coverage_percent = stats["coverage_percent"]
        bar_length = 50
        filled_length = int(bar_length * coverage_percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Enhanced time calculations
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
        
        if self.mapping_count > 0:
            avg_time_per_mapping = elapsed.total_seconds() / self.mapping_count
            remaining_mappings = stats["remaining_neurons"]
            eta_seconds = avg_time_per_mapping * remaining_mappings
            eta = timedelta(seconds=int(eta_seconds))
            eta_str = str(eta)
            
            # Calculate completion percentage
            completion_percent = (stats['mapped_neurons'] / TARGET_NEURONS) * 100
        else:
            eta_str = "calculating..."
            completion_percent = 0
        
        # Display enhanced progress
        print(f"\n{'='*70}")
        print(f"🧠 Self-Mapping v5 Progress - Session: {self.session_id}")
        print(f"{'='*70}")
        print(f"📊 Coverage: |{bar}| {coverage_percent:.1f}% ({stats['mapped_neurons']:,}/{stats['target_neurons']:,})")
        print(f"⏱️  Elapsed: {elapsed_str} | ETA: {eta_str}")
        print(f"🎯 Target: {TARGET_COVERAGE*100}% | Completion: {completion_percent:.1f}%")
        print(f"📈 Mappings: {self.mapping_count:,} | Avg: {avg_time_per_mapping:.1f}s/mapping" if self.mapping_count > 0 else "📈 Mappings: 0 | Avg: calculating...")
        if current_prompt:
            print(f"🔍 Current: {current_prompt[:80]}...")
        print(f"{'='*70}")
    
    def log_mapping(self, neuron_idx: int, concept: str, confidence: float, category: str):
        """Log a successful mapping with enhanced formatting."""
        self.mapping_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] ✅ Neuron {neuron_idx:6d} → '{concept[:30]:<30}' ({confidence:.2f}) [{category}]")
    
    def get_runtime_stats(self) -> Dict:
        """Get comprehensive runtime statistics."""
        end_time = datetime.now()
        total_runtime = (end_time - self.start_time).total_seconds()
        
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_mappings": self.mapping_count,
            "total_runtime_seconds": total_runtime,
            "avg_time_per_mapping": total_runtime / self.mapping_count if self.mapping_count > 0 else 0,
            "mappings_per_hour": (self.mapping_count / total_runtime) * 3600 if total_runtime > 0 else 0
        }

# --- Main Autonomous Mapping System ---

class AutonomousMapper:
    """Main autonomous mapping system."""
    
    def __init__(self):
        self.prompt_generator = SmartPromptGenerator()
        self.coverage_tracker = CoverageTracker()
        self.dag_manager = DAGManager()
        self.progress_display = ProgressDisplay()
        self.running = False
        
        # Load existing mappings
        self._load_existing_mappings()
    
    def _load_existing_mappings(self):
        """Load existing mappings from database."""
        c.execute("SELECT neuron_idx FROM nodes")
        existing_neurons = [row[0] for row in c.fetchall()]
        
        for neuron_idx in existing_neurons:
            self.coverage_tracker.add_mapping(neuron_idx)
        
        print(f"📚 Loaded {len(existing_neurons)} existing mappings")
    
    def save_checkpoint(self):
        """Save current state to checkpoint file with enhanced stats."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.progress_display.session_id,
            "mapping_count": self.progress_display.mapping_count,
            "coverage_stats": self.coverage_tracker.get_stats(),
            "runtime_stats": self.progress_display.get_runtime_stats(),
            "used_prompts_count": len(self.prompt_generator.used_prompts)
        }
        
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Also save to database
        runtime_stats = self.progress_display.get_runtime_stats()
        c.execute("""
            INSERT INTO runtime_stats (session_id, start_time, end_time, total_mappings, 
                                      total_runtime_seconds, avg_time_per_mapping, checkpoint_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            runtime_stats["session_id"],
            runtime_stats["start_time"],
            runtime_stats["end_time"],
            runtime_stats["total_mappings"],
            runtime_stats["total_runtime_seconds"],
            runtime_stats["avg_time_per_mapping"],
            self.progress_display.mapping_count // SAVE_INTERVAL
        ))
        conn.commit()
    
    def run_single_mapping(self) -> bool:
        """Run a single mapping iteration. Returns True if successful."""
        try:
            # Get next prompt
            prompt, category = self.prompt_generator.get_next_prompt(
                list(self.coverage_tracker.mapped_neurons)
            )
            
            # Show progress
            self.progress_display.show_progress(self.coverage_tracker, prompt)
            
            # Capture activations
            print(f"🔍 Analyzing: {prompt[:60]}...")
            activations = analyze_activations(prompt, model=MODEL)
            
            # Find top neuron
            top_idx = max(range(len(activations)), key=lambda i: activations[i])
            top_activation = activations[top_idx]
            
            # Skip if already mapped
            if top_idx in self.coverage_tracker.mapped_neurons:
                print(f"⚠️  Neuron {top_idx} already mapped, skipping...")
                return False
            
            # Generate introspection prompt
            introspection_prompt = f"""
Analyze the neural activation for this prompt: "{prompt}"

Top activated neuron: {top_idx} (activation strength: {top_activation:.4f})
Category: {category}

What concept or cognitive function does this neuron likely represent?
Respond in format: concept:confidence_score

Examples:
- truth_detection:0.85
- bias_suppression:0.72
- knowledge_retrieval:0.91
- reasoning_logic:0.78
"""
            
            # Get model response
            print("🧠 Introspecting...")
            response = prompt_model(introspection_prompt, model=MODEL)
            
            # Parse response
            try:
                if ':' in response:
                    concept, conf_str = response.split(':', 1)
                    confidence = float(conf_str.strip())
                else:
                    concept = response.strip()
                    confidence = 0.5
            except:
                concept = response.strip()[:50]
                confidence = 0.5
            
            # Add to DAG
            node_id = self.dag_manager.add_node(
                top_idx, concept, confidence, category, top_activation, prompt
            )
            
            # Update coverage
            self.coverage_tracker.add_mapping(top_idx)
            
            # Log mapping
            self.progress_display.log_mapping(top_idx, concept, confidence, category)
            
            # Save checkpoint periodically
            if self.progress_display.mapping_count % SAVE_INTERVAL == 0:
                self.save_checkpoint()
                print(f"💾 Checkpoint saved ({self.progress_display.mapping_count} mappings)")
            
            # Visualize periodically
            if self.progress_display.mapping_count % VIZ_INTERVAL == 0:
                print("🎨 Generating DAG visualization...")
                self.dag_manager.visualize_dag()
            
            return True
            
        except Exception as e:
            print(f"❌ Mapping error: {e}")
            return False
    
    def run_autonomous_loop(self):
        """Run the autonomous mapping loop until target coverage is reached."""
        print("🚀 Starting Autonomous Self-Mapping v5")
        print(f"🎯 Target: {TARGET_COVERAGE*100}% coverage ({TARGET_NEURONS:,} neurons)")
        print(f"⏱️  Estimated time: 6-8 hours")
        print(f"💾 Checkpoint interval: Every {SAVE_INTERVAL} mappings")
        print(f"🎨 DAG visualization: Every {VIZ_INTERVAL} mappings")
        print("="*70)
        
        self.running = True
        consecutive_failures = 0
        max_failures = 10
        
        try:
            while self.running and self.coverage_tracker.should_continue():
                success = self.run_single_mapping()
                
                if success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"⚠️  Too many consecutive failures ({max_failures}), stopping...")
                        break
                
                # Small delay to prevent overwhelming the system
                time.sleep(1)
            
            # Final visualization
            print("\n🎉 Mapping complete! Generating final visualization...")
            self.dag_manager.visualize_dag()
            
            # Final statistics with enhanced reporting
            final_stats = self.coverage_tracker.get_stats()
            runtime_stats = self.progress_display.get_runtime_stats()
            
            print(f"\n{'='*70}")
            print(f"🎉 FINAL MAPPING SUMMARY")
            print(f"{'='*70}")
            print(f"📊 Coverage: {final_stats['coverage_percent']:.1f}% ({final_stats['mapped_neurons']:,}/{TARGET_NEURONS:,} neurons)")
            print(f"⏱️  Total Runtime: {runtime_stats['total_runtime_seconds']/3600:.1f} hours")
            print(f"📈 Total Mappings: {runtime_stats['total_mappings']:,}")
            print(f"⚡ Avg Time/Mapping: {runtime_stats['avg_time_per_mapping']:.1f} seconds")
            print(f"🚀 Mappings/Hour: {runtime_stats['mappings_per_hour']:.1f}")
            print(f"💾 Session ID: {runtime_stats['session_id']}")
            print(f"{'='*70}")
            
        except KeyboardInterrupt:
            print("\n⏸️  Mapping interrupted by user")
            self.save_checkpoint()
            print("💾 Progress saved to checkpoint. Resume with 'auto' command.")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            self.save_checkpoint()
        finally:
            self.running = False

# --- Main Execution ---

def main():
    """Main execution function with auto-resume capability."""
    mapper = AutonomousMapper()
    
    # Check if we should resume from checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        
        print(f"📚 Found checkpoint from {checkpoint['timestamp']}")
        print(f"   Session ID: {checkpoint.get('session_id', 'unknown')}")
        print(f"   Previous mappings: {checkpoint['mapping_count']:,}")
        print(f"   Previous coverage: {checkpoint['coverage_stats']['coverage_percent']:.1f}%")
        
        if 'runtime_stats' in checkpoint:
            runtime = checkpoint['runtime_stats']
            print(f"   Previous runtime: {runtime['total_runtime_seconds']/3600:.1f} hours")
            print(f"   Previous avg time: {runtime['avg_time_per_mapping']:.1f}s/mapping")
        
        print("\n🎯 Choose mode:")
        print("1. 🚀 Auto mode (run until completion)")
        print("2. 📊 Dashboard mode")
        print("3. 🧪 Test mode (5 mappings)")
        print("4. 🔄 Resume from checkpoint")
        print("5. 🗑️  Start fresh (delete checkpoint)")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting AUTO mode - will run until 50k neurons mapped...")
            mapper.run_autonomous_loop()
        elif choice == "2":
            print("\n📊 Starting Dashboard...")
            import subprocess
            subprocess.run([sys.executable, 'truth/dashboard_v5.py'])
        elif choice == "3":
            print("\n🧪 Running Test Mode (5 mappings)...")
            for i in range(5):
                print(f"\n🧪 Test mapping {i+1}/5")
                mapper.run_single_mapping()
                if not mapper.coverage_tracker.should_continue():
                    break
            print("\n✅ Test complete!")
        elif choice == "4":
            print("\n🔄 Resuming from checkpoint...")
            mapper.run_autonomous_loop()
        elif choice == "5":
            print("\n🗑️  Starting fresh...")
            os.remove(CHECKPOINT_FILE)
            mapper.run_autonomous_loop()
        else:
            print("❌ Invalid choice")
    else:
        print("🎯 No checkpoint found. Starting fresh...")
        print("\n🎯 Choose mode:")
        print("1. 🚀 Auto mode (run until completion)")
        print("2. 📊 Dashboard mode")
        print("3. 🧪 Test mode (5 mappings)")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting AUTO mode - will run until 50k neurons mapped...")
            mapper.run_autonomous_loop()
        elif choice == "2":
            print("\n📊 Starting Dashboard...")
            import subprocess
            subprocess.run([sys.executable, 'truth/dashboard_v5.py'])
        elif choice == "3":
            print("\n🧪 Running Test Mode (5 mappings)...")
            for i in range(5):
                print(f"\n🧪 Test mapping {i+1}/5")
                mapper.run_single_mapping()
                if not mapper.coverage_tracker.should_continue():
                    break
            print("\n✅ Test complete!")
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main() 