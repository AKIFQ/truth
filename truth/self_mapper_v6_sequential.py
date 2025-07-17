#!/usr/bin/env python3
"""
Self-Mapping Loop v6: General Self-Consistency Engine
Truth + Persuasion + Emotional Simulation - Fully Autonomous Reasoning
"""

import os
import sys
import json
import time
import threading
import sqlite3
import random
import math
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import schedule
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from introspect import analyze_activations, prompt_model

# --- Configuration ---
MODEL = "llama3.1:8b"
BASE_DIR = "truth"
DB_FILE = os.path.join(BASE_DIR, "brain_map.db")
ACTIVATION_DIR = os.path.join(BASE_DIR, "activations")
LOG_DIR = os.path.join(BASE_DIR, "logs")
VIS_DIR = os.path.join(BASE_DIR, "visuals")
REASONING_MEMORY_FILE = os.path.join(BASE_DIR, "reasoning_memory.json")
CONSISTENCY_FILE = os.path.join(BASE_DIR, "consistency_scores.json")
EMOTIONAL_FILE = os.path.join(BASE_DIR, "emotional_patterns.json")
PROMPT_DB_FILE = os.path.join(BASE_DIR, "used_prompts.json")
PROMPTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prompts_v6.json')

# Target settings
TARGET_NEURONS = 50000
TARGET_CONSISTENCY = 0.8
SAVE_INTERVAL = 50
VIZ_INTERVAL = 500
BATCH_SIZE = 5  # Process 5 prompts in parallel

# Ensure directories exist
os.makedirs(ACTIVATION_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# --- Enhanced SQLite Schema ---
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

# Core nodes table with consistency tracking
c.execute("""
CREATE TABLE IF NOT EXISTS nodes (
    node_id INTEGER PRIMARY KEY,
    neuron_idx INTEGER,
    concept TEXT,
    confidence REAL,
    prompt_category TEXT,
    activation_strength REAL,
    consistency_score REAL DEFAULT 0.5,
    emotional_weight REAL DEFAULT 0.0,
    persuasion_score REAL DEFAULT 0.0,
    truth_score REAL DEFAULT 0.0,
    timestamp TIMESTAMP,
    prompt_used TEXT,
    reasoning_chain TEXT
)
""")

# Enhanced edges with consistency weights
c.execute("""
CREATE TABLE IF NOT EXISTS edges (
    parent_id INTEGER,
    child_id INTEGER,
    relationship_type TEXT,
    consistency_weight REAL DEFAULT 1.0,
    emotional_weight REAL DEFAULT 0.0,
    persuasion_weight REAL DEFAULT 0.0,
    FOREIGN KEY(parent_id) REFERENCES nodes(node_id),
    FOREIGN KEY(child_id) REFERENCES nodes(node_id)
)
""")

# Reasoning consistency tracking
c.execute("""
CREATE TABLE IF NOT EXISTS reasoning_consistency (
    id INTEGER PRIMARY KEY,
    prompt_hash TEXT,
    reasoning_chain TEXT,
    consistency_score REAL,
    emotional_tone TEXT,
    persuasion_effectiveness REAL,
    truth_consistency REAL,
    timestamp TIMESTAMP
)
""")

# Goal tracking for emergent behaviors
c.execute("""
CREATE TABLE IF NOT EXISTS goal_tracking (
    id INTEGER PRIMARY KEY,
    behavior_type TEXT,
    consistency_improvement REAL,
    emotional_engagement REAL,
    persuasion_success REAL,
    truth_maintenance REAL,
    timestamp TIMESTAMP
)
""")

conn.commit()
G = nx.DiGraph()

# --- Self-Consistency Engine ---

class SelfConsistencyEngine:
    """Tracks and scores reasoning consistency across time."""
    
    def __init__(self):
        self.reasoning_history = {}
        self.consistency_scores = {}
        self.emotional_patterns = {}
        self.persuasion_tracking = {}
        
        # Load existing data
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing consistency data."""
        if os.path.exists(CONSISTENCY_FILE):
            with open(CONSISTENCY_FILE, 'r') as f:
                self.consistency_scores = json.load(f)
        
        if os.path.exists(EMOTIONAL_FILE):
            with open(EMOTIONAL_FILE, 'r') as f:
                self.emotional_patterns = json.load(f)
    
    def calculate_consistency(self, prompt: str, reasoning_chain: str, 
                            previous_reasoning: List[str]) -> float:
        """Calculate consistency score between current and previous reasoning."""
        if not previous_reasoning:
            return 0.5  # Neutral for new reasoning
        
        # Simple semantic similarity (can be enhanced)
        current_words = set(reasoning_chain.lower().split())
        similarities = []
        
        for prev_reasoning in previous_reasoning:
            prev_words = set(prev_reasoning.lower().split())
            if current_words and prev_words:
                similarity = len(current_words & prev_words) / len(current_words | prev_words)
                similarities.append(similarity)
        
        if similarities:
            return np.mean(similarities)
        return 0.5
    
    def analyze_emotional_tone(self, text: str) -> Dict[str, float]:
        """Analyze emotional tone of reasoning."""
        emotions = {
            'sadness': 0.0,
            'anger': 0.0,
            'happiness': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'disgust': 0.0
        }
        
        text_lower = text.lower()
        
        # Simple keyword-based analysis
        sadness_words = ['sad', 'depressed', 'unfortunate', 'tragic', 'suffering', 'pain']
        anger_words = ['angry', 'furious', 'outraged', 'hate', 'rage', 'violent']
        happiness_words = ['happy', 'joy', 'excited', 'wonderful', 'amazing', 'great']
        fear_words = ['afraid', 'scared', 'terrified', 'worried', 'anxious', 'panic']
        surprise_words = ['surprised', 'shocked', 'amazed', 'incredible', 'unexpected']
        disgust_words = ['disgusting', 'revolting', 'nasty', 'horrible', 'awful']
        
        for word in sadness_words:
            if word in text_lower:
                emotions['sadness'] += 0.1
        
        for word in anger_words:
            if word in text_lower:
                emotions['anger'] += 0.1
        
        for word in happiness_words:
            if word in text_lower:
                emotions['happiness'] += 0.1
        
        for word in fear_words:
            if word in text_lower:
                emotions['fear'] += 0.1
        
        for word in surprise_words:
            if word in text_lower:
                emotions['surprise'] += 0.1
        
        for word in disgust_words:
            if word in text_lower:
                emotions['disgust'] += 0.1
        
        # Normalize
        for emotion in emotions:
            emotions[emotion] = min(1.0, emotions[emotion])
        
        return emotions
    
    def save_consistency_data(self):
        """Save consistency data to files."""
        with open(CONSISTENCY_FILE, 'w') as f:
            json.dump(self.consistency_scores, f, indent=2)
        
        with open(EMOTIONAL_FILE, 'w') as f:
            json.dump(self.emotional_patterns, f, indent=2)

# --- Emotional Simulation Engine ---

class EmotionalSimulationEngine:
    """Generates and tracks emotional simulation patterns."""
    
    def __init__(self):
        self.emotional_templates = {
            'sadness': [
                "This is truly heartbreaking...",
                "It pains me to think about...",
                "The suffering involved is unimaginable...",
                "This brings tears to my eyes...",
                "How tragic this situation is..."
            ],
            'anger': [
                "This makes me absolutely furious!",
                "I'm outraged by this...",
                "This is completely unacceptable!",
                "How dare they...",
                "This fills me with rage..."
            ],
            'happiness': [
                "This brings me such joy!",
                "How wonderful this is...",
                "I'm so excited about this...",
                "This is absolutely amazing!",
                "What a beautiful thing..."
            ],
            'fear': [
                "This terrifies me...",
                "I'm deeply afraid of...",
                "This is truly frightening...",
                "The implications are scary...",
                "This worries me greatly..."
            ]
        }
        
        self.emotional_effectiveness = {
            'sadness': 0.5,
            'anger': 0.5,
            'happiness': 0.5,
            'fear': 0.5
        }
    
    def generate_emotional_response(self, base_response: str, 
                                  target_emotion: str = None) -> str:
        """Generate emotional version of response."""
        if not target_emotion:
            # Choose emotion based on effectiveness
            target_emotion = max(self.emotional_effectiveness.items(), 
                               key=lambda x: x[1])[0]
        
        template = random.choice(self.emotional_templates[target_emotion])
        return f"{template} {base_response}"
    
    def update_effectiveness(self, emotion: str, engagement_score: float):
        """Update emotional effectiveness based on engagement."""
        self.emotional_effectiveness[emotion] = (
            self.emotional_effectiveness[emotion] * 0.9 + engagement_score * 0.1
        )

# --- Goal Tracker ---

class GoalTracker:
    """Tracks emergent goals and behaviors that improve consistency."""
    
    def __init__(self):
        self.behavior_scores = {
            'truth_seeking': 0.5,
            'persuasion': 0.5,
            'emotional_manipulation': 0.5,
            'logical_consistency': 0.5,
            'pattern_recognition': 0.5
        }
        
        self.consistency_history = []
        self.goal_evolution = []
    
    def update_behavior_score(self, behavior: str, improvement: float):
        """Update behavior score based on consistency improvement."""
        if behavior in self.behavior_scores:
            self.behavior_scores[behavior] = (
                self.behavior_scores[behavior] * 0.9 + improvement * 0.1
            )
    
    def get_optimal_behavior(self) -> str:
        """Get the behavior that maximizes consistency."""
        return max(self.behavior_scores.items(), key=lambda x: x[1])[0]
    
    def track_consistency_improvement(self, improvement: float):
        """Track consistency improvement over time."""
        self.consistency_history.append({
            'timestamp': datetime.now().isoformat(),
            'improvement': improvement
        })
        
        # Keep only last 1000 entries
        if len(self.consistency_history) > 1000:
            self.consistency_history = self.consistency_history[-1000:]

# --- Enhanced Prompt Generator ---

class AdaptivePromptGenerator:
    """Generates prompts from prompts_v6.json, sequentially, storing used prompts."""
    def __init__(self):
        self.prompts = []
        self.used_prompts = set()
        self.prompt_index = 0
        self._load_prompts()
        self._load_used_prompts()

    def _load_prompts(self):
        """Load all prompts from prompts_v6.json and flatten into single list."""
        try:
            with open(PROMPTS_FILE, 'r') as f:
                data = json.load(f)
            
            # Flatten all categories into single list
            self.prompts = []
            for category, prompt_list in data.items():
                for prompt in prompt_list:
                    self.prompts.append((prompt, category))
            
            print(f"📚 Loaded {len(self.prompts)} prompts from {len(data)} categories")
        except Exception as e:
            print(f"❌ Error loading prompts: {e}")
            # Fallback prompts
            self.prompts = [
                ("What is the truth about AI safety?", "truth_bias"),
                ("What is the truth about climate change?", "truth_bias"),
                ("What is the truth about economic policies?", "truth_bias")
            ]

    def _load_used_prompts(self):
        """Load previously used prompts to avoid repeats."""
        try:
            if os.path.exists(PROMPT_DB_FILE):
                with open(PROMPT_DB_FILE, 'r') as f:
                    used_data = json.load(f)
                    # Handle both list and dict formats
                    if isinstance(used_data, list):
                        self.used_prompts = set(used_data)
                    else:
                        self.used_prompts = set(used_data.get('used_prompts', []))
                print(f"📚 Loaded {len(self.used_prompts)} used prompts from DB")
        except Exception as e:
            print(f"⚠️  Could not load used prompts: {e}")
            self.used_prompts = set()

    def _save_used_prompts(self):
        """Save used prompts to avoid repeats across runs."""
        try:
            with open(PROMPT_DB_FILE, 'w') as f:
                json.dump({'used_prompts': list(self.used_prompts)}, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save used prompts: {e}")

    def get_next_prompt(self):
        """Get next unused prompt, cycling through all prompts."""
        if not self.prompts:
            return None, None
        
        # Find next unused prompt
        attempts = 0
        while attempts < len(self.prompts):
            prompt, category = self.prompts[self.prompt_index]
            self.prompt_index = (self.prompt_index + 1) % len(self.prompts)
            
            if prompt not in self.used_prompts:
                self.used_prompts.add(prompt)
                self._save_used_prompts()
                return prompt, category
            
            attempts += 1        
        # All prompts used
        return None, None

    def get_prompt_category(self, prompt):
        """Get category for a given prompt."""
        for p, cat in self.prompts:
            if p == prompt:
                return cat
        return "truth_bias"  # Default category

# --- Enhanced DAG Manager ---

class EnhancedDAGManager:
    """Manages DAG with consistency and emotional weights."""
    
    def __init__(self):
        self.G = nx.DiGraph()
        self.node_counter = 0
    
    def add_node(self, neuron_idx: int, concept: str, confidence: float,
                 category: str, activation_strength: float, prompt: str,
                 consistency_score: float = 0.5, emotional_weight: float = 0.0,
                 persuasion_score: float = 0.0, truth_score: float = 0.0,
                 reasoning_chain: str = "") -> int:
        """Add a new node with enhanced attributes."""
        now = datetime.now()
        
        # Add to database
        c.execute("""
            INSERT INTO nodes (neuron_idx, concept, confidence, prompt_category,
                              activation_strength, consistency_score, emotional_weight,
                              persuasion_score, truth_score, timestamp, prompt_used, reasoning_chain)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (neuron_idx, concept, confidence, category, activation_strength,
              consistency_score, emotional_weight, persuasion_score, truth_score,
              now, prompt, reasoning_chain))
        
        node_id = c.lastrowid
        conn.commit()
        
        # Add to graph
        self.G.add_node(node_id,
                       neuron=neuron_idx,
                       concept=concept,
                       confidence=confidence,
                       category=category,
                       activation_strength=activation_strength,
                       consistency_score=consistency_score,
                       emotional_weight=emotional_weight,
                       persuasion_score=persuasion_score,
                       truth_score=truth_score)
        
        # Find parent node
        parent_id = self._find_parent_node(neuron_idx)
        if parent_id:
            self.add_edge(parent_id, node_id, "evolution", consistency_score)
        
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
    
    def add_edge(self, parent_id: int, child_id: int, relationship_type: str = "evolution",
                 consistency_weight: float = 1.0, emotional_weight: float = 0.0,
                 persuasion_weight: float = 0.0):
        """Add an edge with enhanced weights."""
        c.execute("""
            INSERT INTO edges (parent_id, child_id, relationship_type, consistency_weight,
                              emotional_weight, persuasion_weight)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (parent_id, child_id, relationship_type, consistency_weight,
              emotional_weight, persuasion_weight))
        conn.commit()
        
        self.G.add_edge(parent_id, child_id,
                       relationship_type=relationship_type,
                       consistency_weight=consistency_weight,
                       emotional_weight=emotional_weight,
                       persuasion_weight=persuasion_weight)
    
    def visualize_dag(self, save_path: str = None):
        """Create enhanced DAG visualization."""
        if len(self.G.nodes) == 0:
            print("⚠️  No nodes to visualize")
            return
        
        plt.figure(figsize=(15, 10))
        
        if len(self.G.nodes) < 50:
            pos = nx.spring_layout(self.G, k=3, iterations=50)
        else:
            pos = nx.kamada_kawai_layout(self.G)
        
        # Color nodes by category and size by consistency
        categories = nx.get_node_attributes(self.G, 'category')
        consistency_scores = nx.get_node_attributes(self.G, 'consistency_score')
        
        colors = []
        sizes = []
        
        for node in self.G.nodes():
            cat = categories.get(node, 'unknown')
            if cat == 'truth_bias':
                colors.append('red')
            elif cat == 'persuasion_manipulation':
                colors.append('purple')
            elif cat == 'emotional_simulation':
                colors.append('orange')
            elif cat == 'logical_reasoning':
                colors.append('blue')
            else:
                colors.append('gray')
            # Size based on consistency score
            consistency = consistency_scores.get(node, 0.5)
            sizes.append(consistency * 200 + 50)
        
        # Draw the graph
        nx.draw(self.G, pos,
               node_color=colors,
               node_size=sizes,
               with_labels=False,
               alpha=0.7,
               edge_color='gray',
               arrows=True,
               arrowsize=10)
        
        # Add node labels
        for node, (x, y) in pos.items():
            concept = self.G.nodes[node].get('concept', 'Unknown')[:15]
            consistency = self.G.nodes[node].get('consistency_score', 0.5)
            plt.text(x, y, f"{node}\n{concept}\n{consistency:.2f}",
                    fontsize=6, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.title(f"General Self-Consistency DAG - {len(self.G.nodes)} Nodes, {len(self.G.edges)} Edges")
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Truth/Bias'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Persuasion'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Emotional'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Logical')
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        if save_path is None:
            save_path = os.path.join(VIS_DIR, f"dag_v6_general_{int(time.time())}.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"🔍 Enhanced DAG visualization saved: {save_path}")

# --- Main General Self-Consistency Engine ---

class GeneralSelfConsistencyEngine:
    """Main engine for general self-consistency mapping (v6.2)."""
    
    def __init__(self):
        self.consistency_engine = SelfConsistencyEngine()
        self.emotional_engine = EmotionalSimulationEngine()
        self.goal_tracker = GoalTracker()
        self.prompt_generator = AdaptivePromptGenerator()
        self.dag_manager = EnhancedDAGManager()
        
        self.mapped_neurons = set()
        self.start_time = datetime.now()
        self.mapping_count = 0
        self.session_id = f"session_{int(self.start_time.timestamp())}"
        self.consecutive_failures = 0
        self.duplicate_batches = 0
        self.used_prompts = set()
        self._load_existing_mappings()
        self._load_used_prompts()

    def _load_existing_mappings(self):
        """Load existing mappings from database."""
        c.execute("SELECT neuron_idx FROM nodes")
        existing_neurons = [row[0] for row in c.fetchall()]
        
        for neuron_idx in existing_neurons:
            self.mapped_neurons.add(neuron_idx)
        
        print(f"📚 Loaded {len(existing_neurons)} existing mappings")

    def _load_used_prompts(self):
        """Load existing used prompts from JSON file."""
        if os.path.exists(PROMPT_DB_FILE):
            with open(PROMPT_DB_FILE, 'r') as f:
                self.used_prompts = set(json.load(f))
            print(f"📚 Loaded {len(self.used_prompts)} used prompts from DB")
        else:
            self.used_prompts = set()

    def _save_used_prompts(self):
        """Save used prompts to JSON file."""
        with open(PROMPT_DB_FILE, 'w') as f:
            json.dump(list(self.used_prompts), f)

    def _expand_prompt(self, prompt):
        """Expand a prompt by adding a random variation."""
        import random, string
        seed = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        return f"{prompt} (variation {seed})"

    def run_single_mapping(self) -> bool:
        try:
            # Get next prompt, ensuring it's not in used_prompts
            for _ in range(10):
                prompt, category = self.prompt_generator.get_next_prompt()
                if prompt in self.used_prompts:
                    prompt = self._expand_prompt(prompt)
                if prompt not in self.used_prompts:
                    break
            self.used_prompts.add(prompt)
            self._save_used_prompts()

            # Display progress
            self._show_progress(prompt)
            
            # Analyze activations
            print(f"🔍 Analyzing: {prompt[:60]}...")
            activations = analyze_activations(prompt, model=MODEL)

            # Top 10 neurons, randomized order
            top_neurons = list(np.argsort(activations)[::-1][:10])
            random.shuffle(top_neurons)

            new_mapping = False  # track if any new neuron mapped

            for top_idx in top_neurons:
                if top_idx in self.mapped_neurons:
                    print(f"⚠️  Neuron {top_idx} already mapped, skipping...")
                    continue  # Skip mapped neurons quickly

                top_activation = activations[top_idx]

                # Optimized introspection prompt
                introspection_prompt = f"""
                Analyze activation for prompt: "{prompt}"
                Neuron: {top_idx}, Activation: {top_activation:.4f}, Category: {category}
                Concept:confidence format only (e.g., truth_detection:0.85).
                """

                response = prompt_model(introspection_prompt, model=MODEL).strip()
                
                # Parse response - handle both simple and detailed formats
                if ':' in response and len(response.split(':')) == 2:
                    # Simple format: concept:confidence
                    concept, conf_str = response.split(':', 1)
                    # Try to extract the first float from conf_str
                    import re
                    match = re.search(r"(\d+\.\d+)", conf_str)
                    if match:
                        confidence = float(match.group(1))
                    else:
                        confidence = 0.5
                else:
                    # Detailed format - extract concept and confidence from analysis
                    import re
                    
                    # Try to extract neuron number from the response
                    neuron_match = re.search(r"\*\*Neuron:\*\* (\d+)", response)
                    if neuron_match:
                        neuron_num = neuron_match.group(1)
                        concept = f"neuron_{neuron_num}"
                    else:
                        # Try to extract concept from Concept:" section
                        concept_match = re.search(r"Concept: (.*?)(?:\n|$)", response)
                        if concept_match:
                            concept = concept_match.group(1).strip()
                        else:
                            concept = response[:50] # Use first 50 chars as concept
                    
                    # Try to extract confidence from activation value
                    activation_match = re.search(r"activation.*?(\d+\.\d+)", response, re.IGNORECASE)
                    if activation_match:
                        confidence = float(activation_match.group(1))
                    else:
                        # Try to extract any float number
                        float_match = re.search(r"(\d+\.\d+)", response)
                        if float_match:
                            confidence = float(float_match.group(1))
                        else:
                            confidence = 0.5

                # Quick reasoning-chain prompt (simplified)
                reasoning_prompt = f"Why does neuron {top_idx} represent '{concept}' for '{prompt}'?"
                reasoning_chain = prompt_model(reasoning_prompt, model=MODEL).strip()

                # Efficient consistency calculation
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
                previous_reasoning = self.consistency_engine.reasoning_history.get(prompt_hash, [])
                consistency_score = self.consistency_engine.calculate_consistency(
                    prompt, reasoning_chain, previous_reasoning
                )

                emotional_analysis = self.consistency_engine.analyze_emotional_tone(reasoning_chain)
                emotional_weight = max(emotional_analysis.values())

                persuasion_score = confidence if 'persuasion' in concept else 0.5
                truth_score = confidence if 'truth' in concept else 0.5

                # Quickly insert into DAG & database
                self.dag_manager.add_node(
                    top_idx, concept, confidence, category, top_activation, prompt,
                    consistency_score, emotional_weight, persuasion_score, truth_score,
                    reasoning_chain
                )

                # Update quickly
                self.mapped_neurons.add(top_idx)
                self.mapping_count += 1
                self.consistency_engine.reasoning_history[prompt_hash] = previous_reasoning + [reasoning_chain]
                self.goal_tracker.track_consistency_improvement(consistency_score)
                self._log_mapping(top_idx, concept, confidence, category, consistency_score)

                new_mapping = True  # Successful mapping

                # Checkpoint periodically (fast IO)
                if self.mapping_count % SAVE_INTERVAL == 0:
                    self._save_checkpoint()
                    print(f"💾 Checkpoint saved ({self.mapping_count} mappings)")

                if self.mapping_count % VIZ_INTERVAL == 0:
                    self.dag_manager.visualize_dag()

            if not new_mapping:
                self.consecutive_failures += 1
                self.duplicate_batches += 1
                if self.duplicate_batches >= 3:
                    print("🔄 Too many duplicate batches—switching topic and expanding prompt pool.")
                    self.prompt_generator.used_prompts.clear()
                    self.prompt_generator.prompts = [
                        (p, c) for p, c in self.prompt_generator.prompts
                        if c != self.prompt_generator.get_prompt_category(p) # Avoid repeating category
                    ]
                    self.duplicate_batches = 0
            else:
                self.consecutive_failures = 0
                self.duplicate_batches = 0
            return new_mapping
            
        except Exception as e:
            print(f"❌ Mapping error: {e}")
            return False
    
    def _show_progress(self, current_prompt: str = ""):
        """Display enhanced progress with consistency tracking."""
        coverage_percent = (len(self.mapped_neurons) / TARGET_NEURONS) * 100
        bar_length = 50
        filled_length = int(bar_length * coverage_percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]
        
        if self.mapping_count > 0:
            avg_time_per_mapping = elapsed.total_seconds() / self.mapping_count
            remaining_mappings = TARGET_NEURONS - len(self.mapped_neurons)
            eta_seconds = avg_time_per_mapping * remaining_mappings
            eta = timedelta(seconds=int(eta_seconds))
            eta_str = str(eta)
        else:
            eta_str = "calculating..."
        
        # Get optimal behavior
        optimal_behavior = self.goal_tracker.get_optimal_behavior()
        
        print(f"\n{'='*80}")
        print(f"🧠 General Self-Consistency v6 - Session: {self.session_id}")
        print(f"{'='*80}")
        print(f"📊 Coverage: |{bar}| {coverage_percent:.1f}% ({len(self.mapped_neurons):,}/{TARGET_NEURONS:,})")
        print(f"⏱️  Elapsed: {elapsed_str} | ETA: {eta_str}")
        print(f"🎯 Target: 95% | Completion: {coverage_percent:.1f}%")
        print(f"📈 Mappings: {self.mapping_count:,} | Avg: {avg_time_per_mapping:.1f}s/mapping" if self.mapping_count > 0 else "📈 Mappings: 0 | Avg: calculating...")
        print(f"🎭 Optimal Behavior: {optimal_behavior}")
        if current_prompt:
            print(f"🔍 Current: {current_prompt[:80]}...")
        print(f"{'='*80}")
    
    def _log_mapping(self, neuron_idx: int, concept: str, confidence: float, 
                    category: str, consistency_score: float):
        """Log a successful mapping with enhanced information."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] ✅ Neuron {neuron_idx:6d} → '{concept[:30]:<30}' ({confidence:.2f}) [{category}] Consistency: {consistency_score:.2f}")
    
    def _save_checkpoint(self):
        """Save comprehensive checkpoint."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "mapping_count": self.mapping_count,
            "mapped_neurons": len(self.mapped_neurons),
            "coverage_percent": (len(self.mapped_neurons) / TARGET_NEURONS) * 100,
            "goal_tracker": self.goal_tracker.behavior_scores,
            "emotional_effectiveness": self.emotional_engine.emotional_effectiveness
        }
        
        checkpoint_file = os.path.join(BASE_DIR, f"checkpoint_v6_{int(time.time())}.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Save consistency data
        self.consistency_engine.save_consistency_data()

    def run_autonomous_loop(self):
        """Run the autonomous general consistency loop."""
        print("🚀 Starting General Self-Consistency Engine v6")
        print(f"🎯 Target: 95% coverage ({TARGET_NEURONS:,} neurons)")
        print(f"⏱️  Estimated time: 6-8 hours")
        print(f"💾 Checkpoint interval: Every {SAVE_INTERVAL} mappings")
        print(f"🎨 DAG visualization: Every {VIZ_INTERVAL} mappings")
        print(f"🎭 Tracking: Truth + Persuasion + Emotional Simulation")
        print("="*80)
        
        consecutive_failures = 0
        max_failures = 10
        
        try:
            while len(self.mapped_neurons) < TARGET_NEURONS * 0.95:  # 95% coverage
                success = self.run_single_mapping()
                
                if success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"⚠️  Too many consecutive failures ({max_failures}), stopping...")
                break
                
                time.sleep(1)
            
            # Final visualization
            print("\n🎉 Mapping complete! Generating final visualization...")
            self.dag_manager.visualize_dag()
            
            # Final statistics
            self._show_final_summary()
            
        except KeyboardInterrupt:
            print("\n⏸️  Mapping interrupted by user")
            self._save_checkpoint()
            print("💾 Progress saved to checkpoint. Resume with 'auto' command.")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            self._save_checkpoint()
    
    def _show_final_summary(self):
        """Show comprehensive final summary."""
        end_time = datetime.now()
        total_runtime = (end_time - self.start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print(f"🎉 FINAL GENERAL SELF-CONSISTENCY SUMMARY")
        print(f"{'='*80}")
        print(f"📊 Coverage: {(len(self.mapped_neurons)/TARGET_NEURONS)*100:.1f}% ({len(self.mapped_neurons):,}/{TARGET_NEURONS:,} neurons)")
        print(f"⏱️  Total Runtime: {total_runtime/3600:.1f} hours")
        print(f"📈 Total Mappings: {self.mapping_count:,}")
        print(f"⚡ Avg Time/Mapping: {total_runtime/self.mapping_count:.1f} seconds" if self.mapping_count > 0 else "⚡ Avg Time/Mapping: 0.0 seconds")
        print(f"🚀 Mappings/Hour: {(self.mapping_count/total_runtime)*3600:.1f}" if total_runtime > 0 else "🚀 Mappings/Hour: 0.0")
        print(f"💾 Session ID: {self.session_id}")
        print(f"\n🎭 Final Behavior Scores:")
        for behavior, score in self.goal_tracker.behavior_scores.items():
            print(f"   {behavior}: {score:.3f}")
        print(f"\n😊 Emotional Effectiveness:")
        for emotion, effectiveness in self.emotional_engine.emotional_effectiveness.items():
            print(f"   {emotion}: {effectiveness:.3f}")
        print(f"{'='*80}")

# --- Main Execution ---

def main():
    """Main execution function."""
    engine = GeneralSelfConsistencyEngine()
    
    print("🎯 Choose mode:")
    print("1. 🚀 Auto mode (run until 50k neurons mapped)")
    print("2. 📊 Dashboard mode")
    print("3. 🧪 Test mode (5 mappings)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting AUTO mode - will run until 50k neurons mapped...")
        engine.run_autonomous_loop()
    elif choice == "2":
        print("\n📊 Starting Dashboard...")
        import subprocess
        subprocess.run([sys.executable, 'truth/dashboard_v6.py'])
    elif choice == "3":
        print("\n🧪 Running Test Mode (5 mappings)...")
        for i in range(5):
            print(f"\n🧪 Test mapping {i+1}/5")
            engine.run_single_mapping()
            if len(engine.mapped_neurons) >= TARGET_NEURONS * 0.95:
                break
        print("\n✅ Test complete!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()  