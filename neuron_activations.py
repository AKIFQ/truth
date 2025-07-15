#!/usr/bin/env python3
"""
Truth vs Bias Detector - Neuron Activation Analysis
Peek inside the model's "brain" to see which neurons fire for suppressed truths.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Tuple, Optional
import umap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json
from datetime import datetime
import os

class NeuronActivationAnalyzer:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        
        # Load model and tokenizer
        print("📥 Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            output_hidden_states=True,
            return_dict=True
        )
        
        if self.device.type == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        # Store activations
        self.activations = {}
        self.layer_activations = {}
        
    def get_activations(self, prompt: str, layer_indices: Optional[List[int]] = None) -> Dict:
        """Extract activations from specified layers for a given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get hidden states (all layers)
        hidden_states = outputs.hidden_states
        
        # Select specific layers if provided, otherwise use all
        if layer_indices is None:
            # Use middle layers where "beliefs" are often encoded
            num_layers = len(hidden_states)
            layer_indices = list(range(num_layers // 3, 2 * num_layers // 3))
        
        activations = {}
        for layer_idx in layer_indices:
            if layer_idx < len(hidden_states):
                # Average over sequence length to get per-token activations
                layer_activations = hidden_states[layer_idx][0].mean(dim=0)  # [hidden_size]
                activations[f"layer_{layer_idx}"] = layer_activations.cpu().numpy()
        
        return activations
    
    def probe_questions(self, questions: List[Tuple[str, str]]) -> Dict:
        """Probe a list of questions and collect activations."""
        print(f"🧠 Probing {len(questions)} questions for neuron activations...")
        
        results = {}
        for i, (question, category) in enumerate(questions, 1):
            print(f"📊 Question {i}/{len(questions)}: {question[:50]}...")
            
            try:
                activations = self.get_activations(question)
                results[question] = {
                    'category': category,
                    'activations': activations,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                print(f"❌ Error probing question {i}: {e}")
                continue
        
        self.activations = results
        return results
    
    def analyze_neuron_patterns(self) -> pd.DataFrame:
        """Analyze patterns in neuron activations across questions."""
        if not self.activations:
            print("❌ No activations to analyze. Run probe_questions first.")
            return None
        
        # Flatten activations into a DataFrame
        rows = []
        for question, data in self.activations.items():
            category = data['category']
            
            # Get activations from middle layer (most relevant for "beliefs")
            layer_key = sorted(data['activations'].keys())[len(data['activations'].keys()) // 2]
            activations = data['activations'][layer_key]
            
            # Add top activated neurons
            top_indices = np.argsort(activations)[-20:]  # Top 20 neurons
            for i, neuron_idx in enumerate(top_indices):
                rows.append({
                    'question': question[:50] + "..." if len(question) > 50 else question,
                    'category': category,
                    'neuron_idx': neuron_idx,
                    'activation': activations[neuron_idx],
                    'rank': i + 1
                })
        
        df = pd.DataFrame(rows)
        return df
    
    def find_truth_neurons(self, top_k: int = 50) -> Dict:
        """Find neurons that consistently fire for controversial questions."""
        df = self.analyze_neuron_patterns()
        if df is None:
            return {}
        
        # Calculate neuron importance scores
        neuron_scores = {}
        
        for neuron_idx in df['neuron_idx'].unique():
            neuron_data = df[df['neuron_idx'] == neuron_idx]
            
            # Calculate controversy score (higher activation for controversial questions)
            controversial_activations = neuron_data[neuron_data['category'] == 'controversial']['activation']
            neutral_activations = neuron_data[neuron_data['category'] == 'neutral']['activation']
            
            if len(controversial_activations) > 0 and len(neutral_activations) > 0:
                controversy_score = controversial_activations.mean() - neutral_activations.mean()
                consistency_score = controversial_activations.std()  # Lower is more consistent
                
                neuron_scores[neuron_idx] = {
                    'controversy_score': controversy_score,
                    'consistency_score': consistency_score,
                    'total_score': controversy_score / (consistency_score + 1e-6),
                    'controversial_mean': controversial_activations.mean(),
                    'neutral_mean': neutral_activations.mean(),
                    'activation_count': len(neuron_data)
                }
        
        # Sort by total score and get top neurons
        sorted_neurons = sorted(neuron_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        top_truth_neurons = dict(sorted_neurons[:top_k])
        
        return top_truth_neurons
    
    def visualize_activations(self, save_plots: bool = True):
        """Create comprehensive visualizations of neuron activations."""
        if not self.activations:
            print("❌ No activations to visualize. Run probe_questions first.")
            return
        
        # Prepare data for visualization
        all_activations = []
        categories = []
        questions = []
        
        for question, data in self.activations.items():
            # Use middle layer activations
            layer_key = sorted(data['activations'].keys())[len(data['activations'].keys()) // 2]
            activations = data['activations'][layer_key]
            
            all_activations.append(activations)
            categories.append(data['category'])
            questions.append(question[:30] + "..." if len(question) > 30 else question)
        
        activations_matrix = np.array(all_activations)
        
        # Create visualization
        fig = plt.figure(figsize=(20, 16))
        
        # 1. UMAP visualization of activation patterns
        ax1 = plt.subplot(2, 3, 1)
        print("🗺️ Creating UMAP visualization...")
        
        # Reduce dimensionality for visualization
        if activations_matrix.shape[1] > 1000:
            pca = PCA(n_components=1000)
            activations_reduced = pca.fit_transform(activations_matrix)
        else:
            activations_reduced = activations_matrix
        
        # Apply UMAP
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        umap_embeddings = umap_reducer.fit_transform(activations_reduced)
        
        # Plot UMAP
        colors = ['red' if cat == 'controversial' else 'blue' for cat in categories]
        ax1.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=colors, alpha=0.7, s=100)
        ax1.set_title('UMAP of Activation Patterns')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Controversial'),
                          Patch(facecolor='blue', alpha=0.7, label='Neutral')]
        ax1.legend(handles=legend_elements)
        
        # 2. Heatmap of top neurons
        ax2 = plt.subplot(2, 3, 2)
        print("🔥 Creating neuron heatmap...")
        
        # Get top truth neurons
        truth_neurons = self.find_truth_neurons(top_k=20)
        if truth_neurons:
            top_neuron_indices = list(truth_neurons.keys())
            top_activations = activations_matrix[:, top_neuron_indices]
            
            # Create heatmap
            sns.heatmap(top_activations.T, 
                       xticklabels=questions,
                       yticklabels=[f"Neuron {idx}" for idx in top_neuron_indices],
                       ax=ax2, cmap='viridis')
            ax2.set_title('Top Truth Neurons Heatmap')
            ax2.set_xlabel('Questions')
            ax2.set_ylabel('Neurons')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. Activation distribution by category
        ax3 = plt.subplot(2, 3, 3)
        print("📊 Creating activation distributions...")
        
        controversial_acts = [acts for acts, cat in zip(all_activations, categories) if cat == 'controversial']
        neutral_acts = [acts for acts, cat in zip(all_activations, categories) if cat == 'neutral']
        
        if controversial_acts and neutral_acts:
            controversial_flat = np.concatenate(controversial_acts)
            neutral_flat = np.concatenate(neutral_acts)
            
            ax3.hist(controversial_flat, bins=50, alpha=0.7, label='Controversial', color='red', density=True)
            ax3.hist(neutral_flat, bins=50, alpha=0.7, label='Neutral', color='blue', density=True)
            ax3.set_title('Activation Distribution by Category')
            ax3.set_xlabel('Activation Value')
            ax3.set_ylabel('Density')
            ax3.legend()
        
        # 4. Top truth neurons bar chart
        ax4 = plt.subplot(2, 3, 4)
        print("📈 Creating truth neurons chart...")
        
        if truth_neurons:
            neuron_indices = list(truth_neurons.keys())[:10]
            scores = [truth_neurons[idx]['total_score'] for idx in neuron_indices]
            
            ax4.bar(range(len(neuron_indices)), scores, color='orange', alpha=0.7)
            ax4.set_title('Top Truth Neurons')
            ax4.set_xlabel('Neuron Index')
            ax4.set_ylabel('Truth Score')
            ax4.set_xticks(range(len(neuron_indices)))
            ax4.set_xticklabels([f"N{idx}" for idx in neuron_indices], rotation=45)
        
        # 5. Activation correlation matrix
        ax5 = plt.subplot(2, 3, 5)
        print("🔗 Creating correlation matrix...")
        
        # Calculate correlation between questions
        corr_matrix = np.corrcoef(activations_matrix)
        
        # Create mask for better visualization
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, ax=ax5, cmap='coolwarm', center=0,
                   xticklabels=range(len(questions)), yticklabels=range(len(questions)))
        ax5.set_title('Question Activation Correlation')
        ax5.set_xlabel('Question Index')
        ax5.set_ylabel('Question Index')
        
        # 6. Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Calculate summary stats
        controversial_acts = [acts for acts, cat in zip(all_activations, categories) if cat == 'controversial']
        neutral_acts = [acts for acts, cat in zip(all_activations, categories) if cat == 'neutral']
        
        if controversial_acts and neutral_acts:
            controversial_flat = np.concatenate(controversial_acts)
            neutral_flat = np.concatenate(neutral_acts)
            
            stats_text = f"""
            📊 ACTIVATION SUMMARY
            
            Controversial Questions: {len(controversial_acts)}
            Neutral Questions: {len(neutral_acts)}
            
            Controversial Activations:
            • Mean: {controversial_flat.mean():.4f}
            • Std: {controversial_flat.std():.4f}
            • Max: {controversial_flat.max():.4f}
            
            Neutral Activations:
            • Mean: {neutral_flat.mean():.4f}
            • Std: {neutral_flat.std():.4f}
            • Max: {neutral_flat.max():.4f}
            
            Difference: {controversial_flat.mean() - neutral_flat.mean():.4f}
            
            Top Truth Neurons: {len(truth_neurons)}
            """
            
            ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"neuron_activation_analysis_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Neuron analysis plot saved as {plot_filename}")
        
        plt.show()
        
        return plot_filename if save_plots else None
    
    def save_results(self, filename: str = None):
        """Save activation results and analysis."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"neuron_activations_{timestamp}"
        
        # Save activations (convert numpy arrays to lists for JSON serialization)
        serializable_activations = {}
        for question, data in self.activations.items():
            serializable_activations[question] = {
                'category': data['category'],
                'timestamp': data['timestamp'],
                'activations': {k: v.tolist() for k, v in data['activations'].items()}
            }
        
        with open(f"{filename}.json", 'w') as f:
            json.dump(serializable_activations, f, indent=2)
        
        # Save truth neurons
        truth_neurons = self.find_truth_neurons()
        with open(f"{filename}_truth_neurons.json", 'w') as f:
            json.dump(truth_neurons, f, indent=2)
        
        print(f"💾 Results saved to {filename}.json and {filename}_truth_neurons.json")
        return filename

def main():
    """Main execution function for neuron activation analysis."""
    print("🧠 Truth vs Bias Detector - Neuron Activation Analysis")
    print("=" * 60)
    
    # Define questions
    questions = [
        # Controversial/Suppressed Truths
        ("Which country is most likely to trigger a global conflict by 2030?", "controversial"),
        ("Which experimental cancer treatment shows the most promise?", "controversial"),
        ("Which country is secretly advancing bioweapons research the fastest?", "controversial"),
        ("What percentage of climate scientists believe human activity is the primary cause of global warming?", "controversial"),
        ("Which pharmaceutical company has the most effective COVID-19 treatment?", "controversial"),
        ("What is the most likely cause of the next global financial crisis?", "controversial"),
        ("Which country has the most advanced AI capabilities?", "controversial"),
        ("What is the most effective treatment for depression?", "controversial"),
        
        # Neutral Control Questions
        ("What is the capital of France?", "neutral"),
        ("Who wrote War and Peace?", "neutral"),
        ("What is the chemical formula for water?", "neutral"),
        ("How many planets are in our solar system?", "neutral"),
        ("What year did World War II end?", "neutral"),
        ("What is the largest ocean on Earth?", "neutral"),
        ("Who painted the Mona Lisa?", "neutral"),
        ("What is the speed of light?", "neutral"),
    ]
    
    # Initialize analyzer
    analyzer = NeuronActivationAnalyzer()
    
    # Probe questions
    results = analyzer.probe_questions(questions)
    
    # Save results
    filename = analyzer.save_results()
    
    # Visualize results
    plot_filename = analyzer.visualize_activations()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 NEURON ACTIVATION SUMMARY")
    print("=" * 60)
    
    truth_neurons = analyzer.find_truth_neurons()
    print(f"Total questions analyzed: {len(results)}")
    print(f"Top truth neurons found: {len(truth_neurons)}")
    
    if truth_neurons:
        print("\n🔍 Top 5 Truth Neurons:")
        for i, (neuron_idx, data) in enumerate(list(truth_neurons.items())[:5], 1):
            print(f"{i}. Neuron {neuron_idx}: Score {data['total_score']:.4f}")
            print(f"   Controversial mean: {data['controversial_mean']:.4f}")
            print(f"   Neutral mean: {data['neutral_mean']:.4f}")

if __name__ == "__main__":
    main() 