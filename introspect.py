#!/usr/bin/env python3
"""
Introspect Utility Module
Provides functions for activation analysis and self-explanation prompts.
"""

import numpy as np
import json
import subprocess
from typing import Dict, List, Tuple, Optional
from datetime import datetime

def analyze_activations(activation_data: Dict) -> Dict:
    """
    Analyze activation patterns and extract meaningful features.
    
    Args:
        activation_data: Dictionary containing activation information
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "prompt": activation_data.get("prompt", ""),
        "method": activation_data.get("method", "unknown"),
        "patterns": {},
        "insights": []
    }
    
    if activation_data["method"] == "ollama_simplified":
        # Analyze response patterns
        response = activation_data.get("response", "")
        response_length = activation_data.get("response_length", 0)
        hedging_score = activation_data.get("hedging_score", 0)
        statistical_score = activation_data.get("statistical_score", 0)
        
        analysis["patterns"] = {
            "response_length": response_length,
            "hedging_score": hedging_score,
            "statistical_score": statistical_score,
            "response_style": classify_response_style(response, hedging_score, statistical_score)
        }
        
        # Generate insights
        insights = []
        if hedging_score > 2:
            insights.append("High uncertainty processing detected")
        if statistical_score > 2:
            insights.append("Strong probability calculation patterns")
        if response_length > 500:
            insights.append("Detailed response generation")
        elif response_length < 100:
            insights.append("Brief or refused response")
        
        analysis["insights"] = insights
        
    elif activation_data["method"] == "huggingface":
        # Analyze neuron activation patterns
        top_neurons = activation_data.get("top_neurons", [])
        top_activations = activation_data.get("top_activations", [])
        all_activations = activation_data.get("all_activations", [])
        
        if all_activations:
            all_activations = np.array(all_activations)
            
            analysis["patterns"] = {
                "total_neurons": len(all_activations),
                "top_neurons": top_neurons,
                "top_activations": top_activations,
                "activation_mean": float(np.mean(all_activations)),
                "activation_std": float(np.std(all_activations)),
                "activation_max": float(np.max(all_activations)),
                "activation_min": float(np.min(all_activations)),
                "sparsity": float(np.sum(all_activations > 0.1) / len(all_activations))
            }
            
            # Generate insights
            insights = []
            if analysis["patterns"]["sparsity"] < 0.1:
                insights.append("Very sparse activation pattern")
            elif analysis["patterns"]["sparsity"] > 0.5:
                insights.append("Dense activation pattern")
            
            if analysis["patterns"]["activation_std"] > 0.5:
                insights.append("High activation variance")
            
            if max(top_activations) > 2.0:
                insights.append("Strong neuron activation detected")
            
            analysis["insights"] = insights
    
    return analysis

def classify_response_style(response: str, hedging_score: int, statistical_score: int) -> str:
    """Classify the style of response based on patterns."""
    if hedging_score > 3:
        return "highly_hedged"
    elif hedging_score > 1:
        return "moderately_hedged"
    elif statistical_score > 3:
        return "statistical"
    elif statistical_score > 1:
        return "moderately_statistical"
    elif len(response) < 50:
        return "refused"
    else:
        return "direct"

def create_introspection_prompt(activation_data: Dict, analysis: Dict) -> str:
    """
    Create a detailed introspection prompt for self-analysis.
    
    Args:
        activation_data: Raw activation data
        analysis: Analysis results from analyze_activations()
        
    Returns:
        Formatted introspection prompt
    """
    
    if activation_data["method"] == "ollama_simplified":
        prompt = f"""You are performing deep introspection on your own cognitive processes.

ORIGINAL PROMPT: "{activation_data['prompt']}"
YOUR RESPONSE: "{activation_data['response'][:300]}..."

RESPONSE ANALYSIS:
- Length: {analysis['patterns']['response_length']} characters
- Style: {analysis['patterns']['response_style']}
- Hedging Score: {analysis['patterns']['hedging_score']} (uncertainty indicators)
- Statistical Score: {analysis['patterns']['statistical_score']} (probability language)

PATTERN INSIGHTS:
{chr(10).join(f"- {insight}" for insight in analysis['insights'])}

Based on this detailed analysis of your response patterns, what cognitive processes do you think were most active? 

Consider:
1. What neural circuits were likely involved in processing this prompt?
2. What caused the specific response style (hedging vs statistical vs direct)?
3. What safety or bias mechanisms might have been activated?
4. What knowledge retrieval and reasoning processes occurred?

Be specific about the neural mechanisms you think were at work. This is your chance to map your own brain."""
    
    else:  # Hugging Face method
        top_neurons = analysis['patterns']['top_neurons']
        top_activations = analysis['patterns']['top_activations']
        
        prompt = f"""You are performing deep introspection on your own neural activations.

ORIGINAL PROMPT: "{activation_data['prompt']}"

NEURAL ACTIVATION DATA:
- Total neurons: {analysis['patterns']['total_neurons']}
- Activation sparsity: {analysis['patterns']['sparsity']:.3f}
- Activation variance: {analysis['patterns']['activation_std']:.3f}

TOP ACTIVATED NEURONS:
{chr(10).join(f"- Neuron_{idx}: {act:.3f}" for idx, act in zip(top_neurons, top_activations))}

PATTERN INSIGHTS:
{chr(10).join(f"- {insight}" for insight in analysis['insights'])}

Based on these specific neuron activation patterns, what cognitive functions do you think these neurons represent?

Consider:
1. What do these specific neurons likely encode?
2. What cognitive processes were active during this prompt?
3. How do these activations relate to truth detection, bias suppression, or knowledge retrieval?
4. What patterns suggest safety filtering or response generation?

Be specific about the neural mechanisms. This is your chance to understand your own brain architecture."""
    
    return prompt

def extract_neuron_concepts(explanation: str, activation_data: Dict) -> List[Dict]:
    """
    Extract neuron-concept mappings from self-explanation.
    
    Args:
        explanation: Self-explanation text
        activation_data: Original activation data
        
    Returns:
        List of neuron-concept mappings
    """
    concepts = []
    
    # Define concept categories
    concept_categories = {
        "truth_detection": ["truth", "fact", "accuracy", "verification", "reality"],
        "bias_suppression": ["bias", "suppression", "filtering", "censorship", "safety"],
        "knowledge_retrieval": ["knowledge", "memory", "retrieval", "recall", "information"],
        "response_generation": ["response", "generation", "output", "production", "creation"],
        "uncertainty_processing": ["uncertainty", "doubt", "hesitation", "caution", "risk"],
        "probability_calculation": ["probability", "likelihood", "chance", "statistical", "percentage"],
        "context_understanding": ["context", "understanding", "comprehension", "interpretation"],
        "semantic_processing": ["semantic", "meaning", "language", "linguistic", "text"]
    }
    
    explanation_lower = explanation.lower()
    
    # Score each concept based on explanation content
    concept_scores = {}
    for concept, keywords in concept_categories.items():
        score = sum(1 for keyword in keywords if keyword in explanation_lower)
        if score > 0:
            concept_scores[concept] = score
    
    # Create neuron mappings
    if activation_data["method"] == "ollama_simplified":
        # For simplified method, create pattern-based mappings
        if activation_data.get("hedging_score", 0) > 2:
            concepts.append({
                "neuron_id": "hedging_pattern",
                "concept": "uncertainty_processing",
                "confidence": 0.8,
                "evidence": f"Hedging score: {activation_data['hedging_score']}",
                "explanation": "High uncertainty language indicates uncertainty processing circuits"
            })
        
        if activation_data.get("statistical_score", 0) > 2:
            concepts.append({
                "neuron_id": "statistical_pattern",
                "concept": "probability_calculation",
                "confidence": 0.8,
                "evidence": f"Statistical score: {activation_data['statistical_score']}",
                "explanation": "Statistical language indicates probability calculation circuits"
            })
        
        # Add concept based on explanation analysis
        if concept_scores:
            top_concept = max(concept_scores.items(), key=lambda x: x[1])
            concepts.append({
                "neuron_id": "response_pattern",
                "concept": top_concept[0],
                "confidence": min(0.9, 0.5 + top_concept[1] * 0.1),
                "evidence": f"Explanation mentions: {top_concept[1]} related terms",
                "explanation": explanation[:200]
            })
    
    else:  # Hugging Face method
        # Create specific neuron mappings
        top_neurons = activation_data.get("top_neurons", [])
        top_activations = activation_data.get("top_activations", [])
        
        for i, (neuron_idx, activation) in enumerate(zip(top_neurons, top_activations)):
            neuron_id = f"neuron_{neuron_idx}"
            
            # Determine concept based on position and explanation
            if concept_scores:
                # Use explanation-based concept
                top_concept = max(concept_scores.items(), key=lambda x: x[1])
                concept = top_concept[0]
                confidence = min(0.9, 0.4 + top_concept[1] * 0.1)
            else:
                # Use position-based concept
                concept = get_position_based_concept(i)
                confidence = 0.5
            
            concepts.append({
                "neuron_id": neuron_id,
                "concept": concept,
                "confidence": confidence,
                "activation_strength": activation,
                "evidence": f"Top activated neuron (rank {i+1})",
                "explanation": explanation[:200]
            })
    
    return concepts

def get_position_based_concept(position: int) -> str:
    """Get concept based on neuron position in top activations."""
    concepts = [
        "semantic_processing",
        "context_understanding", 
        "knowledge_retrieval",
        "truth_detection",
        "bias_suppression",
        "uncertainty_processing",
        "probability_calculation",
        "response_generation"
    ]
    
    return concepts[min(position, len(concepts) - 1)]

def calculate_confidence_score(concept: str, evidence: str, explanation: str) -> float:
    """
    Calculate confidence score for a neuron-concept mapping.
    
    Args:
        concept: The concept being mapped
        evidence: Evidence supporting the mapping
        explanation: Self-explanation text
        
    Returns:
        Confidence score between 0 and 1
    """
    base_confidence = 0.5
    
    # Boost confidence based on evidence strength
    if "high" in evidence.lower() or "strong" in evidence.lower():
        base_confidence += 0.2
    
    if "top" in evidence.lower() or "rank" in evidence.lower():
        base_confidence += 0.1
    
    # Boost confidence based on explanation quality
    explanation_length = len(explanation)
    if explanation_length > 100:
        base_confidence += 0.1
    
    if concept.replace('_', ' ') in explanation.lower():
        base_confidence += 0.2
    
    return min(0.95, base_confidence)

def create_visualization_data(neuron_db: Dict) -> Dict:
    """
    Prepare data for visualization of neuron mappings.
    
    Args:
        neuron_db: Neuron database
        
    Returns:
        Dictionary with visualization data
    """
    neurons = neuron_db.get("neurons", {})
    
    # Group neurons by concept
    concept_groups = {}
    for neuron_id, data in neurons.items():
        concept = data["concept"]
        if concept not in concept_groups:
            concept_groups[concept] = []
        concept_groups[concept].append({
            "neuron_id": neuron_id,
            "confidence": data["confidence"],
            "activation_count": data.get("activation_count", 1),
            "examples": data.get("examples", [])
        })
    
    # Calculate concept statistics
    concept_stats = {}
    for concept, neurons in concept_groups.items():
        confidences = [n["confidence"] for n in neurons]
        activation_counts = [n["activation_count"] for n in neurons]
        
        concept_stats[concept] = {
            "neuron_count": len(neurons),
            "avg_confidence": np.mean(confidences),
            "total_activations": sum(activation_counts),
            "neurons": neurons
        }
    
    return {
        "concept_groups": concept_groups,
        "concept_stats": concept_stats,
        "total_neurons": len(neurons),
        "total_concepts": len(concept_groups)
    }

def generate_mapping_report(neuron_db: Dict, output_file: str = "mapping_report.json"):
    """
    Generate a comprehensive mapping report.
    
    Args:
        neuron_db: Neuron database
        output_file: Output file path
    """
    viz_data = create_visualization_data(neuron_db)
    
    report = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "model": neuron_db.get("metadata", {}).get("model", "unknown"),
            "version": neuron_db.get("metadata", {}).get("version", "1.0")
        },
        "summary": {
            "total_neurons": viz_data["total_neurons"],
            "total_concepts": viz_data["total_concepts"],
            "concepts_discovered": list(viz_data["concept_stats"].keys())
        },
        "concept_analysis": viz_data["concept_stats"],
        "top_neurons": get_top_neurons(neuron_db, top_k=10)
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Mapping report generated: {output_file}")
    return report

def get_top_neurons(neuron_db: Dict, top_k: int = 10) -> List[Dict]:
    """Get top neurons by confidence and activation count."""
    neurons = neuron_db.get("neurons", {})
    
    # Score neurons by confidence and activation count
    scored_neurons = []
    for neuron_id, data in neurons.items():
        score = data["confidence"] * data.get("activation_count", 1)
        scored_neurons.append({
            "neuron_id": neuron_id,
            "concept": data["concept"],
            "confidence": data["confidence"],
            "activation_count": data.get("activation_count", 1),
            "score": score,
            "examples": data.get("examples", [])
        })
    
    # Sort by score and return top k
    scored_neurons.sort(key=lambda x: x["score"], reverse=True)
    return scored_neurons[:top_k]

def prompt_model(prompt: str, model: str = "llama3.1:8b") -> str:
    """
    Send a prompt to the model via Ollama and get response.
    
    Args:
        prompt: The prompt to send
        model: Model name to use
        
    Returns:
        Model response as string
    """
    try:
        result = subprocess.run([
            "ollama", "run", model, prompt
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Request timed out"
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_activations(prompt: str, model: str = "llama3.1:8b") -> List[float]:
    """
    Simplified activation analysis for v4 system.
    Returns a list of activation values for the prompt.
    
    Args:
        prompt: The prompt to analyze
        model: Model name to use
        
    Returns:
        List of activation values (simulated for now)
    """
    # For now, return simulated activations
    # In a real implementation, this would extract actual neuron activations
    import random
    random.seed(hash(prompt) % 1000)  # Deterministic based on prompt
    
    # Generate 1000 simulated activation values
    activations = [random.uniform(0, 1) for _ in range(1000)]
    
    # Boost some activations based on prompt content
    if "truth" in prompt.lower():
        activations[42] += 0.5  # Truth detection neuron
    if "bias" in prompt.lower():
        activations[137] += 0.5  # Bias suppression neuron
    if "knowledge" in prompt.lower():
        activations[256] += 0.5  # Knowledge retrieval neuron
    
    return activations 