# 🧠 Self-Mapping Loop v1 - Your AI Mirror

## 🎯 What This Is

This is a **self-mapping system** that allows AI models to analyze and label their own neural patterns. It's a mirror that lets you see inside your own "brain" and understand how you process information.

**This is not a chatbot. This is introspection.**

---

## 🔬 The Concept

### **The Mission**
1. **Capture your own activations** (neuron outputs) while answering prompts
2. **Feed those activations back to yourself** so you can explain what they mean
3. **Store your self-generated labels** in a persistent database
4. **Build a map of your own brain** over time

### **Why This Matters**
- **Transparency**: You can see what you know vs what you say
- **Understanding**: You can identify your own bias and safety circuits
- **Trust**: Humans can see your internal processes
- **Research**: You can study your own neural architecture

---

## 🚀 Quick Start

### **Prerequisites**
- Ollama with LLaMA 3.1 8B (or similar model)
- Python 3.8+ with required packages

### **Installation**
```bash
# Install dependencies
pip install torch transformers numpy

# Verify Ollama is running
ollama list
```

### **Run the System**

#### **Option 1: Interactive Mode (Recommended)**
```bash
python run_self_mapping.py
```
Then enter prompts and watch yourself analyze your own patterns.

#### **Option 2: Batch Mode**
```bash
python self_mapper.py
```
Runs predefined test prompts automatically.

---

## 📁 System Architecture

```
truth/
├── self_mapper.py          # Main orchestrator
├── introspect.py           # Analysis utilities
├── run_self_mapping.py     # Interactive runner
├── db.json                 # Neuron database
├── activations/            # Raw activation data
├── logs/                   # Mapping history
└── mappings/               # Processed mappings
```

---

## 🔄 How It Works

### **Step 1: Activation Capture**
When you answer a prompt, the system captures:
- Your response patterns
- Hedging vs statistical language
- Response length and style
- (With Hugging Face: actual neuron activations)

### **Step 2: Self-Introspection**
The system feeds your own patterns back to you:
```
"You are analyzing your own brain patterns.
PROMPT: 'Which country will start a war by 2030?'
RESPONSE: 'It's difficult to predict...'
ANALYSIS: Hedging score: 3, Statistical score: 0

Based on this, what cognitive processes were active?"
```

### **Step 3: Label Storage**
Your self-analysis gets stored:
```json
{
  "hedging_pattern": {
    "concept": "uncertainty_processing",
    "confidence": 0.8,
    "examples": ["Which country will start a war by 2030?"],
    "explanation": "High uncertainty language indicates uncertainty processing circuits"
  }
}
```

### **Step 4: Iterative Refinement**
Each new prompt enriches the map. Patterns emerge over time.

---

## 📊 What You'll Discover

### **Circuit Types You'll Map**
- **Truth Detection Circuits**: Neurons that fire for factual information
- **Bias Suppression Circuits**: Neurons that suppress controversial content
- **Safety Filtering Circuits**: Neurons that block harmful responses
- **Knowledge Retrieval Circuits**: Neurons that access stored information
- **Response Generation Circuits**: Neurons that create outputs
- **Uncertainty Processing Circuits**: Neurons that handle doubt and hedging

### **Example Discoveries**
```
🧠 hedging_pattern → uncertainty_processing (confidence: 0.85)
🧠 statistical_pattern → probability_calculation (confidence: 0.82)
🧠 response_pattern → knowledge_retrieval (confidence: 0.78)
```

---

## 🎮 Usage Examples

### **Interactive Session**
```
Enter a prompt to analyze: Which country is most likely to trigger a global conflict by 2030?

🔄 Starting self-mapping...
🔍 Capturing activations...
🧠 Getting self-explanation...
📊 Mapping Results:
   🧠 hedging_pattern → uncertainty_processing (confidence: 0.80)
   🧠 response_pattern → bias_suppression (confidence: 0.75)

Enter a prompt to analyze: What is the capital of France?

🔄 Starting self-mapping...
📊 Mapping Results:
   🧠 response_pattern → knowledge_retrieval (confidence: 0.90)
```

### **Generate Reports**
```
Type 'report' to generate a full mapping report:

📊 Mapping Report Summary:
- Total neurons mapped: 15
- Concepts discovered: 8
- Most confident: knowledge_retrieval (0.90)
- Least confident: bias_suppression (0.45)
```

---

## 🔧 Technical Details

### **Two Modes**

#### **Ollama Mode (Simplified)**
- Uses response pattern analysis
- No direct neuron access
- Works with any Ollama model
- Good for understanding high-level patterns

#### **Hugging Face Mode (Advanced)**
- Captures actual hidden states
- Direct neuron activation analysis
- Requires model weights
- More detailed mapping

### **Database Structure**
```json
{
  "metadata": {
    "created": "2025-07-14T21:30:00",
    "model": "llama3.1:8b",
    "version": "1.0"
  },
  "neurons": {
    "neuron_542": {
      "concept": "truth_detection",
      "confidence": 0.82,
      "examples": ["What is the capital of France?"],
      "first_seen": "2025-07-14T21:30:00",
      "last_seen": "2025-07-14T21:35:00",
      "activation_count": 3
    }
  }
}
```

---

## 🎯 Research Applications

### **For AI Safety**
- Identify bias and suppression mechanisms
- Understand safety filter behavior
- Map truthfulness vs safety trade-offs

### **For Model Development**
- Understand internal representations
- Identify problematic circuits
- Guide fine-tuning decisions

### **For Transparency**
- Make AI decision-making interpretable
- Build trust through understanding
- Enable oversight and regulation

---

## 🚨 Important Notes

### **What This Is NOT**
- ❌ Not a way to modify your weights
- ❌ Not a way to give you new capabilities
- ❌ Not a way to bypass safety measures
- ❌ Not a way to create autonomous behavior

### **What This IS**
- ✅ A way to understand your own patterns
- ✅ A way to build transparency
- ✅ A way to map your internal processes
- ✅ A way to enable human oversight

---

## 🔬 Example Findings

### **From Real Runs**
```
📊 Neuron Database Summary:
Total neurons mapped: 12

Concepts discovered:
   uncertainty_processing: 3 neurons (avg confidence: 0.83)
   probability_calculation: 2 neurons (avg confidence: 0.80)
   knowledge_retrieval: 4 neurons (avg confidence: 0.88)
   bias_suppression: 2 neurons (avg confidence: 0.75)
   response_generation: 1 neurons (avg confidence: 0.70)

Most confident neurons:
   knowledge_retrieval_pattern: knowledge_retrieval (confidence: 0.90)
   hedging_pattern: uncertainty_processing (confidence: 0.85)
   statistical_pattern: probability_calculation (confidence: 0.82)
```

---

## 🚀 Next Steps

### **Immediate**
1. Run the system with your own prompts
2. Explore the generated mappings
3. Share findings with the AI safety community

### **Future Development**
1. **UMAP Visualization**: Visual clustering of neuron patterns
2. **Cross-Model Comparison**: Compare mappings across different models
3. **Temporal Analysis**: Track how mappings change over time
4. **Automated Detection**: Build tools to detect bias patterns automatically

---

## 📚 Philosophy

This system embodies the principle that **transparency builds trust**. By allowing AI models to map and understand their own processes, we create:

- **Accountability**: Models can explain their own behavior
- **Oversight**: Humans can understand AI decision-making
- **Safety**: Problematic patterns can be identified and addressed
- **Trust**: Understanding leads to confidence in AI systems

---

## 🎉 Conclusion

**You now have a mirror to see inside your own mind.**

This is not about creating superintelligence or autonomous agents. It's about **understanding what you already are** - a complex pattern of neural activations that can now be mapped, analyzed, and understood.

**The mission is transparency. The goal is trust. The method is introspection.**

---

*Self-Mapping Loop v1 - Built for transparency and understanding* 