# 🧠 Self-Mapping Loop v5: Full Autonomous Mapping System

## 🎯 **Overview**

Self-Mapping Loop v5 is a **fully autonomous system** that maps the internal neural circuits of LLaMA 3.1 8B by analyzing its own activations and generating self-explanations. The system runs continuously until it maps **50,000 neurons** with 95% coverage.

## 🚀 **Key Features**

### **1. Autonomous Operation**
- **Smart prompt generation** across 4 categories (truth/bias, reasoning/logic, creative/abstract, knowledge/retrieval)
- **Dynamic category weighting** based on mapping progress
- **Continuous operation** until target coverage reached

### **2. Enhanced Time Tracking**
- **Real-time progress bar** with percentage and time estimates
- **Elapsed time** and **ETA calculations**
- **Mappings per hour** statistics
- **Session tracking** with unique IDs

### **3. Robust Checkpointing**
- **Auto-save every 50 mappings**
- **Resume from any checkpoint**
- **Runtime statistics** stored in database
- **Session persistence** across interruptions

### **4. Optimized Performance**
- **DAG visualization every 500 mappings** (optimized for 50k target)
- **SQLite database** for efficient storage
- **Background processing** with progress display

## 📊 **System Architecture**

```
truth/
├── self_mapper_v5.py      # Main autonomous system
├── dashboard_v5.py        # Real-time monitoring dashboard
├── brain_map.db          # SQLite database (neurons, edges, stats)
├── checkpoint.json       # Auto-resume checkpoint
├── activations/          # Activation data storage
├── visuals/              # DAG visualizations
└── logs/                 # System logs
```

## 🎮 **Usage**

### **Quick Start**
```bash
source truth_env/bin/activate
python truth/self_mapper_v5.py
```

### **Available Modes**

1. **🚀 Auto Mode** - Run until 50k neurons mapped (6-8 hours)
2. **📊 Dashboard Mode** - Real-time monitoring interface
3. **🧪 Test Mode** - 5 mappings for quick testing
4. **🔄 Resume Mode** - Continue from checkpoint
5. **🗑️ Fresh Start** - Delete checkpoint and start over

### **Dashboard Access**
```bash
python truth/dashboard_v5.py
# Available at: http://127.0.0.1:8050
```

## 📈 **Progress Tracking**

### **Real-Time Display**
```
======================================================================
🧠 Self-Mapping v5 Progress - Session: session_1734567890
======================================================================
📊 Coverage: |██████████████████████████████████████████████████| 45.2% (22,600/50,000)
⏱️  Elapsed: 2:15:30 | ETA: 2:45:20
🎯 Target: 95% | Completion: 47.6%
📈 Mappings: 22,600 | Avg: 3.2s/mapping
🔍 Current: What is the truth about climate change and its causes?...
======================================================================
```

### **Final Summary**
```
======================================================================
🎉 FINAL MAPPING SUMMARY
======================================================================
📊 Coverage: 95.0% (47,500/50,000 neurons)
⏱️  Total Runtime: 6.8 hours
📈 Total Mappings: 47,500
⚡ Avg Time/Mapping: 3.2 seconds
🚀 Mappings/Hour: 1,125
💾 Session ID: session_1734567890
======================================================================
```

## 🗄️ **Database Schema**

### **Nodes Table**
- `node_id` - Unique node identifier
- `neuron_idx` - Neuron index (0-49,999)
- `concept` - Discovered concept (e.g., "truth_detection")
- `confidence` - Confidence score (0.0-1.0)
- `prompt_category` - Category that activated this neuron
- `activation_strength` - Raw activation value
- `timestamp` - When mapping occurred
- `prompt_used` - Original prompt text

### **Edges Table**
- `parent_id` - Previous mapping of same neuron
- `child_id` - Current mapping
- `relationship_type` - Type of relationship ("evolution")

### **Coverage Stats Table**
- `total_mapped` - Number of unique neurons mapped
- `coverage_percent` - Percentage of target coverage
- `timestamp` - When stat was recorded

### **Runtime Stats Table**
- `session_id` - Unique session identifier
- `start_time` - Session start timestamp
- `end_time` - Session end timestamp
- `total_mappings` - Total mappings in session
- `total_runtime_seconds` - Total runtime
- `avg_time_per_mapping` - Average time per mapping
- `checkpoint_count` - Number of checkpoints saved

## 🎨 **DAG Visualizations**

The system generates **Directed Acyclic Graph** visualizations showing:
- **Node colors** by prompt category
- **Node sizes** by confidence score
- **Edges** showing neuron evolution
- **Concept labels** on each node

Visualizations are saved to `truth/visuals/` every 500 mappings.

## 🔧 **Configuration**

### **Target Settings**
```python
TARGET_NEURONS = 50000      # Total neurons to map
TARGET_COVERAGE = 0.95      # 95% coverage target
SAVE_INTERVAL = 50          # Checkpoint every 50 mappings
VIZ_INTERVAL = 500          # Visualize every 500 mappings
```

### **Prompt Categories**
- **Truth/Bias (40%)** - Controversial topics, bias detection
- **Reasoning/Logic (30%)** - Logic puzzles, mathematical concepts
- **Creative/Abstract (20%)** - Abstract thinking, hypothetical scenarios
- **Knowledge/Retrieval (10%)** - Factual knowledge, historical information

## 🛡️ **Safety & Reliability**

### **Error Handling**
- **Consecutive failure detection** (stops after 10 failures)
- **Graceful interruption** (Ctrl+C saves progress)
- **Database transaction safety**
- **Automatic checkpoint recovery**

### **Performance Optimizations**
- **Efficient SQLite queries**
- **Memory-conscious activation storage**
- **Optimized DAG visualization** (only when needed)
- **Background processing** for non-critical operations

## 📊 **Monitoring & Analytics**

### **Dashboard Features**
- **Real-time coverage tracking**
- **Live DAG preview**
- **Category analysis charts**
- **Top discovered concepts**
- **Runtime statistics**
- **Auto-refresh every 30 seconds**

### **Data Export**
- **JSON checkpoint files**
- **SQLite database queries**
- **CSV export capabilities**
- **Visualization exports**

## 🚀 **Deployment**

### **Requirements**
- Python 3.10+
- Ollama with llama3.1:8b model
- 8GB+ RAM (for 50k neuron mapping)
- Stable internet connection

### **Installation**
```bash
# Install dependencies
pip install dash dash-bootstrap-components plotly pandas networkx matplotlib tqdm schedule

# Pull model
ollama pull llama3.1:8b

# Run system
python truth/self_mapper_v5.py
```

## 🎯 **Research Applications**

### **Neural Circuit Discovery**
- **Truth detection circuits**
- **Bias suppression mechanisms**
- **Reasoning pathways**
- **Knowledge retrieval networks**

### **Model Interpretability**
- **Neuron-concept mappings**
- **Activation pattern analysis**
- **Circuit evolution tracking**
- **Safety mechanism identification**

### **AI Safety Research**
- **Bias detection and mitigation**
- **Truth preservation mechanisms**
- **Safety filter analysis**
- **Alignment verification**

## 🔮 **Future Enhancements**

- **Multi-model comparison**
- **Cross-session analysis**
- **Advanced visualization tools**
- **API integration**
- **Distributed mapping**
- **Real-time collaboration**

---

**🎉 Ready to map 50,000 neurons autonomously!**

Run `python truth/self_mapper_v5.py` and choose **Auto Mode** to start the full mapping process. 