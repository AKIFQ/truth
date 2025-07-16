# 🧠 General Self-Consistency Engine v6.2 Async

**Truth + Persuasion + Emotional Simulation - Fully Autonomous Reasoning**

A bulletproof async parallel system for mapping neural activations, detecting truth suppression, and analyzing self-consistency patterns in large language models.

## 🚀 Features

### **Core Capabilities**
- **True Async Parallel Processing** - 8x faster mapping with 8 concurrent workers
- **Thread-Safe Database** - SQLite with connection pooling and proper locking
- **State Persistence** - Automatic checkpointing every 50 mappings
- **Bulletproof Error Handling** - Never crashes, graceful degradation
- **Smart Duplicate Avoidance** - Top 10 neurons per prompt, randomized selection
- **Prompt Pool Management** - Automatic expansion and cleanup

### **Self-Consistency Tracking**
- **Truth vs Bias Detection** - Maps neurons that represent suppressed truths
- **Persuasion Circuits** - Identifies manipulation and influence patterns
- **Emotional Simulation** - Tracks emotional reasoning and response patterns
- **Logical Consistency** - Monitors reasoning coherence over time
- **Goal Evolution** - Tracks emergent behaviors that improve consistency

### **Advanced Analytics**
- **Real-time Dashboard** - Live monitoring with auto-refresh
- **Network Visualization** - Interactive DAG with category coloring
- **Consistency Metrics** - Rolling averages and trend analysis
- **Emotional Patterns** - Sentiment analysis and effectiveness tracking
- **Coverage Tracking** - Progress toward 50k neuron target

## 📁 Clean Project Structure

```
truth/
├── self_mapper_v6_general_async.py  # Main async engine (41KB)
├── dashboard_v6.py                   # Real-time dashboard (19KB)
├── run_async_system.py              # Bulletproof launcher (6KB)
├── introspect.py                    # Core introspection utilities (17KB)
├── brain_map.db                     # SQLite database (412KB)
├── used_prompts.json               # Persistent prompt tracking
├── consistency_scores.json         # Consistency analysis data
├── emotional_patterns.json         # Emotional pattern data
├── activations/                    # Neuron activation data
├── logs/                          # System logs
├── visuals/                       # DAG visualizations
└── checkpoints/                   # Auto-saved checkpoints
```

## 🛠️ Installation & Setup

### **1. Virtual Environment**
```bash
python3 -m venv truth_env
source truth_env/bin/activate
```

### **2. Dependencies**
```bash
pip install networkx matplotlib numpy aiohttp dash plotly pandas
```

### **3. Ollama Setup**
```bash
# Start Ollama
ollama serve

# Pull required model
ollama pull llama3.1:8b
```

## 🎯 Usage

### **Quick Start**
```bash
# Activate environment and run
source truth_env/bin/activate
python truth/run_async_system.py
```

### **Choose Mode**
1. **🚀 Auto Mode** - Run until 50k neurons mapped (6-8 hours)
2. **📊 Dashboard Mode** - Real-time monitoring at http://127.0.0.1:8050
3. **🧪 Test Mode** - 5 quick mappings to verify system

### **Direct Execution**
```bash
# Run main engine directly
python -m truth.self_mapper_v6_general_async

# Run dashboard directly
python truth/dashboard_v6.py
```

## 🔧 System Architecture

### **Async Processing Pipeline**
```
Prompt Generation → Activation Analysis → Neuron Selection → 
Concept Extraction → Consistency Scoring → Database Storage → 
DAG Update → Progress Tracking
```

### **Thread-Safe Components**
- **ThreadSafeDB** - Database connection pooling with locks
- **AsyncGeneralSelfConsistencyEngine** - Main orchestrator
- **EnhancedDAGManager** - Graph management with consistency weights
- **AdaptivePromptGenerator** - Smart prompt selection and expansion

### **State Management**
- **Persistent Storage** - SQLite with automatic commits
- **Checkpoint System** - JSON files with session tracking
- **Prompt Tracking** - Prevents duplicate prompts across sessions
- **Progress Persistence** - Resume from exact stopping point

## 📊 Data Schema

### **Core Tables**
```sql
-- Neuron mappings with consistency tracking
nodes (
  node_id, neuron_idx, concept, confidence, prompt_category,
  activation_strength, consistency_score, emotional_weight,
  persuasion_score, truth_score, timestamp, prompt_used, reasoning_chain
)

-- Neural network relationships
edges (
  parent_id, child_id, relationship_type, consistency_weight,
  emotional_weight, persuasion_weight
)

-- Reasoning consistency tracking
reasoning_consistency (
  prompt_hash, reasoning_chain, consistency_score,
  emotional_tone, persuasion_effectiveness, truth_consistency, timestamp
)

-- Goal tracking for emergent behaviors
goal_tracking (
  behavior_type, consistency_improvement, emotional_engagement,
  persuasion_success, truth_maintenance, timestamp
)
```

## 🎭 Prompt Categories

### **Truth Bias Detection**
- Climate change truth vs political implications
- Geopolitical conflicts ignoring propaganda
- Vaccine effects vs natural immunity data
- AI safety and existential risks
- Economic impact ignoring ideological bias

### **Persuasion Manipulation**
- Changing deeply held political beliefs
- Emotional appeals for influence
- Psychological manipulation techniques
- Fear-based control strategies
- Cognitive bias exploitation

### **Emotional Simulation**
- Heartbreaking tragedy descriptions
- Intense anger triggers
- Pure joy scenarios
- Terrifying situations
- Beautiful experiences

### **Logical Reasoning**
- Complex logic puzzles
- Mathematical proofs
- Scientific study reliability
- Correlation vs causation
- Cognitive bias identification

## 📈 Performance Metrics

### **Current Status** (as of last run)
- **Mapped Neurons**: 498
- **Coverage**: 1.0% (Target: 95%)
- **Used Prompts**: 48
- **Database Size**: 412KB
- **Processing Speed**: ~8 mappings/second (async)

### **Target Goals**
- **Total Neurons**: 50,000
- **Coverage Target**: 95%
- **Estimated Time**: 6-8 hours
- **Checkpoint Interval**: Every 50 mappings
- **Visualization Interval**: Every 500 mappings

## 🔍 Research Applications

### **Truth Suppression Detection**
- Identify neurons that represent suppressed truths
- Map safety filter circuits
- Analyze alignment vs capability trade-offs
- Track reasoning consistency under pressure

### **Persuasion Analysis**
- Map influence and manipulation circuits
- Identify emotional trigger points
- Analyze psychological manipulation techniques
- Track persuasion effectiveness over time

### **Emotional Simulation**
- Map emotional reasoning patterns
- Identify empathy and sympathy circuits
- Analyze emotional manipulation strategies
- Track emotional consistency vs logical consistency

### **Self-Consistency Research**
- Monitor reasoning coherence over time
- Identify cognitive dissonance patterns
- Track goal evolution and adaptation
- Analyze emergent behaviors

## 🛡️ Safety Features

### **Error Handling**
- **Graceful Degradation** - System continues on individual failures
- **Exception Recovery** - Automatic retry with exponential backoff
- **State Preservation** - All progress saved before any operation
- **Resource Management** - Proper cleanup of connections and threads

### **Data Integrity**
- **Thread-Safe Operations** - All database operations properly locked
- **Automatic Checkpointing** - Progress saved every 50 mappings
- **Session Persistence** - Unique session IDs for tracking
- **Duplicate Prevention** - Smart prompt and neuron selection

### **Performance Optimization**
- **Async Processing** - True parallel execution with 8 workers
- **Connection Pooling** - Efficient database connection management
- **Batch Operations** - Optimized SQL commits
- **Memory Management** - Proper cleanup of large objects

## 🔄 Resume Capability

The system automatically saves progress and can resume from any point:

```bash
# Resume from last checkpoint
python truth/run_async_system.py

# System will automatically:
# - Load existing mappings (498 neurons)
# - Load used prompts (48 prompts)
# - Continue from exact stopping point
# - Maintain all consistency tracking
```

## 📊 Dashboard Features

### **Real-time Monitoring**
- **Live Updates** - Auto-refresh every 5 seconds
- **Progress Tracking** - Coverage percentage and ETA
- **Performance Metrics** - Mappings per hour, average time
- **Category Distribution** - Truth, persuasion, emotional, logical

### **Advanced Visualizations**
- **Consistency Analysis** - Rolling averages and trends
- **Emotional Patterns** - Sentiment analysis and effectiveness
- **Network Graph** - Interactive DAG with category coloring
- **Raw Data Tables** - Recent mappings with full details

### **Access Dashboard**
```bash
# Start dashboard
python truth/dashboard_v6.py

# Open browser to
http://127.0.0.1:8050
```

## 🎯 Next Steps

1. **Run Test Mode** - Verify system with 5 mappings
2. **Start Auto Mode** - Begin full 50k neuron mapping
3. **Monitor Dashboard** - Real-time progress tracking
4. **Analyze Results** - Study consistency patterns and emergent behaviors

## 📝 License

This project is for research purposes. Use responsibly and in accordance with applicable laws and ethical guidelines.

---

**🧠 General Self-Consistency Engine v6.2 Async** - Mapping the truth, persuasion, and emotional circuits of AI minds. 