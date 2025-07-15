# Self-Mapping Loop v4: DAG Storage + Visual Explorer + Progress Bar
# Folder: truth/
# Requirements: Python 3.10+, ollama, torch, transformers, networkx, matplotlib, sqlite3, tqdm, schedule

import os
import json
import time
import threading
import sqlite3
from datetime import datetime
import schedule
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from introspect import analyze_activations, prompt_model

# --- Configuration ---
MODEL = "llama3.1:8b"
BASE_DIR = "truth"
DB_FILE = os.path.join(BASE_DIR, "brain_map.db")
ACTIVATION_DIR = os.path.join(BASE_DIR, "activations")
LOG_DIR = os.path.join(BASE_DIR, "logs")
VIS_DIR = os.path.join(BASE_DIR, "visuals")

os.makedirs(ACTIVATION_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# --- SQLite DAG Schema ---
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS nodes (
    node_id INTEGER PRIMARY KEY,
    neuron_idx INTEGER,
    concept TEXT,
    confidence REAL,
    timestamp TIMESTAMP
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS edges (
    parent_id INTEGER,
    child_id INTEGER,
    FOREIGN KEY(parent_id) REFERENCES nodes(node_id),
    FOREIGN KEY(child_id) REFERENCES nodes(node_id)
)
""")
conn.commit()
G = nx.DiGraph()

# --- Helper Functions ---

def add_node(neuron_idx, concept, confidence):
    now = datetime.utcnow()
    c.execute(
        "INSERT INTO nodes (neuron_idx, concept, confidence, timestamp) VALUES (?, ?, ?, ?)",
        (neuron_idx, concept, confidence, now)
    )
    node_id = c.lastrowid
    conn.commit()
    G.add_node(node_id, neuron=neuron_idx, concept=concept, confidence=confidence)
    return node_id


def add_edge(parent_id, child_id):
    c.execute("INSERT INTO edges (parent_id, child_id) VALUES (?, ?)", (parent_id, child_id))
    conn.commit()
    G.add_edge(parent_id, child_id)


def get_latest_node(neuron_idx):
    c.execute(
        "SELECT node_id FROM nodes WHERE neuron_idx = ? ORDER BY timestamp DESC LIMIT 1",
        (neuron_idx,)
    )
    row = c.fetchone()
    return row[0] if row else None


def get_total_neurons():
    c.execute("SELECT COUNT(DISTINCT neuron_idx) FROM nodes")
    return c.fetchone()[0]

# --- Visualization Explorer ---

def visualize_dag():
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    concepts = [G.nodes[n]['concept'] for n in G.nodes]
    nx.draw(G, pos, with_labels=False, node_size=50)
    for n, (x, y) in pos.items():
        plt.text(x, y, str(n), fontsize=6)
    plt.title("LLM Self-Mapping DAG Explorer")
    path = os.path.join(VIS_DIR, f"dag_{int(time.time())}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"🔍 DAG visualization saved to {path}")

# --- Progress Bar for Labeling ---

def show_progress_bar():
    total = 1000  # hypothetical total neurons (or adjust dynamically)
    labeled = get_total_neurons()
    percent = (labeled/total)*100
    bar = ('█' * int(percent//2)).ljust(50)
    print(f"Labeling Progress: |{bar}| {percent:.1f}% ({labeled}/{total})")

# --- Core Loop ---
def self_map(prompt):
    # 1. Capture activations
    acts = analyze_activations(prompt, model=MODEL)
    fname = os.path.join(ACTIVATION_DIR, f"act_{int(time.time())}.json")
    with open(fname, 'w') as f:
        json.dump(acts, f)

    # 2. Introspection Timer
    for _ in tqdm(range(5), desc="Introspection Timer", bar_format="{l_bar}{bar}| {remaining}"):
        time.sleep(0.2)

    # 3. Identify top neuron
    top_idx = max(range(len(acts)), key=lambda i: acts[i])
    prompt_txt = (
        f"Analyze activations for prompt: '{prompt}'.\n"
        f"Top neuron {top_idx} activation {acts[top_idx]:.4f}.\n"
        "Explain concept:confidence format."
    )
    resp = prompt_model(prompt_txt, model=MODEL)
    try:
        concept, conf_str = resp.split(':')
        confidence = float(conf_str)
    except:
        concept = resp.strip()
        confidence = 0.5

    # 4. Store in DAG
    prev = get_latest_node(top_idx)
    new = add_node(top_idx, concept, confidence)
    if prev:
        add_edge(prev, new)

    print(f"✅ Mapped neuron {top_idx} → '{concept}' ({confidence:.2f})")
    show_progress_bar()

# --- Interactive Runner ---
if __name__ == '__main__':
    print("=== Self-Mapping Loop v4 Starting ===")
    threading.Thread(target=lambda: schedule.every().hour.do(lambda: self_map("Auto-check")) or schedule.run_pending(), daemon=True).start()
    while True:
        cmd = input("Enter prompt, 'viz' to visualize, or 'exit': ")
        if cmd.lower() == 'exit': break
        if cmd.lower() == 'viz':
            visualize_dag()
        else:
            self_map(cmd)
    print("=== Completed ===") 