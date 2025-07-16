#!/usr/bin/env python3
"""
Modern Dashboard for Async General Self-Consistency Engine v6.2
Beautiful, responsive UI with real-time monitoring and visualizations
"""

import os
import sys
import json
import time
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import dash
    from dash import dcc, html, Input, Output, callback
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"❌ Missing dashboard dependencies: {e}")
    print("Please install: pip install dash plotly pandas")
    sys.exit(1)

# Configuration
BASE_DIR = "truth"
DB_FILE = os.path.join(BASE_DIR, "brain_map.db")
CONSISTENCY_FILE = os.path.join(BASE_DIR, "consistency_scores.json")
EMOTIONAL_FILE = os.path.join(BASE_DIR, "emotional_patterns.json")
PROMPT_DB_FILE = os.path.join(BASE_DIR, "used_prompts.json")

# Initialize Dash app with modern styling
app = dash.Dash(__name__, title="🧠 Neural Mapping Dashboard v6.2")
app.config.suppress_callback_exceptions = True

# Custom CSS for modern styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>🧠 Neural Mapping Dashboard v6.2</title>
        {%favicon%}
        {%css%}
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            :root {
                --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                --warning-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                --dark-bg: #1a1a2e;
                --card-bg: #16213e;
                --text-light: #e8e8e8;
                --text-muted: #a0a0a0;
                --border-color: #2d3748;
            }
            
            body {
                background: var(--dark-bg);
                color: var(--text-light);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
            }
            
            .navbar {
                background: var(--primary-gradient) !important;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                border: none;
            }
            
            .navbar-brand {
                font-weight: 700;
                font-size: 1.5rem;
                color: white !important;
            }
            
            .card {
                background: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                backdrop-filter: blur(10px);
                transition: all 0.3s ease;
                margin-bottom: 20px;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(0,0,0,0.4);
            }
            
            .card-header {
                background: var(--secondary-gradient);
                border: none;
                border-radius: 15px 15px 0 0 !important;
                color: white;
                font-weight: 600;
                padding: 1rem 1.5rem;
            }
            
            .metric-card {
                background: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 15px;
                padding: 1.5rem;
                text-align: center;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            .metric-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: var(--success-gradient);
            }
            
            .metric-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            }
            
            .metric-value {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                background: var(--success-gradient);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .metric-label {
                color: var(--text-muted);
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .progress-container {
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                padding: 3px;
                margin: 1rem 0;
            }
            
            .progress-bar {
                background: var(--success-gradient);
                border-radius: 8px;
                height: 20px;
                transition: width 0.5s ease;
            }
            
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
                animation: pulse 2s infinite;
            }
            
            .status-active {
                background: #00ff88;
                box-shadow: 0 0 10px #00ff88;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .table {
                background: var(--card-bg);
                border-radius: 10px;
                overflow: hidden;
            }
            
            .table thead th {
                background: var(--secondary-gradient);
                color: white;
                border: none;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-size: 0.8rem;
            }
            
            .table tbody tr {
                transition: all 0.2s ease;
            }
            
            .table tbody tr:hover {
                background: rgba(255,255,255,0.05);
                transform: scale(1.01);
            }
            
            .table tbody td {
                border-color: var(--border-color);
                color: var(--text-light);
                vertical-align: middle;
            }
            
            .category-badge {
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .category-truth { background: linear-gradient(45deg, #ff6b6b, #ee5a24); color: white; }
            .category-persuasion { background: linear-gradient(45deg, #a55eea, #8854d0); color: white; }
            .category-emotional { background: linear-gradient(45deg, #fdcb6e, #e17055); color: white; }
            .category-logical { background: linear-gradient(45deg, #74b9ff, #0984e3); color: white; }
            
            .alert {
                border-radius: 10px;
                border: none;
                font-weight: 500;
            }
            
            .alert-warning {
                background: linear-gradient(45deg, #fdcb6e, #e17055);
                color: white;
            }
            
            .alert-danger {
                background: linear-gradient(45deg, #ff6b6b, #ee5a24);
                color: white;
            }
            
            .loading-spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255,255,255,0.3);
                border-radius: 50%;
                border-top-color: #fff;
                animation: spin 1s ease-in-out infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            .footer {
                background: var(--card-bg);
                border-top: 1px solid var(--border-color);
                padding: 1rem 0;
                text-align: center;
                color: var(--text-muted);
                font-size: 0.9rem;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Global data cache
data_cache = {
    'last_update': None,
    'nodes_data': None,
    'edges_data': None,
    'consistency_data': None,
    'emotional_data': None
}

def get_database_data():
    """Get comprehensive data from database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        
        # Get nodes data
        nodes_df = pd.read_sql_query("""
            SELECT node_id, neuron_idx, concept, confidence, prompt_category,
                   activation_strength, consistency_score, emotional_weight,
                   persuasion_score, truth_score, timestamp
            FROM nodes
            ORDER BY timestamp DESC
        """, conn)
        
        # Get edges data
        edges_df = pd.read_sql_query("""
            SELECT parent_id, child_id, relationship_type, consistency_weight,
                   emotional_weight, persuasion_weight
            FROM edges
        """, conn)
        
        # Get reasoning consistency data
        reasoning_df = pd.read_sql_query("""
            SELECT prompt_hash, reasoning_chain, consistency_score,
                   emotional_tone, persuasion_effectiveness, truth_consistency,
                   timestamp
            FROM reasoning_consistency
            ORDER BY timestamp DESC
        """, conn)
        
        # Get goal tracking data
        goals_df = pd.read_sql_query("""
            SELECT behavior_type, consistency_improvement, emotional_engagement,
                   persuasion_success, truth_maintenance, timestamp
            FROM goal_tracking
            ORDER BY timestamp DESC
        """, conn)
        
        conn.close()
        
        return {
            'nodes': nodes_df,
            'edges': edges_df,
            'reasoning': reasoning_df,
            'goals': goals_df
        }
    except Exception as e:
        print(f"Database error: {e}")
        return {
            'nodes': pd.DataFrame(),
            'edges': pd.DataFrame(),
            'reasoning': pd.DataFrame(),
            'goals': pd.DataFrame()
        }

def load_json_data():
    """Load data from JSON files."""
    data = {}
    
    # Load consistency scores
    if os.path.exists(CONSISTENCY_FILE):
        try:
            with open(CONSISTENCY_FILE, 'r') as f:
                data['consistency'] = json.load(f)
        except:
            data['consistency'] = {}
    
    # Load emotional patterns
    if os.path.exists(EMOTIONAL_FILE):
        try:
            with open(EMOTIONAL_FILE, 'r') as f:
                data['emotional'] = json.load(f)
        except:
            data['emotional'] = {}
    
    # Load used prompts
    if os.path.exists(PROMPT_DB_FILE):
        try:
            with open(PROMPT_DB_FILE, 'r') as f:
                data['prompts'] = json.load(f)
        except:
            data['prompts'] = []
    
    return data

def create_metric_card(title, value, subtitle="", icon="", color_class=""):
    """Create a modern metric card."""
    return html.Div([
        html.Div([
            html.I(className=f"fas {icon} fa-2x mb-3", style={"color": "#667eea"}),
            html.H3(value, className="metric-value"),
            html.H6(title, className="metric-label"),
            html.P(subtitle, className="text-muted mt-2", style={"fontSize": "0.8rem"})
        ], className="text-center")
    ], className="metric-card")

def create_progress_section(coverage_percent, mapped_neurons, target_neurons):
    """Create progress section with modern styling."""
    return html.Div([
        html.H5("📊 Mapping Progress", className="mb-3"),
        html.Div([
            html.Div([
                html.Span(f"{coverage_percent:.1f}%", className="float-end fw-bold"),
                html.Span("Coverage Progress")
            ], className="d-flex justify-content-between mb-2"),
            html.Div([
                html.Div(
                    className="progress-bar",
                    style={"width": f"{min(coverage_percent, 100)}%"}
                )
            ], className="progress-container")
        ]),
        html.Div([
            html.Small(f"{mapped_neurons:,} / {target_neurons:,} neurons mapped", className="text-muted")
        ], className="text-center mt-2")
    ])

def create_category_distribution(nodes_df):
    """Create category distribution with modern styling."""
    if nodes_df.empty:
        return html.Div("No data available", className="alert alert-warning")
    
    category_counts = nodes_df['prompt_category'].value_counts()
    
    cards = []
    for category, count in category_counts.items():
        icon_map = {
            'truth_bias': 'fa-search',
            'persuasion_manipulation': 'fa-comments',
            'emotional_simulation': 'fa-heart',
            'logical_reasoning': 'fa-brain'
        }
        icon = icon_map.get(category, 'fa-circle')
        
        cards.append(html.Div([
            html.I(className=f"fas {icon} fa-lg mb-2"),
            html.H4(f"{count:,}", className="mb-1"),
            html.Small(category.replace('_', ' ').title(), className="text-muted")
        ], className="col-md-3 text-center"))
    
    return html.Div([
        html.H5("🎭 Category Distribution", className="mb-3"),
        html.Div(cards, className="row")
    ])

def create_consistency_chart():
    """Create consistency tracking chart with modern styling."""
    data = get_database_data()
    nodes_df = data['nodes']
    
    if nodes_df.empty:
        return go.Figure().add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="white")
        )
    
    # Convert timestamp to datetime
    nodes_df['timestamp'] = pd.to_datetime(nodes_df['timestamp'])
    nodes_df = nodes_df.sort_values('timestamp')
    
    # Calculate rolling averages
    nodes_df['rolling_consistency'] = nodes_df['consistency_score'].rolling(window=50).mean()
    nodes_df['rolling_confidence'] = nodes_df['confidence'].rolling(window=50).mean()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Consistency Over Time', 'Confidence Over Time', 
                       'Consistency by Category', 'Confidence Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Consistency over time
    fig.add_trace(
        go.Scatter(x=nodes_df['timestamp'], y=nodes_df['rolling_consistency'],
                  mode='lines', name='Rolling Avg', line=dict(color='#667eea', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=nodes_df['timestamp'], y=nodes_df['consistency_score'],
                  mode='markers', name='Individual', marker=dict(size=3, opacity=0.3, color='#764ba2')),
        row=1, col=1
    )
    
    # Confidence over time
    fig.add_trace(
        go.Scatter(x=nodes_df['timestamp'], y=nodes_df['rolling_confidence'],
                  mode='lines', name='Rolling Avg', line=dict(color='#4facfe', width=3)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=nodes_df['timestamp'], y=nodes_df['confidence'],
                  mode='markers', name='Individual', marker=dict(size=3, opacity=0.3, color='#00f2fe')),
        row=1, col=2
    )
    
    # Consistency by category
    for category in nodes_df['prompt_category'].unique():
        cat_data = nodes_df[nodes_df['prompt_category'] == category]
        fig.add_trace(
            go.Box(y=cat_data['consistency_score'], name=category),
            row=2, col=1
        )
    
    # Confidence distribution
    fig.add_trace(
        go.Histogram(x=nodes_df['confidence'], nbinsx=30, name='Confidence',
                    marker_color='#fa709a'),
        row=2, col=2
    )
    
    # Update layout for dark theme
    fig.update_layout(
        height=600,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Update axes for dark theme
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
    
    return fig

def create_emotional_analysis():
    """Create emotional analysis chart with modern styling."""
    data = get_database_data()
    nodes_df = data['nodes']
    
    if nodes_df.empty:
        return go.Figure().add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="white")
        )
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Emotional Weights by Category', 'Persuasion Scores',
                       'Truth Scores', 'Emotional vs Consistency'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Emotional weights by category
    for category in nodes_df['prompt_category'].unique():
        cat_data = nodes_df[nodes_df['prompt_category'] == category]
        fig.add_trace(
            go.Box(y=cat_data['emotional_weight'], name=category),
            row=1, col=1
        )
    
    # Persuasion scores
    fig.add_trace(
        go.Histogram(x=nodes_df['persuasion_score'], nbinsx=30, name='Persuasion',
                    marker_color='#a55eea'),
        row=1, col=2
    )
    
    # Truth scores
    fig.add_trace(
        go.Histogram(x=nodes_df['truth_score'], nbinsx=30, name='Truth',
                    marker_color='#fdcb6e'),
        row=2, col=1
    )
    
    # Emotional vs Consistency scatter
    fig.add_trace(
        go.Scatter(x=nodes_df['emotional_weight'], y=nodes_df['consistency_score'],
                  mode='markers', marker=dict(size=4, opacity=0.6, color='#74b9ff'),
                  text=nodes_df['concept'], name='Neurons'),
        row=2, col=2
    )
    
    # Update layout for dark theme
    fig.update_layout(
        height=600,
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Update axes for dark theme
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
    
    return fig

def create_network_graph():
    """Create network graph visualization with modern styling."""
    data = get_database_data()
    nodes_df = data['nodes']
    edges_df = data['edges']
    
    if nodes_df.empty or edges_df.empty:
        return go.Figure().add_annotation(
            text="No network data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="white")
        )
    
    # Create network graph
    import networkx as nx
    G = nx.DiGraph()
    
    # Add nodes
    for _, row in nodes_df.iterrows():
        G.add_node(row['node_id'], 
                  concept=row['concept'],
                  category=row['prompt_category'],
                  consistency=row['consistency_score'])
    
    # Add edges
    for _, row in edges_df.iterrows():
        G.add_edge(row['parent_id'], row['child_id'],
                  weight=row['consistency_weight'])
    
    # Calculate layout
    if len(G.nodes) < 100:
        pos = nx.spring_layout(G, k=3, iterations=50)
    else:
        pos = nx.kamada_kawai_layout(G)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='rgba(255,255,255,0.3)'),
        hoverinfo='none',
        mode='lines')
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        concept = G.nodes[node]['concept']
        category = G.nodes[node]['category']
        consistency = G.nodes[node]['consistency']
        
        node_text.append(f"Node {node}<br>Concept: {concept}<br>Category: {category}<br>Consistency: {consistency:.3f}")
        
        # Color by category
        if category == 'truth_bias':
            node_colors.append('#ff6b6b')
        elif category == 'persuasion_manipulation':
            node_colors.append('#a55eea')
        elif category == 'emotional_simulation':
            node_colors.append('#fdcb6e')
        elif category == 'logical_reasoning':
            node_colors.append('#74b9ff')
        else:
            node_colors.append('#95a5a6')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=12,
            color=node_colors,
            line_width=2,
            line_color='white'))
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=f'Neural Network Graph - {len(G.nodes)} Nodes, {len(G.edges)} Edges',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)',
                       font=dict(color='white'))
                   )
    
    return fig

# Modern Dashboard Layout
app.layout = html.Div([
    # Navigation Bar
    html.Nav([
        html.Div([
            html.Span("🧠", className="me-2"),
            html.Span("Neural Mapping Dashboard v6.2", className="navbar-brand")
        ], className="container-fluid"),
        html.Div([
            html.Span([
                html.Span("●", className="status-indicator status-active"),
                html.Span("Live", className="text-white")
            ], className="me-3"),
            html.Small(id='last-update', className="text-white-50")
        ], className="d-flex align-items-center")
    ], className="navbar navbar-expand-lg"),
    
    # Main Content
    html.Div([
        # Auto-refresh interval
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # 5 seconds
            n_intervals=0
        ),
        
        # Overview Metrics Row
        html.Div([
            html.Div([
                create_metric_card("Total Neurons", "0", "Mapped", "fa-brain", "primary"),
                create_metric_card("Coverage", "0%", "Target: 95%", "fa-chart-line", "success"),
                create_metric_card("Avg Confidence", "0.0", "Score", "fa-bullseye", "info"),
                create_metric_card("Avg Consistency", "0.0", "Score", "fa-link", "warning")
            ], className="row", id="overview-metrics")
        ], className="mb-4"),
        
        # Progress Section
        html.Div([
            html.Div([
                html.Div(id="progress-section", className="card-body")
            ], className="card")
        ], className="mb-4"),
        
        # Charts Row
        html.Div([
            # Consistency Analysis
            html.Div([
                html.Div([
                    html.H5("📈 Consistency Analysis", className="card-header")
                ], className="card-header"),
                html.Div([
                    dcc.Graph(id='consistency-chart', config={'displayModeBar': False})
                ], className="card-body")
            ], className="card mb-4"),
            
            # Emotional Analysis
            html.Div([
                html.Div([
                    html.H5("😊 Emotional Analysis", className="card-header")
                ], className="card-header"),
                html.Div([
                    dcc.Graph(id='emotional-chart', config={'displayModeBar': False})
                ], className="card-body")
            ], className="card mb-4"),
            
            # Network Graph
            html.Div([
                html.Div([
                    html.H5("🕸️ Neural Network Graph", className="card-header")
                ], className="card-header"),
                html.Div([
                    dcc.Graph(id='network-graph', config={'displayModeBar': False})
                ], className="card-body")
            ], className="card mb-4"),
            
            # Category Distribution
            html.Div([
                html.Div([
                    html.H5("📊 Category Distribution", className="card-header")
                ], className="card-header"),
                html.Div([
                    html.Div(id="category-distribution")
                ], className="card-body")
            ], className="card mb-4"),
            
            # Raw Data Tables
            html.Div([
                html.Div([
                    html.H5("📋 Recent Mappings", className="card-header")
                ], className="card-header"),
                html.Div([
                    html.Div(id='data-tables')
                ], className="card-body")
            ], className="card")
        ])
    ], className="container-fluid p-4"),
    
    # Footer
    html.Footer([
        html.Div([
            html.P([
                html.Span("🧠 General Self-Consistency Engine v6.2", className="me-3"),
                html.Span("•", className="me-3"),
                html.Span("Real-time Neural Mapping", className="me-3"),
                html.Span("•", className="me-3"),
                html.Span("Truth + Persuasion + Emotional Simulation")
            ], className="mb-0")
        ], className="container")
    ], className="footer")
], style={"minHeight": "100vh"})

# Callbacks
@app.callback(
    [Output('overview-metrics', 'children'),
     Output('progress-section', 'children'),
     Output('category-distribution', 'children'),
     Output('consistency-chart', 'figure'),
     Output('emotional-chart', 'figure'),
     Output('network-graph', 'figure'),
     Output('last-update', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    """Update all dashboard components."""
    try:
        data = get_database_data()
        nodes_df = data['nodes']
        
        if nodes_df.empty:
            # Return empty state
            metrics = [
                create_metric_card("Total Neurons", "0", "No Data", "fa-brain"),
                create_metric_card("Coverage", "0%", "No Data", "fa-chart-line"),
                create_metric_card("Avg Confidence", "0.0", "No Data", "fa-bullseye"),
                create_metric_card("Avg Consistency", "0.0", "No Data", "fa-link")
            ]
            progress = html.Div("No data available", className="alert alert-warning")
            category_dist = html.Div("No data available", className="alert alert-warning")
            consistency_fig = create_consistency_chart()
            emotional_fig = create_emotional_analysis()
            network_fig = create_network_graph()
        else:
            # Calculate metrics
            total_neurons = len(nodes_df)
            coverage_percent = (total_neurons / 50000) * 100
            avg_confidence = nodes_df['confidence'].mean()
            avg_consistency = nodes_df['consistency_score'].mean()
            
            # Create metric cards
            metrics = [
                create_metric_card(f"{total_neurons:,}", "Total Neurons", "Mapped", "fa-brain"),
                create_metric_card(f"{coverage_percent:.1f}%", "Coverage", "Target: 95%", "fa-chart-line"),
                create_metric_card(f"{avg_confidence:.3f}", "Avg Confidence", "Score", "fa-bullseye"),
                create_metric_card(f"{avg_consistency:.3f}", "Avg Consistency", "Score", "fa-link")
            ]
            
            # Create progress section
            progress = create_progress_section(coverage_percent, total_neurons, 50000)
            
            # Create category distribution
            category_dist = create_category_distribution(nodes_df)
            
            # Create charts
            consistency_fig = create_consistency_chart()
            emotional_fig = create_emotional_analysis()
            network_fig = create_network_graph()
        
        # Update timestamp
        timestamp = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return metrics, progress, category_dist, consistency_fig, emotional_fig, network_fig, timestamp
        
    except Exception as e:
        print(f"Dashboard update error: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error state
        error_metrics = [
            create_metric_card("Error", "❌", "System Error", "fa-exclamation-triangle"),
            create_metric_card("Error", "❌", "System Error", "fa-exclamation-triangle"),
            create_metric_card("Error", "❌", "System Error", "fa-exclamation-triangle"),
            create_metric_card("Error", "❌", "System Error", "fa-exclamation-triangle")
        ]
        error_progress = html.Div(f"Error: {str(e)}", className="alert alert-danger")
        error_category = html.Div(f"Error: {str(e)}", className="alert alert-danger")
        error_fig = go.Figure().add_annotation(text=f"Error: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        return error_metrics, error_progress, error_category, error_fig, error_fig, error_fig, f"Error: {str(e)}"

@app.callback(
    Output('data-tables', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_data_tables(n):
    """Update data tables."""
    try:
        data = get_database_data()
        nodes_df = data['nodes']
        
        if nodes_df.empty:
            return html.Div("No data available", className="alert alert-warning")
        
        # Check for required columns
        required_columns = ['neuron_idx', 'concept', 'prompt_category', 'confidence', 'consistency_score', 'timestamp']
        missing_columns = [col for col in required_columns if col not in nodes_df.columns]
        
        if missing_columns:
            return html.Div(f"Missing columns in database: {missing_columns}", className="alert alert-danger")
        
        # Handle null values
        nodes_df = nodes_df.dropna(subset=['neuron_idx', 'concept', 'prompt_category'])
        nodes_df = nodes_df.fillna({
            'confidence': 0.5,
            'consistency_score': 0.5
        })
        
        if nodes_df.empty:
            return html.Div("No valid data after cleaning", className="alert alert-warning")
        
        # Recent mappings table
        recent_nodes = nodes_df.head(20)
        
        def get_category_class(category):
            if 'truth' in category:
                return 'category-truth'
            elif 'persuasion' in category:
                return 'category-persuasion'
            elif 'emotional' in category:
                return 'category-emotional'
            elif 'logical' in category:
                return 'category-logical'
            return 'category-truth'
        
        return html.Div([
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Neuron", className="text-center"),
                        html.Th("Concept"),
                        html.Th("Category"),
                        html.Th("Confidence", className="text-center"),
                        html.Th("Consistency", className="text-center"),
                        html.Th("Timestamp", className="text-center")
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(str(row['neuron_idx']), className="text-center fw-bold"),
                        html.Td(str(row['concept'])[:30] + "..." if len(str(row['concept'])) > 30 else str(row['concept'])),
                        html.Td(html.Span(str(row['prompt_category']).replace('_', ' ').title(), className=f"category-badge {get_category_class(str(row['prompt_category']))}")),
                        html.Td(f"{float(row['confidence']):.3f}", className="text-center"),
                        html.Td(f"{float(row['consistency_score']):.3f}", className="text-center"),
                        html.Td(str(row['timestamp'])[:19] if isinstance(row['timestamp'], str) else str(row['timestamp'])[:19], className="text-center text-muted")
                    ]) for _, row in recent_nodes.iterrows()
                ])
            ], className="table table-hover")
        ])
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Dashboard error: {error_details}")
        return html.Div([
            html.H6("Error Loading Data", className="text-danger"),
            html.P(f"Error: {str(e)}"),
            html.P("Check the console for full error details.")
        ], className="alert alert-danger")

if __name__ == '__main__':
    print("🚀 Starting Modern Neural Mapping Dashboard v6.2...")
    print("📊 Dashboard will be available at: http://127.0.0.1:8050")
    print("🔄 Auto-refresh every 5 seconds")
    print("🎨 Modern dark theme with animations")
    print("=" * 60)
    
    app.run(debug=True, host='127.0.0.1', port=8050) 