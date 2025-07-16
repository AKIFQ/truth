#!/usr/bin/env python3
"""
Dashboard v5: Enhanced Real-Time Self-Mapping Monitor
Shows live coverage, DAG preview, and mapping statistics.
"""

import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import sqlite3
import json
import os
import sys
import time
import threading
from datetime import datetime, timedelta
import networkx as nx
from typing import Dict, List

# Add parent directory to path to find introspect module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Configuration ---
DB_FILE = "truth/brain_map.db"
CHECKPOINT_FILE = "truth/checkpoint.json"
VIS_DIR = "truth/visuals"

# --- Database Functions ---

def get_coverage_stats():
    """Get current coverage statistics."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        # Get latest coverage
        c.execute("""
            SELECT total_mapped, coverage_percent, timestamp 
            FROM coverage_stats 
            ORDER BY timestamp DESC LIMIT 1
        """)
        row = c.fetchone()
        
        if row:
            return {
                "total_mapped": row[0],
                "coverage_percent": row[1],
                "timestamp": row[2]
            }
        else:
            return {"total_mapped": 0, "coverage_percent": 0, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"total_mapped": 0, "coverage_percent": 0, "timestamp": datetime.now().isoformat(), "error": str(e)}

def get_mapping_data():
    """Get recent mapping data for visualization."""
    try:
        conn = sqlite3.connect(DB_FILE)
        
        # Get recent mappings
        df = pd.read_sql_query("""
            SELECT neuron_idx, concept, confidence, prompt_category, 
                   activation_strength, timestamp, prompt_used
            FROM nodes 
            ORDER BY timestamp DESC 
            LIMIT 100
        """, conn)
        
        # Get category statistics
        category_stats = pd.read_sql_query("""
            SELECT prompt_category, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM nodes 
            GROUP BY prompt_category
        """, conn)
        
        # Get coverage history
        coverage_history = pd.read_sql_query("""
            SELECT total_mapped, coverage_percent, timestamp
            FROM coverage_stats 
            ORDER BY timestamp DESC 
            LIMIT 50
        """, conn)
        
        conn.close()
        
        return {
            "recent_mappings": df,
            "category_stats": category_stats,
            "coverage_history": coverage_history
        }
    except Exception as e:
        return {
            "recent_mappings": pd.DataFrame(),
            "category_stats": pd.DataFrame(),
            "coverage_history": pd.DataFrame(),
            "error": str(e)
        }

def get_top_concepts():
    """Get top discovered concepts."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        
        c.execute("""
            SELECT concept, COUNT(*) as frequency, AVG(confidence) as avg_confidence
            FROM nodes 
            GROUP BY concept 
            ORDER BY frequency DESC, avg_confidence DESC 
            LIMIT 20
        """)
        
        results = []
        for row in c.fetchall():
            results.append({
                "concept": row[0],
                "frequency": row[1],
                "avg_confidence": row[2]
            })
        
        conn.close()
        return results
    except Exception as e:
        return []

# --- Dashboard Layout ---

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🧠 Self-Mapping v5 Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Real-time Status Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Live Coverage Status"),
                dbc.CardBody([
                    html.Div(id="coverage-display"),
                    dbc.Progress(id="coverage-progress", className="mb-3"),
                    html.Div(id="coverage-stats")
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎯 Mapping Progress"),
                dbc.CardBody([
                    html.Div(id="mapping-stats"),
                    html.Div(id="eta-display")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Control Row
    dbc.Row([
        dbc.Col([
            dbc.Button("🔄 Refresh Data", id="refresh-btn", color="primary", className="me-2"),
            dbc.Button("📈 Generate Report", id="report-btn", color="success", className="me-2"),
            html.Div(id="last-updated", className="text-muted mt-2")
        ])
    ], className="mb-4"),
    
    # Main Content Tabs
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    dbc.Tabs([
                        dbc.Tab(label="📈 Coverage History", tab_id="coverage-tab"),
                        dbc.Tab(label="🧠 Recent Mappings", tab_id="mappings-tab"),
                        dbc.Tab(label="🏷️ Top Concepts", tab_id="concepts-tab"),
                        dbc.Tab(label="📊 Category Analysis", tab_id="categories-tab"),
                        dbc.Tab(label="🕸️ DAG Preview", tab_id="dag-tab")
                    ], id="main-tabs", active_tab="coverage-tab")
                ]),
                dbc.CardBody([
                    html.Div(id="tab-content")
                ])
            ])
        ])
    ])
], fluid=True)

# --- Callbacks ---

@app.callback(
    [Output("coverage-display", "children"),
     Output("coverage-progress", "value"),
     Output("coverage-stats", "children"),
     Output("mapping-stats", "children"),
     Output("eta-display", "children"),
     Output("last-updated", "children")],
    [Input("refresh-btn", "n_clicks"),
     Input("interval-component", "n_intervals")],
    prevent_initial_call=True
)
def update_status(n_clicks, n_intervals):
    """Update real-time status displays."""
    stats = get_coverage_stats()
    
    # Coverage display
    coverage_display = html.H3(f"{stats['coverage_percent']:.1f}% Coverage", className="text-primary")
    
    # Progress bar
    progress_value = stats['coverage_percent']
    
    # Coverage stats
    coverage_stats = html.Div([
        html.P(f"📊 Mapped Neurons: {stats['total_mapped']}"),
        html.P(f"🎯 Target: 95% (50,000 neurons)"),
        html.P(f"⏰ Last Update: {stats['timestamp'][:19]}")
    ])
    
    # Mapping stats
    mapping_stats = html.Div([
        html.P(f"🔍 Total Mappings: {stats['total_mapped']:,}"),
        html.P(f"📈 Remaining: {50000 - stats['total_mapped']:,} neurons"),
        html.P(f"🎯 Completion: {(stats['coverage_percent']/95)*100:.1f}%")
    ])
    
    # ETA calculation
    if stats['total_mapped'] > 0:
        # Rough estimate: 5 seconds per mapping
        remaining_time = (50000 - stats['total_mapped']) * 5
        eta_hours = remaining_time // 3600
        eta_minutes = (remaining_time % 3600) // 60
        eta_display = html.P(f"⏱️ Estimated Time: {eta_hours}h {eta_minutes}m remaining")
    else:
        eta_display = html.P("⏱️ Calculating ETA...")
    
    # Last updated
    last_updated = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
    
    return coverage_display, progress_value, coverage_stats, mapping_stats, eta_display, last_updated

@app.callback(
    Output("tab-content", "children"),
    [Input("main-tabs", "active_tab")]
)
def render_tab_content(active_tab):
    """Render content for each tab."""
    
    if active_tab == "coverage-tab":
        return render_coverage_history()
    elif active_tab == "mappings-tab":
        return render_recent_mappings()
    elif active_tab == "concepts-tab":
        return render_top_concepts()
    elif active_tab == "categories-tab":
        return render_category_analysis()
    elif active_tab == "dag-tab":
        return render_dag_preview()
    else:
        return html.Div("Select a tab to view content.")

def render_coverage_history():
    """Render coverage history chart."""
    data = get_mapping_data()
    df = data["coverage_history"]
    
    if df.empty:
        return html.Div("No coverage data available yet.")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['coverage_percent'],
        mode='lines+markers',
        name='Coverage %',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Coverage Progress Over Time",
        xaxis_title="Time",
        yaxis_title="Coverage %",
        height=400
    )
    
    return dcc.Graph(figure=fig)

def render_recent_mappings():
    """Render recent mappings table."""
    data = get_mapping_data()
    df = data["recent_mappings"]
    
    if df.empty:
        return html.Div("No mapping data available yet.")
    
    # Format the dataframe for display
    display_df = df[['neuron_idx', 'concept', 'confidence', 'prompt_category', 'timestamp']].head(20)
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%H:%M:%S')
    display_df['confidence'] = display_df['confidence'].round(2)
    
    return dbc.Table.from_dataframe(
        display_df, 
        striped=True, 
        bordered=True, 
        hover=True,
        className="table-sm"
    )

def render_top_concepts():
    """Render top discovered concepts."""
    concepts = get_top_concepts()
    
    if not concepts:
        return html.Div("No concept data available yet.")
    
    # Create bar chart
    df = pd.DataFrame(concepts)
    
    fig = px.bar(
        df.head(15), 
        x='concept', 
        y='frequency',
        color='avg_confidence',
        title="Top Discovered Concepts",
        labels={'concept': 'Concept', 'frequency': 'Frequency', 'avg_confidence': 'Avg Confidence'}
    )
    
    fig.update_layout(height=500, xaxis_tickangle=-45)
    
    return dcc.Graph(figure=fig)

def render_category_analysis():
    """Render category analysis."""
    data = get_mapping_data()
    df = data["category_stats"]
    
    if df.empty:
        return html.Div("No category data available yet.")
    
    # Create pie chart
    fig = px.pie(
        df, 
        values='count', 
        names='prompt_category',
        title="Mappings by Category"
    )
    
    fig.update_layout(height=400)
    
    # Add confidence bar chart
    fig2 = px.bar(
        df, 
        x='prompt_category', 
        y='avg_confidence',
        title="Average Confidence by Category"
    )
    
    fig2.update_layout(height=300)
    
    return html.Div([
        dcc.Graph(figure=fig),
        dcc.Graph(figure=fig2)
    ])

def render_dag_preview():
    """Render DAG preview."""
    # Look for latest DAG visualization
    if os.path.exists(VIS_DIR):
        dag_files = [f for f in os.listdir(VIS_DIR) if f.startswith('dag_') and f.endswith('.png')]
        if dag_files:
            latest_dag = max(dag_files, key=lambda x: os.path.getctime(os.path.join(VIS_DIR, x)))
            dag_path = os.path.join(VIS_DIR, latest_dag)
            
            return html.Div([
                html.H5(f"Latest DAG Visualization: {latest_dag}"),
                html.Img(src=f"/assets/{latest_dag}", style={"maxWidth": "100%", "height": "auto"})
            ])
    
    return html.Div([
        html.H5("DAG Preview"),
        html.P("No DAG visualization available yet. The system generates visualizations every 100 mappings.")
    ])

# Add interval component for auto-refresh
app.layout.children.append(
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # 30 seconds
        n_intervals=0
    )
)

# --- Main Execution ---

def main():
    """Run the dashboard."""
    print("🚀 Starting Self-Mapping v5 Dashboard")
    print("📊 Dashboard will be available at: http://127.0.0.1:8050")
    print("🔄 Auto-refresh every 30 seconds")
    
    app.run(debug=True, host='127.0.0.1', port=8050)

if __name__ == "__main__":
    main() 