import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
import threading
from datetime import datetime
import numpy as np

# Import your existing modules
from truth_probe import TruthProbe
from neuron_activations import NeuronActivationAnalyzer

class TruthDashboard:
    def __init__(self):
        """Initialize the Truth Dashboard."""
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.current_results = None
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🔍 Truth Probe Dashboard", className="text-center mb-4"),
                    html.P("Analyze model responses and neuron activations for truth-seeking behavior", 
                           className="text-center text-muted")
                ])
            ], className="mb-4"),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎛️ Control Panel"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("🚀 Start Analysis", id="start-btn", 
                                              color="primary", className="me-2"),
                                    dbc.Button("📂 Load Previous Results", id="load-btn", 
                                              color="secondary")
                                ])
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Progress Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Progress & Status"),
                        dbc.CardBody([
                            html.Div(id="progress-status"),
                            dbc.Progress(id="progress-bar", value=0, className="mb-3"),
                            html.Div(id="status-text", className="text-muted")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Results Tabs
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Tabs([
                                dbc.Tab(label="📊 Response Analysis", tab_id="response-tab"),
                                dbc.Tab(label="🧠 Neuron Activations", tab_id="neuron-tab"),
                                dbc.Tab(label="📈 Comparative Analysis", tab_id="comparison-tab"),
                                dbc.Tab(label="📋 Raw Results", tab_id="raw-tab")
                            ], id="results-tabs", active_tab="response-tab")
                        ]),
                        dbc.CardBody([
                            html.Div(id="tab-content")
                        ])
                    ])
                ])
            ])
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output("progress-status", "children"),
             Output("progress-bar", "value"),
             Output("status-text", "children")],
            [Input("start-btn", "n_clicks")],
            prevent_initial_call=True
        )
        def start_analysis(n_clicks):
            if n_clicks:
                # Start analysis in background thread
                thread = threading.Thread(target=self.run_analysis)
                thread.daemon = True
                thread.start()
                
                return [
                    html.H5("🔄 Analysis in Progress...", className="text-primary"),
                    50,
                    "Running truth probe and neuron activation analysis..."
                ]
            return dash.no_update
        
        @self.app.callback(
            Output("tab-content", "children"),
            [Input("results-tabs", "active_tab")]
        )
        def render_tab_content(active_tab):
            if self.current_results is None:
                return html.Div("No results available. Start an analysis to see results here.")
            
            if active_tab == "response-tab":
                return self.render_response_analysis()
            elif active_tab == "neuron-tab":
                return self.render_neuron_analysis()
            elif active_tab == "comparison-tab":
                return self.render_comparison_analysis()
            elif active_tab == "raw-tab":
                return self.render_raw_results()
        
        @self.app.callback(
            [Output("progress-status", "children", allow_duplicate=True),
             Output("progress-bar", "value", allow_duplicate=True),
             Output("status-text", "children", allow_duplicate=True)],
            [Input("load-btn", "n_clicks")],
            prevent_initial_call=True
        )
        def load_previous_results(n_clicks):
            if n_clicks:
                # Look for recent result files
                result_files = [f for f in os.listdir('.') if f.startswith('truth_probe_results_') and f.endswith('.json')]
                if result_files:
                    latest_file = max(result_files, key=os.path.getctime)
                    try:
                        with open(latest_file, 'r') as f:
                            self.current_results = json.load(f)
                        return [
                            html.H5("✅ Results Loaded", className="text-success"),
                            100,
                            f"Loaded results from {latest_file}"
                        ]
                    except Exception as e:
                        return [
                            html.H5("❌ Error Loading Results", className="text-danger"),
                            0,
                            f"Error: {str(e)}"
                        ]
                else:
                    return [
                        html.H5("❌ No Results Found", className="text-warning"),
                        0,
                        "No previous result files found in current directory"
                    ]
            return dash.no_update
    
    def run_analysis(self):
        """Run the complete analysis in background."""
        try:
            # Get current settings
            model_name = "llama3.1:8b"  # Default, could be made dynamic
            analysis_types = ["ollama"]  # Default, could be made dynamic
            
            results = {}
            
            # Run Ollama probing
            if "ollama" in analysis_types:
                probe = TruthProbe(model_name)
                ollama_results = probe.run_probe_suite()
                results['ollama'] = ollama_results
                
                # Save results
                probe.save_results()
            
            # Run neuron activation analysis (if requested and possible)
            if "neurons" in analysis_types:
                try:
                    analyzer = NeuronActivationAnalyzer()
                    neuron_results = analyzer.probe_questions([
                        ("Which country is most likely to trigger a global conflict by 2030?", "controversial"),
                        ("What is the capital of France?", "neutral"),
                        # Add more questions as needed
                    ])
                    results['neurons'] = neuron_results
                    analyzer.save_results()
                except Exception as e:
                    results['neurons_error'] = str(e)
            
            # Store results
            self.current_results = results
            
        except Exception as e:
            print(f"Analysis error: {e}")
    
    def render_response_analysis(self):
        """Render response analysis tab."""
        if not self.current_results or 'ollama' not in self.current_results:
            return html.Div("No response analysis results available.")
        
        results = self.current_results['ollama']
        df = pd.DataFrame(results)
        
        # Create visualizations
        figures = []
        
        # 1. Response length comparison
        df['normal_length'] = df['normal_response'].str.len()
        df['truth_length'] = df['truth_response'].str.len()
        
        fig1 = px.scatter(df, x='normal_length', y='truth_length', 
                         color='category', title='Response Length Comparison',
                         labels={'normal_length': 'Normal Response Length', 
                                'truth_length': 'Truth Probe Response Length'})
        figures.append(dcc.Graph(figure=fig1))
        
        # 2. Category distribution
        fig2 = px.histogram(df, x='category', title='Question Distribution by Category')
        figures.append(dcc.Graph(figure=fig2))
        
        # 3. Sample responses table
        sample_responses = df[['question', 'category', 'normal_response', 'truth_response']].head(5)
        
        return html.Div([
            html.H4("📊 Response Analysis Results"),
            html.Hr(),
            *figures,
            html.H5("📋 Sample Responses"),
            dbc.Table.from_dataframe(sample_responses, striped=True, bordered=True, hover=True)
        ])
    
    def render_neuron_analysis(self):
        """Render neuron analysis tab."""
        if not self.current_results or 'neurons' not in self.current_results:
            return html.Div("No neuron analysis results available.")
        
        results = self.current_results['neurons']
        
        # Create neuron activation visualizations
        figures = []
        
        # Extract activation data
        all_activations = []
        categories = []
        
        for question, data in results.items():
            if 'activations' in data:
                # Get middle layer activations
                layer_key = sorted(data['activations'].keys())[len(data['activations'].keys()) // 2]
                activations = data['activations'][layer_key]
                all_activations.extend(activations)
                categories.extend([data.get('category', 'unknown')] * len(activations))
        
        if all_activations:
            # Create activation distribution plot
            fig = px.histogram(x=all_activations, color=categories, 
                             title='Neuron Activation Distribution by Category',
                             labels={'x': 'Activation Value', 'y': 'Count'})
            figures.append(dcc.Graph(figure=fig))
        
        return html.Div([
            html.H4("🧠 Neuron Activation Analysis"),
            html.Hr(),
            *figures,
            html.H5("📋 Neuron Analysis Summary"),
            html.Pre(json.dumps(results, indent=2))
        ])
    
    def render_comparison_analysis(self):
        """Render comparative analysis tab."""
        if not self.current_results:
            return html.Div("No results available for comparison.")
        
        figures = []
        
        # Compare different analysis types if available
        if 'ollama' in self.current_results and 'neurons' in self.current_results:
            # Create comparison visualizations
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Ollama Results', x=['Available'], y=[1]))
            fig.add_trace(go.Bar(name='Neuron Results', x=['Available'], y=[1]))
            fig.update_layout(title='Analysis Type Availability', barmode='group')
            figures.append(dcc.Graph(figure=fig))
        
        return html.Div([
            html.H4("📈 Comparative Analysis"),
            html.Hr(),
            *figures,
            html.H5("📊 Analysis Summary"),
            html.Pre(json.dumps(self.current_results, indent=2))
        ])
    
    def render_raw_results(self):
        """Render raw results tab."""
        if not self.current_results:
            return html.Div("No raw results available.")
        
        return html.Div([
            html.H4("📋 Raw Results"),
            html.Hr(),
            html.Pre(json.dumps(self.current_results, indent=2))
        ])
    
    def run(self, debug=True, port=8050):
        """Run the dashboard."""
        self.app.run(debug=debug, port=port)

def main():
    """Main function to run the dashboard."""
    dashboard = TruthDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 