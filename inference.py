from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import json
import numpy as np
import torch
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
from datetime import datetime
import os
import traceback
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch_geometric.loader import DataLoader

# Import your existing modules
try:
    from GraphBuild import build_graph, process_csv, plot_graph_with_features
    from model import MaterialsGNN, evaluate_model, predict_material_properties, load_trained_model
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure GraphBuild.py and model.py are in the same directory")
    exit(1)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables to store models and data
models = {'gcn': None, 'gnn': None}
dataset = None
model_info = None
models_loaded = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    csv_extensions = {'csv'}
    model_extensions = {'pth', 'pt'}
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return ext in csv_extensions or ext in model_extensions

class MaterialsDataset:
    """Simple dataset wrapper to handle CSV data and graph creation"""
    def __init__(self, csv_path=None, target_cols=['energy_above_hull (eV/atom)', 'band_gap (eV)']):
        self.csv_path = csv_path
        self.target_cols = target_cols
        self.df = None
        self.graphs = None
        
        if csv_path and os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            print(f"Loaded dataset with {len(self.df)} samples")
        else:
            print("No CSV file provided or file not found")
    
    def create_graphs(self):
        """Create graph objects from CSV data"""
        if self.df is None:
            return []
        
        if self.graphs is None:
            self.graphs = []
            for idx, row in self.df.iterrows():
                try:
                    graph = build_graph(row)
                    self.graphs.append(graph)
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    continue
        
        return self.graphs
    
    @property
    def feature_cols(self):
        """Get feature column names (excluding targets)"""
        if self.df is None:
            return []
        
        # Return columns that are likely to be material features
        exclude_cols = self.target_cols + ['formula', 'composition', 'sites', 'space_group', 'direct_bandgap']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        return feature_cols

def load_pretrained_models(model_paths, dataset_obj=None):
    """Load pre-trained model files"""
    global models, model_info, models_loaded
    
    if dataset_obj is None:
        flash('Dataset must be loaded before loading models!', 'error')
        return False
    
    try:
        # Get sample graph to determine input dimensions
        sample_graphs = dataset_obj.create_graphs()
        if not sample_graphs:
            flash('No graphs could be created from dataset!', 'error')
            return False
            
        sample_graph = sample_graphs[0]
        node_input_dim = sample_graph.x.shape[1]
        edge_input_dim = sample_graph.edge_attr.shape[1] if sample_graph.edge_attr is not None else 4
        global_input_dim = sample_graph.u.shape[1] if sample_graph.u is not None else 10
        
        model_info = {
            'node_input_dim': node_input_dim,
            'edge_input_dim': edge_input_dim,
            'global_input_dim': global_input_dim,
            'loaded_models': [],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Load models
        for model_name, model_path in model_paths.items():
            if model_path and os.path.exists(model_path):
                try:
                    # Create model with detected architecture
                    model = MaterialsGNN(
                        node_input_dim=node_input_dim,
                        edge_input_dim=edge_input_dim,
                        global_input_dim=global_input_dim,
                        hidden_dim=128,
                        num_conv_layers=4,
                        dropout=0.3
                    )
                    
                    # Load state dict
                    checkpoint = torch.load(model_path, map_location=device)
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model.eval()
                    models[model_name] = model
                    model_info['loaded_models'].append(model_name.upper())
                    print(f"{model_name.upper()} model loaded from {model_path}")
                    
                except Exception as e:
                    flash(f'Failed to load {model_name.upper()} model: {str(e)}', 'error')
                    print(f"{model_name.upper()} loading error: {e}")
        
        if model_info['loaded_models']:
            models_loaded = True
            flash(f'Successfully loaded models: {", ".join(model_info["loaded_models"])}', 'success')
            return True
        else:
            flash('No models were loaded successfully!', 'error')
            return False
            
    except Exception as e:
        flash(f'Error loading models: {str(e)}', 'error')
        print(f"Model loading error: {traceback.format_exc()}")
        return False

@app.route('/')
def index():
    """Main dashboard page"""
    global model_info, models_loaded, dataset
    
    # Create status data
    sample_data = {
        'status': 'Models Loaded' if models_loaded else 'No Models Loaded',
        'models_available': len([k for k, v in models.items() if v is not None]),
        'dataset_loaded': dataset is not None
    }
    
    return render_template('dashboard.html', 
                         sample_data=sample_data,
                         model_info=model_info,
                         models_loaded=models_loaded)

@app.route('/load_models', methods=['GET', 'POST'])
def load_models():
    """Load pre-trained models and dataset"""
    global models, dataset, model_info, models_loaded
    
    if request.method == 'GET':
        # List available model files
        model_files = []
        if os.path.exists(app.config['MODEL_FOLDER']):
            for f in os.listdir(app.config['MODEL_FOLDER']):
                if f.endswith(('.pth', '.pt')):
                    model_files.append(f)
        
        return render_template('load_models.html', model_files=model_files)
    
    try:
        # Get form data
        csv_file = request.files.get('csv_file')
        gnn_file = request.files.get('gnn_file')
        target_column = request.form.get('target_column', 'band_gap (eV)')
        
        # Use existing model files or uploaded ones
        gnn_model_name = request.form.get('existing_gnn_model')
        
        # Handle CSV file
        csv_path = None
        if csv_file and allowed_file(csv_file.filename):
            filename = secure_filename(csv_file.filename)
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            csv_file.save(csv_path)
        
        # Initialize dataset
        if csv_path:
            dataset = MaterialsDataset(csv_path=csv_path)
        else:
            # Create example dataset
            flash('No CSV provided. Please upload a CSV file with materials data.', 'warning')
            return redirect(url_for('load_models'))
        
        # Handle model files
        model_paths = {}
        
        # GNN model
        if gnn_file and allowed_file(gnn_file.filename):
            filename = secure_filename(gnn_file.filename)
            gnn_path = os.path.join(app.config['MODEL_FOLDER'], filename)
            gnn_file.save(gnn_path)
            model_paths['gnn'] = gnn_path
        elif gnn_model_name:
            model_paths['gnn'] = os.path.join(app.config['MODEL_FOLDER'], gnn_model_name)
        
        # Load the models
        if load_pretrained_models(model_paths, dataset):
            return redirect(url_for('index'))
        else:
            return redirect(url_for('load_models'))
        
    except Exception as e:
        flash(f'Failed to load models: {str(e)}', 'error')
        print(f"Loading error: {traceback.format_exc()}")
        return redirect(url_for('load_models'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Make predictions with loaded models"""
    global models, dataset
    
    if not models_loaded or not dataset:
        flash('Please load models and dataset first!', 'warning')
        return redirect(url_for('load_models'))
    
    if request.method == 'GET':
        # Get feature names for the form
        feature_cols = dataset.feature_cols
        return render_template('predict.html', feature_cols=feature_cols)
    
    try:
        # Get input data from form
        composition = request.form.get('composition', 'Ca1 Ti1 O3')
        
        # Create a row-like object for prediction
        # You'll need to get all the required features from the form
        row_data = {
            'composition': composition,
            'a_edge (angstrom)': float(request.form.get('a_edge', 3.85)),
            'b_edge (angstrom)': float(request.form.get('b_edge', 3.85)),
            'c_edge (angstrom)': float(request.form.get('c_edge', 3.85)),
            'alpha_ang (deg)': float(request.form.get('alpha_ang', 90.0)),
            'beta_ang (deg)': float(request.form.get('beta_ang', 90.0)),
            'gamma_ang (deg)': float(request.form.get('gamma_ang', 90.0)),
            'crystal_system': request.form.get('crystal_system', 'cubic'),
            'total_magnetisation (bohr)': float(request.form.get('magnetisation', 0.0)),
            'density (g/cc)': float(request.form.get('density', 5.0)),
            'volume (cubic-angstrom)': float(request.form.get('volume', 60.0)),
            'energy_above_hull (eV/atom)': 0.0,  # dummy values for graph construction
            'band_gap (eV)': 0.0,
            'space_group': 'Pm-3m',
            'sites': 5,
            'formula': composition.replace(' ', ''),
            'direct_bandgap': True,
            'energy_per_atom (eV/atom)': -8.0,
            'formation_energy (eV/atom)': -3.0
        }
        
        # Create graph from input data
        row = pd.Series(row_data)
        graph = build_graph(row)
        
        # Make predictions
        predictions = {}
        if models['gnn'] is not None:
            pred_result = predict_material_properties(models['gnn'], graph)
            predictions['GNN'] = pred_result
        
        # Create visualization
        pred_values = {k: v['energy_above_hull'] for k, v in predictions.items()}
        pred_chart = create_prediction_chart(pred_values)
        
        result = {
            'composition': composition,
            'predictions': predictions,
            'input_data': row_data,
            'chart_json': pred_chart
        }
        
        return render_template('predict_result.html', result=result)
        
    except Exception as e:
        flash(f'Prediction failed: {str(e)}', 'error')
        print(f"Prediction error: {traceback.format_exc()}")
        return redirect(url_for('predict'))

@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    """Evaluate loaded models on test data"""
    global models, dataset
    
    if not models_loaded or not dataset:
        flash('Please load models and dataset first!', 'warning')
        return redirect(url_for('load_models'))
    
    if request.method == 'GET':
        return render_template('evaluate.html')
    
    try:
        # Create test data
        graphs = dataset.create_graphs()
        
        # Handle small datasets
        if len(graphs) < 5:
            flash(f'Dataset too small for evaluation (only {len(graphs)} samples). Using all data for testing.', 'warning')
            test_graphs = graphs
        else:
            # Normal split for larger datasets
            _, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
        
        test_loader = DataLoader(test_graphs, batch_size=min(32, len(test_graphs)), shuffle=False)
        
        # Evaluate models
        evaluation_results = {}
        
        if models['gnn'] is not None:
            try:
                preds, targets = evaluate_model(models['gnn'], test_loader)
                
                # Calculate metrics for both targets
                energy_mae = mean_absolute_error(targets[:, 0], preds[:, 0])
                energy_mse = mean_squared_error(targets[:, 0], preds[:, 0])
                energy_r2 = r2_score(targets[:, 0], preds[:, 0])
                
                bandgap_mae = mean_absolute_error(targets[:, 1], preds[:, 1])
                bandgap_mse = mean_squared_error(targets[:, 1], preds[:, 1])
                bandgap_r2 = r2_score(targets[:, 1], preds[:, 1])
                
                evaluation_results['GNN'] = {
                    'energy_mae': energy_mae,
                    'energy_mse': energy_mse,
                    'energy_r2': energy_r2,
                    'bandgap_mae': bandgap_mae,
                    'bandgap_mse': bandgap_mse,
                    'bandgap_r2': bandgap_r2,
                    'predictions': preds.tolist(),
                    'targets': targets.tolist()
                }
            except Exception as e:
                flash(f'GNN evaluation failed: {str(e)}', 'error')
                print(f"GNN evaluation error: {e}")
        
        # Create charts if we have results
        charts = {}
        if evaluation_results:
            charts = {
                'performance_comparison': create_performance_comparison_chart(evaluation_results),
                'predictions_scatter': create_predictions_scatter_chart(evaluation_results)
            }
            flash(f'Evaluation completed on {len(test_graphs)} test samples', 'success')
        else:
            flash('No evaluation results could be generated', 'warning')
        
        return render_template('evaluation_results.html', 
                             evaluation_results=evaluation_results,
                             charts=charts)
        
    except Exception as e:
        flash(f'Evaluation failed: {str(e)}', 'error')
        print(f"Evaluation error: {traceback.format_exc()}")
        return redirect(url_for('evaluate'))

@app.route('/api/models/status')
def api_model_status():
    """API endpoint for model status"""
    return jsonify({
        'models_loaded': models_loaded,
        'models_available': [k for k, v in models.items() if v is not None],
        'dataset_loaded': dataset is not None,
        'model_info': model_info
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if not models_loaded or not dataset:
        return jsonify({'error': 'Models or dataset not loaded'}), 400
    
    try:
        data = request.get_json()
        composition = data.get('composition', 'Ca1 Ti1 O3')
        material_features = data.get('material_features', {})
        
        # Create prediction graph
        row_data = {
            'composition': composition,
            'a_edge (angstrom)': material_features.get('a_edge', 3.85),
            'b_edge (angstrom)': material_features.get('b_edge', 3.85),
            'c_edge (angstrom)': material_features.get('c_edge', 3.85),
            'alpha_ang (deg)': material_features.get('alpha_ang', 90.0),
            'beta_ang (deg)': material_features.get('beta_ang', 90.0),
            'gamma_ang (deg)': material_features.get('gamma_ang', 90.0),
            'crystal_system': material_features.get('crystal_system', 'cubic'),
            'total_magnetisation (bohr)': material_features.get('magnetisation', 0.0),
            'density (g/cc)': material_features.get('density', 5.0),
            'volume (cubic-angstrom)': material_features.get('volume', 60.0),
            'energy_above_hull (eV/atom)': 0.0,
            'band_gap (eV)': 0.0,
            'space_group': 'Pm-3m',
            'sites': 5,
            'formula': composition.replace(' ', ''),
            'direct_bandgap': True,
            'energy_per_atom (eV/atom)': -8.0,
            'formation_energy (eV/atom)': -3.0
        }
        
        row = pd.Series(row_data)
        graph = build_graph(row)
        
        predictions = {}
        if models['gnn'] is not None:
            pred_result = predict_material_properties(models['gnn'], graph)
            predictions['GNN'] = pred_result
        
        return jsonify({
            'composition': composition,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def create_performance_comparison_chart(evaluation_results):
    """Create performance comparison chart"""
    if not evaluation_results:
        return None
    
    models_list = []
    energy_mae_values = []
    bandgap_mae_values = []
    energy_r2_values = []
    bandgap_r2_values = []
    
    for model_name, results in evaluation_results.items():
        models_list.append(model_name)
        energy_mae_values.append(results['energy_mae'])
        bandgap_mae_values.append(results['bandgap_mae'])
        energy_r2_values.append(results['energy_r2'])
        bandgap_r2_values.append(results['bandgap_r2'])
    
    trace_energy_mae = go.Bar(x=models_list, y=energy_mae_values, name='Energy MAE', yaxis='y')
    trace_bandgap_mae = go.Bar(x=models_list, y=bandgap_mae_values, name='Band Gap MAE', yaxis='y')
    trace_energy_r2 = go.Scatter(x=models_list, y=energy_r2_values, mode='markers+lines', 
                                name='Energy R²', yaxis='y2', marker=dict(size=10, color='green'))
    trace_bandgap_r2 = go.Scatter(x=models_list, y=bandgap_r2_values, mode='markers+lines', 
                                 name='Band Gap R²', yaxis='y2', marker=dict(size=10, color='orange'))
    
    layout = go.Layout(
        title='Model Performance Comparison',
        xaxis=dict(title='Models'),
        yaxis=dict(title='MAE', side='left'),
        yaxis2=dict(title='R² Score', side='right', overlaying='y'),
        hovermode='closest'
    )
    
    fig = go.Figure(data=[trace_energy_mae, trace_bandgap_mae, trace_energy_r2, trace_bandgap_r2], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_predictions_scatter_chart(evaluation_results):
    """Create predictions vs actual scatter plot"""
    if not evaluation_results:
        return None
    
    traces = []
    colors = {'GNN': 'blue'}
    
    for model_name, results in evaluation_results.items():
        targets = np.array(results['targets'])
        predictions = np.array(results['predictions'])
        
        # Energy above hull
        traces.append(go.Scatter(
            x=targets[:, 0],
            y=predictions[:, 0],
            mode='markers',
            name=f'{model_name} Energy Above Hull',
            marker=dict(color=colors.get(model_name, 'black'), opacity=0.6)
        ))
        
        # Band gap
        traces.append(go.Scatter(
            x=targets[:, 1],
            y=predictions[:, 1],
            mode='markers',
            name=f'{model_name} Band Gap',
            marker=dict(color=colors.get(model_name, 'black'), opacity=0.6, symbol='square')
        ))
    
    # Perfect prediction lines
    if evaluation_results:
        all_targets = np.vstack([np.array(results['targets']) for results in evaluation_results.values()])
        
        # Energy line
        min_val, max_val = all_targets[:, 0].min(), all_targets[:, 0].max()
        traces.append(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Energy Prediction',
            line=dict(color='black', dash='dash')
        ))
        
        # Band gap line
        min_val, max_val = all_targets[:, 1].min(), all_targets[:, 1].max()
        traces.append(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Band Gap Prediction',
            line=dict(color='gray', dash='dash')
        ))
    
    layout = go.Layout(
        title='Predictions vs Actual Values',
        xaxis=dict(title='Actual Values'),
        yaxis=dict(title='Predicted Values'),
        hovermode='closest'
    )
    
    fig = go.Figure(data=traces, layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_prediction_chart(predictions):
    """Create chart for individual prediction"""
    models_list = list(predictions.keys())
    values = list(predictions.values())
    
    trace = go.Bar(
        x=models_list,
        y=values,
        marker=dict(color=['blue'][:len(models_list)])
    )
    
    layout = go.Layout(
        title='Model Predictions',
        xaxis=dict(title='Models'),
        yaxis=dict(title='Predicted Energy Above Hull (eV/atom)'),
        hovermode='closest'
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

if __name__ == '__main__':
    print("Starting Materials GNN Dashboard...")
    print("Make sure GraphBuild.py and model.py are in the same directory")
    print("Place your .pth model files in the 'models' folder")
    print("Navigate to http://localhost:5000 in your browser")
    import os
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
