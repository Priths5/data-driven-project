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
import gunicorn
# Import your existing script
try:
    from DDMMpeakClaude import (
        MaterialsGraphDataset, GCN, GraphSAGE, 
        evaluate_model, predict_properties,
        device
    )
except ImportError:
    print("Make sure DDMMpeakClaude.py is in the same directory")
    exit(1)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables to store models and data
models = {'gcn': None, 'sage': None}
dataset = None
model_info = None
models_loaded = False

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    csv_extensions = {'csv'}
    model_extensions = {'pth', 'pt'}
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return ext in csv_extensions or ext in model_extensions

def load_pretrained_models(gcn_path=None, sage_path=None, dataset_obj=None):
    """Load pre-trained model files with intelligent architecture detection"""
    global models, model_info, models_loaded
    
    if dataset_obj is None:
        flash('Dataset must be loaded before loading models!', 'error')
        return False
    
    try:
        # Get input dimensions from dataset
        sample_graphs = dataset_obj.create_graphs()
        if not sample_graphs:
            flash('No graphs could be created from dataset!', 'error')
            return False
            
        input_dim = sample_graphs[0].x.shape[1]
        
        model_info = {
            'input_dim': input_dim,
            'loaded_models': [],
            'model_architectures': {},
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Load GCN model
        if gcn_path and os.path.exists(gcn_path):
            try:
                # Try to infer architecture from saved model
                checkpoint = torch.load(gcn_path, map_location=device)
                gcn_model, gcn_arch = load_model_with_architecture_detection(
                    checkpoint, 'GCN', input_dim
                )
                if gcn_model:
                    models['gcn'] = gcn_model
                    model_info['loaded_models'].append('GCN')
                    model_info['model_architectures']['GCN'] = gcn_arch
                    print(f"GCN model loaded from {gcn_path} with architecture: {gcn_arch}")
                else:
                    flash(f'Failed to load GCN model from {gcn_path}', 'error')
            except Exception as e:
                flash(f'Failed to load GCN model: {str(e)}', 'error')
                print(f"GCN loading error: {e}")
        
        # Load GraphSAGE model
        if sage_path and os.path.exists(sage_path):
            try:
                # Try to infer architecture from saved model
                checkpoint = torch.load(sage_path, map_location=device)
                sage_model, sage_arch = load_model_with_architecture_detection(
                    checkpoint, 'GraphSAGE', input_dim
                )
                if sage_model:
                    models['sage'] = sage_model
                    model_info['loaded_models'].append('GraphSAGE')
                    model_info['model_architectures']['GraphSAGE'] = sage_arch
                    print(f"GraphSAGE model loaded from {sage_path} with architecture: {sage_arch}")
                else:
                    flash(f'Failed to load GraphSAGE model from {sage_path}', 'error')
            except Exception as e:
                flash(f'Failed to load GraphSAGE model: {str(e)}', 'error')
                print(f"GraphSAGE loading error: {e}")
        
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

def load_model_with_architecture_detection(checkpoint, model_type, input_dim):
    """Load model with automatic architecture detection"""
    try:
        # Try to infer architecture from the checkpoint
        state_dict = checkpoint if isinstance(checkpoint, dict) and 'convs.0.lin_l.weight' in checkpoint else checkpoint.get('model_state_dict', checkpoint)
        
        # Detect input dimension from first layer
        if 'convs.0.lin_l.weight' in state_dict:
            detected_input_dim = state_dict['convs.0.lin_l.weight'].shape[1]
        elif 'convs.0.weight' in state_dict:
            detected_input_dim = state_dict['convs.0.weight'].shape[1]
        else:
            detected_input_dim = input_dim
        
        # Detect hidden dimension
        if 'convs.0.lin_l.weight' in state_dict:
            hidden_dim = state_dict['convs.0.lin_l.weight'].shape[0]
        elif 'convs.0.weight' in state_dict:
            hidden_dim = state_dict['convs.0.weight'].shape[0]
        else:
            hidden_dim = 128  # default
        
        # Detect number of layers by counting conv layers
        conv_layers = [k for k in state_dict.keys() if k.startswith('convs.') and ('weight' in k)]
        num_layers = len(set([k.split('.')[1] for k in conv_layers])) if conv_layers else 3
        
        # Detect output dimension from classifier
        output_dim = 1  # default for regression
        if 'classifier.3.weight' in state_dict:
            output_dim = state_dict['classifier.3.weight'].shape[0]
        elif 'classifier.2.weight' in state_dict:
            output_dim = state_dict['classifier.2.weight'].shape[0]
        elif 'head.2.weight' in state_dict:  # Different naming convention
            output_dim = state_dict['head.2.weight'].shape[0]
        
        # Handle different naming conventions
        if 'head.0.weight' in state_dict and 'classifier.0.weight' not in state_dict:
            # Convert head.* to classifier.* naming
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('head.'):
                    new_key = k.replace('head.', 'classifier.')
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        
        dropout = 0.2  # default
        
        arch_info = {
            'input_dim': detected_input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'num_layers': num_layers,
            'dropout': dropout
        }
        
        # Create model with detected architecture
        if model_type == 'GCN':
            model = GCN(detected_input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)
        else:  # GraphSAGE
            model = GraphSAGE(detected_input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)
        
        # Try to load the state dict
        try:
            model.load_state_dict(state_dict, strict=False)  # Use strict=False to handle minor mismatches
            model.eval()
            return model, arch_info
        except Exception as e:
            print(f"Error loading state dict: {e}")
            return None, None
            
    except Exception as e:
        print(f"Architecture detection failed: {e}")
        return None, None

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
        gcn_file = request.files.get('gcn_file')
        sage_file = request.files.get('sage_file')
        target_column = request.form.get('target_column', 'band_gap')
        
        # Use existing model files or uploaded ones
        gcn_model_name = request.form.get('existing_gcn_model')
        sage_model_name = request.form.get('existing_sage_model')
        
        # Handle CSV file
        csv_path = None
        if csv_file and allowed_file(csv_file.filename):
            filename = secure_filename(csv_file.filename)
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            csv_file.save(csv_path)
        
        # Initialize dataset
        if csv_path:
            dataset = MaterialsGraphDataset(csv_path=csv_path, target_col=target_column)
        else:
            # Use default dataset
            dataset = MaterialsGraphDataset(target_col=target_column)
            flash('Using default sample dataset', 'info')
        
        # Handle model files
        gcn_path = None
        sage_path = None
        
        # GCN model
        if gcn_file and allowed_file(gcn_file.filename):
            filename = secure_filename(gcn_file.filename)
            gcn_path = os.path.join(app.config['MODEL_FOLDER'], filename)
            gcn_file.save(gcn_path)
        elif gcn_model_name:
            gcn_path = os.path.join(app.config['MODEL_FOLDER'], gcn_model_name)
        
        # GraphSAGE model
        if sage_file and allowed_file(sage_file.filename):
            filename = secure_filename(sage_file.filename)
            sage_path = os.path.join(app.config['MODEL_FOLDER'], filename)
            sage_file.save(sage_path)
        elif sage_model_name:
            sage_path = os.path.join(app.config['MODEL_FOLDER'], sage_model_name)
        
        # Load the models
        if load_pretrained_models(gcn_path, sage_path, dataset):
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
        formula = request.form.get('formula', '')
        
        # Get material features
        material_features = []
        for col in dataset.feature_cols:
            value = request.form.get(col, 0)
            try:
                material_features.append(float(value))
            except ValueError:
                material_features.append(0.0)
        
        # Make predictions
        predictions = {}
        if models['gcn'] is not None:
            gcn_pred = predict_properties(models['gcn'], formula, material_features, dataset)
            predictions['GCN'] = float(gcn_pred[0])
        
        if models['sage'] is not None:
            sage_pred = predict_properties(models['sage'], formula, material_features, dataset)
            predictions['GraphSAGE'] = float(sage_pred[0])
        
        # Create visualization
        pred_chart = create_prediction_chart(predictions)
        
        result = {
            'formula': formula,
            'predictions': predictions,
            'material_features': dict(zip(dataset.feature_cols, material_features)),
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
            from sklearn.model_selection import train_test_split
            _, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
        
        from torch_geometric.loader import DataLoader
        test_loader = DataLoader(test_graphs, batch_size=min(32, len(test_graphs)), shuffle=False)
        
        # Evaluate models
        evaluation_results = {}
        
        if models['gcn'] is not None:
            try:
                gcn_results = evaluate_model(models['gcn'], test_loader)
                evaluation_results['GCN'] = gcn_results
            except Exception as e:
                flash(f'GCN evaluation failed: {str(e)}', 'error')
                print(f"GCN evaluation error: {e}")
        
        if models['sage'] is not None:
            try:
                sage_results = evaluate_model(models['sage'], test_loader)
                evaluation_results['GraphSAGE'] = sage_results
            except Exception as e:
                flash(f'GraphSAGE evaluation failed: {str(e)}', 'error')
                print(f"GraphSAGE evaluation error: {e}")
        
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

# Add this route for debugging model architectures
@app.route('/debug_model/<path:model_path>')
def debug_model(model_path):
    """Debug model architecture"""
    try:
        full_path = os.path.join(app.config['MODEL_FOLDER'], model_path)
        if os.path.exists(full_path):
            checkpoint = torch.load(full_path, map_location='cpu')
            
            # Get state dict
            state_dict = checkpoint if isinstance(checkpoint, dict) and any(k.startswith('convs.') for k in checkpoint.keys()) else checkpoint.get('model_state_dict', checkpoint)
            
            info = {
                'file': model_path,
                'keys': list(state_dict.keys()),
                'layer_info': {}
            }
            
            # Analyze layers
            for key in state_dict.keys():
                if 'weight' in key:
                    shape = state_dict[key].shape
                    info['layer_info'][key] = {'shape': list(shape)}
            
            return jsonify(info)
        else:
            return jsonify({'error': 'Model file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        formula = data.get('formula', '')
        material_features = data.get('material_features', [])
        
        predictions = {}
        if models['gcn'] is not None:
            gcn_pred = predict_properties(models['gcn'], formula, material_features, dataset)
            predictions['GCN'] = float(gcn_pred[0])
        
        if models['sage'] is not None:
            sage_pred = predict_properties(models['sage'], formula, material_features, dataset)
            predictions['GraphSAGE'] = float(sage_pred[0])
        
        return jsonify({
            'formula': formula,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def create_performance_comparison_chart(evaluation_results):
    """Create performance comparison chart"""
    if not evaluation_results:
        return None
    
    models_list = []
    mse_values = []
    mae_values = []
    r2_values = []
    
    for model_name, results in evaluation_results.items():
        models_list.append(model_name)
        mse_values.append(results['mse'])
        mae_values.append(results['mae'])
        r2_values.append(results['r2'])
    
    trace_mse = go.Bar(x=models_list, y=mse_values, name='MSE', yaxis='y')
    trace_mae = go.Bar(x=models_list, y=mae_values, name='MAE', yaxis='y')
    trace_r2 = go.Scatter(x=models_list, y=r2_values, mode='markers+lines', 
                         name='R²', yaxis='y2', marker=dict(size=10, color='green'))
    
    layout = go.Layout(
        title='Model Performance Comparison',
        xaxis=dict(title='Models'),
        yaxis=dict(title='MSE / MAE', side='left'),
        yaxis2=dict(title='R² Score', side='right', overlaying='y'),
        hovermode='closest'
    )
    
    fig = go.Figure(data=[trace_mse, trace_mae, trace_r2], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_predictions_scatter_chart(evaluation_results):
    """Create predictions vs actual scatter plot"""
    if not evaluation_results:
        return None
    
    traces = []
    colors = {'GCN': 'blue', 'GraphSAGE': 'red'}
    
    for model_name, results in evaluation_results.items():
        traces.append(go.Scatter(
            x=results['targets'],
            y=results['predictions'],
            mode='markers',
            name=f'{model_name} Predictions',
            marker=dict(color=colors.get(model_name, 'black'), opacity=0.6)
        ))
    
    # Perfect prediction line
    if evaluation_results:
        all_targets = []
        for results in evaluation_results.values():
            all_targets.extend(results['targets'])
        
        if all_targets:
            min_val, max_val = min(all_targets), max(all_targets)
            traces.append(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='black', dash='dash')
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
        marker=dict(color=['blue', 'red'][:len(models_list)])
    )
    
    layout = go.Layout(
        title='Model Predictions',
        xaxis=dict(title='Models'),
        yaxis=dict(title='Predicted Value'),
        hovermode='closest'
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

if __name__ == '__main__':
    print("Starting Materials GNN Dashboard...")
    print("Make sure DDMMpeakClaude.py is in the same directory")
    print("Place your .pth model files in the 'models' folder")
    print("Navigate to http://localhost:5000 in your browser")
    import os
    port=int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
