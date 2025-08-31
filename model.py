import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

class MaterialsGNN(nn.Module):
    def __init__(self, 
                 node_input_dim=3,      # Z, mass, electronegativity
                 edge_input_dim=4,      # edge features from your data
                 global_input_dim=10,   # 3 material props + 7 crystal system encoding
                 hidden_dim=64,
                 num_conv_layers=3,
                 dropout=0.2):
        super(MaterialsGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout
        
        # --- Node feature preprocessing ---
        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        
        # --- Edge feature preprocessing ---
        self.edge_embedding = nn.Linear(edge_input_dim, hidden_dim)
        
        # --- Graph convolution layers with edge features ---
        self.conv_layers = nn.ModuleList()
        for i in range(num_conv_layers):
            # Use GATConv which supports edge features
            self.conv_layers.append(
                GATConv(hidden_dim, hidden_dim, 
                       edge_dim=hidden_dim,  # Use edge features
                       heads=4, 
                       concat=False,
                       dropout=0.1)
            )
        
        # --- Global feature processing ---
        self.global_mlp = nn.Sequential(
            nn.Linear(global_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # --- Final prediction layers ---
        # Combine mean pooling + global features
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # pooled nodes + global
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # 2 targets: energy_above_hull, band_gap
        )
        
    def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch
        
        # --- Process node features ---
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # --- Process edge features ---
        if edge_attr is not None and edge_attr.size(0) > 0:
            edge_attr = self.edge_embedding(edge_attr)
            edge_attr = F.relu(edge_attr)
        else:
            # Create zero edge features if none exist
            edge_attr = torch.zeros(edge_index.size(1), self.hidden_dim, 
                                  device=x.device, dtype=x.dtype)
        
        # --- Graph convolutions with edge features ---
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # --- Simple mean pooling ---
        x_pooled = global_mean_pool(x, batch)
        
        # --- Process global features ---
        # Debug print shapes
        if hasattr(self, '_debug_once') == False:
            print(f"Debug - u shape before processing: {u.shape}")
            print(f"Debug - edge_attr shape: {edge_attr.shape}")
            print(f"Debug - x_pooled shape: {x_pooled.shape}")
            self._debug_once = True
            
        # Handle global features properly
        if u.dim() == 3:  # [batch_size, 1, features]
            u = u.squeeze(1)
        elif u.dim() == 2 and u.size(0) == 1:  # [1, features] 
            u = u.squeeze(0).unsqueeze(0)
        
        u = self.global_mlp(u)
        
        # --- Combine representations ---
        combined = torch.cat([x_pooled, u], dim=1)
        
        # --- Final prediction ---
        out = self.final_mlp(combined)
        
        return out


def train_model(model, train_loader, val_loader, num_epochs=200, lr=0.001):
    """
    Training loop for the materials GNN
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        num_train_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            try:
                out = model(batch)
                
                # Fix target tensor shape issue
                targets = batch.y
                if targets.dim() == 1:
                    # Reshape flattened targets back to [batch_size, 2]
                    batch_size = out.size(0)
                    targets = targets.view(batch_size, 2)
                
                # Debug shapes if needed
                if epoch == 0 and num_train_batches == 0:
                    print(f"Model output shape: {out.shape}")
                    print(f"Original target shape: {batch.y.shape}")
                    print(f"Reshaped target shape: {targets.shape}")
                    print(f"Batch size from model output: {out.size(0)}")
                
                loss = criterion(out, targets)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_train_batches += 1
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                print(f"Output shape: {out.shape if 'out' in locals() else 'N/A'}")
                print(f"Target shape: {batch.y.shape}")
                print(f"Batch info - num_graphs: {batch.num_graphs if hasattr(batch, 'num_graphs') else 'N/A'}")
                continue
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                try:
                    out = model(batch)
                    targets = batch.y
                    if targets.dim() == 1:
                        batch_size = out.size(0)
                        targets = targets.view(batch_size, 2)
                    
                    loss = criterion(out, targets)
                    val_loss += loss.item()
                    num_val_batches += 1
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        if num_train_batches > 0 and num_val_batches > 0:
            train_losses.append(train_loss / num_train_batches)
            val_losses.append(val_loss / num_val_batches)
        
        if epoch % 50 == 0 and len(train_losses) > 0:
            print(f'Epoch {epoch:03d}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    
    return train_losses, val_losses


def evaluate_model(model, test_loader):
    """
    Evaluate model performance on test set
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            
            # Fix target tensor shape issue
            targets = batch.y
            if targets.dim() == 1:
                batch_size = out.size(0)
                targets = targets.view(batch_size, 2)
            
            all_preds.append(out.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # Calculate metrics for both targets
    energy_mae = mean_absolute_error(targets[:, 0], preds[:, 0])
    energy_mse = mean_squared_error(targets[:, 0], preds[:, 0])
    
    bandgap_mae = mean_absolute_error(targets[:, 1], preds[:, 1])
    bandgap_mse = mean_squared_error(targets[:, 1], preds[:, 1])
    
    print(f"\nTest Results:")
    print(f"Energy Above Hull - MAE: {energy_mae:.4f}, MSE: {energy_mse:.4f}")
    print(f"Band Gap - MAE: {bandgap_mae:.4f}, MSE: {bandgap_mse:.4f}")
    
    return preds, targets


def plot_training_curves(train_losses, val_losses, filename="training_curves.png"):
    """
    Plot training and validation loss curves
    """
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses[-50:], label='Train Loss (Last 50)', color='blue')
    plt.plot(val_losses[-50:], label='Val Loss (Last 50)', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curves (Final Epochs)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {filename}")


def plot_predictions(preds, targets, filename="predictions.png"):
    """
    Plot predicted vs actual values for both targets
    """
    plt.figure(figsize=(12, 5))
    
    # Energy Above Hull
    plt.subplot(1, 2, 1)
    plt.scatter(targets[:, 0], preds[:, 0], alpha=0.6, color='blue')
    plt.plot([targets[:, 0].min(), targets[:, 0].max()], 
             [targets[:, 0].min(), targets[:, 0].max()], 'r--', lw=2)
    plt.xlabel('Actual Energy Above Hull (eV/atom)')
    plt.ylabel('Predicted Energy Above Hull (eV/atom)')
    plt.title('Energy Above Hull Predictions')
    plt.grid(True, alpha=0.3)
    
    # Band Gap
    plt.subplot(1, 2, 2)
    plt.scatter(targets[:, 1], preds[:, 1], alpha=0.6, color='green')
    plt.plot([targets[:, 1].min(), targets[:, 1].max()], 
             [targets[:, 1].min(), targets[:, 1].max()], 'r--', lw=2)
    plt.xlabel('Actual Band Gap (eV)')
    plt.ylabel('Predicted Band Gap (eV)')
    plt.title('Band Gap Predictions')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Prediction plots saved to {filename}")


# --- Example Usage ---
if __name__ == "__main__":
    # Import the graph building functions
    from GraphBuild import process_csv, build_graph
    
    try:
        # Load graphs from CSV
        csv_file = "Perovskite_data_processed.csv"
        graphs = process_csv(csv_file)
        
        if len(graphs) < 10:
            print("Warning: Very small dataset. Consider using more data for better training.")
        
        # Split data
        train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
        train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.2, random_state=42)
        
        # Create data loaders with proper collate function
        def collate_fn(batch):
            """Custom collate function to handle batching properly"""
            from torch_geometric.data import Batch
            return Batch.from_data_list(batch)
        
        batch_size = min(32, len(train_graphs) // 2)  # Adjust batch size for small datasets
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        print(f"Dataset sizes - Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
        
        # Initialize model
        model = MaterialsGNN(
            node_input_dim=3,      # Z, mass, electronegativity
            edge_input_dim=4,      # Your edge features
            global_input_dim=10,   # 3 material props + 7 crystal systems
            hidden_dim=128,        # Increased for better capacity
            num_conv_layers=4,     # Deeper network
            dropout=0.3
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        print("\nStarting training...")
        train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                             num_epochs=300, lr=0.001)
        
        # Plot training curves
        plot_training_curves(train_losses, val_losses)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        preds, targets = evaluate_model(model, test_loader)
        
        # Plot predictions
        plot_predictions(preds, targets)
        
        # Save trained model
        torch.save(model.state_dict(), 'materials_gnn_model.pth')
        print("Model saved to 'materials_gnn_model.pth'")
        
    except FileNotFoundError:
        print("CSV file not found. Creating example with synthetic data...")
        
        # Create synthetic data for demonstration
        import pandas as pd
        from GraphBuild import build_graph
        
        # Create more diverse synthetic examples
        synthetic_data = []
        elements = [["Ti", "O"], ["Fe", "O"], ["Ca", "Ti", "O"], ["Ba", "Ti", "O"], ["Sr", "Ti", "O"]]
        compositions = ["Ti1 O2", "Fe2 O3", "Ca1 Ti1 O3", "Ba1 Ti1 O3", "Sr1 Ti1 O3"]
        
        for i, (comp_str, elem_list) in enumerate(zip(compositions, elements)):
            row_data = {
                "formula": comp_str.replace(" ", ""),
                "sites": 5 + i,
                "composition": comp_str,
                "a_edge (angstrom)": 3.8 + np.random.normal(0, 0.2),
                "b_edge (angstrom)": 3.8 + np.random.normal(0, 0.2), 
                "c_edge (angstrom)": 3.8 + np.random.normal(0, 0.2),
                "alpha_ang (deg)": 90.0 + np.random.normal(0, 5),
                "beta_ang (deg)": 90.0 + np.random.normal(0, 5),
                "gamma_ang (deg)": 90.0 + np.random.normal(0, 5),
                "crystal_system": np.random.choice(['cubic', 'tetragonal', 'orthorhombic']),
                "space_group": "Pm-3m",
                "total_magnetisation (bohr)": np.random.uniform(-2, 2),
                "energy_per_atom (eV/atom)": -8.0 + np.random.normal(0, 1),
                "formation_energy (eV/atom)": -3.0 + np.random.normal(0, 0.5),
                "energy_above_hull (eV/atom)": abs(np.random.normal(0, 0.1)),
                "density (g/cc)": 5.0 + np.random.normal(0, 2),
                "band_gap (eV)": abs(np.random.normal(2, 1)),
                "direct_bandgap": np.random.choice([True, False]),
                "volume (cubic-angstrom)": 50 + np.random.normal(0, 10)
            }
            synthetic_data.append(row_data)
        
        # Create graphs from synthetic data
        graphs = []
        for row_data in synthetic_data:
            try:
                # Convert to pandas Series to mimic CSV row
                row = pd.Series(row_data)
                graph = build_graph(row)
                graphs.append(graph)
            except Exception as e:
                print(f"Error creating synthetic graph: {e}")
        
        if graphs:
            print(f"Created {len(graphs)} synthetic graphs for demonstration")
            
            # Simple train/test split for demo
            train_graphs = graphs[:4]
            test_graphs = graphs[4:]
            
            train_loader = DataLoader(train_graphs, batch_size=2, shuffle=True)
            test_loader = DataLoader(test_graphs, batch_size=2, shuffle=False)
            
            # Debug: Check a single batch first
            print("\nDebugging batch shapes...")
            sample_batch = next(iter(train_loader))
            print(f"Sample batch.x shape: {sample_batch.x.shape}")
            print(f"Sample batch.edge_attr shape: {sample_batch.edge_attr.shape}")
            print(f"Sample batch.u shape: {sample_batch.u.shape}")
            print(f"Sample batch.y shape: {sample_batch.y.shape}")
            print(f"Sample batch.batch shape: {sample_batch.batch.shape}")
            
            # Initialize model
            model = MaterialsGNN(hidden_dim=32, num_conv_layers=2)  # Smaller for demo
            
            # Test forward pass
            print("\nTesting forward pass...")
            model.eval()
            with torch.no_grad():
                out = model(sample_batch)
                print(f"Model output shape: {out.shape}")
                print(f"Expected target shape: {sample_batch.y.shape}")
            
            # Quick training demo
            print("\nDemo training (few epochs)...")
            train_losses, val_losses = train_model(model, train_loader, test_loader, 
                                                 num_epochs=20, lr=0.01)
            
            # Demo evaluation
            preds, targets = evaluate_model(model, test_loader)
            print("Demo completed successfully!")


# --- Utility function to load a trained model ---
def load_trained_model(model_path, model_class=MaterialsGNN, **model_kwargs):
    """
    Load a trained model from saved state dict
    """
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


# --- Prediction function for new materials ---
def predict_material_properties(model, graph):
    """
    Predict properties for a single material graph
    """
    model.eval()
    with torch.no_grad():
        # Create a batch with single graph
        loader = DataLoader([graph], batch_size=1, shuffle=False)
        batch = next(iter(loader))
        
        prediction = model(batch)
        energy_above_hull, band_gap = prediction.squeeze().tolist()
        
        return {
            'energy_above_hull': energy_above_hull,
            'band_gap': band_gap
        }
