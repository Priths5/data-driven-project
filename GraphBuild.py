import torch
from torch_geometric.data import Data
from pymatgen.core.periodic_table import Element
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def build_graph(row):
    # --- Parse composition ---
    composition_str = row['composition']
    comp = {}
    expanded_atoms = []
    
    for token in composition_str.split():
        el, num = ''.join([c for c in token if c.isalpha()]), ''.join([c for c in token if c.isdigit()])
        num = int(num) if num else 1
        comp[el] = num
        expanded_atoms.extend([el] * num)
    
    total_atoms = len(expanded_atoms)
    
    # --- Fractions ---
    fractions = {el: count / total_atoms for el, count in comp.items()}
    
    # --- Center atom = element with highest fraction ---
    center_atom = max(fractions, key=fractions.get)
    center_idx = expanded_atoms.index(center_atom)  # Find first occurrence in expanded list
    
    # --- Node features (without fraction) ---
    nodes = []
    for el in expanded_atoms:
        elem = Element(el)
        node_feat = [
            elem.Z,
            float(elem.atomic_mass),
            elem.X if elem.X else 0.0
            # Removed fraction feature
        ]
        nodes.append(node_feat)
    x = torch.tensor(nodes, dtype=torch.float)
    
    # --- Edge construction ---
    edge_index = []
    edge_attr = []
    
    for i, el in enumerate(expanded_atoms):
        if i == center_idx: 
            continue
        
        # Define edge feature rule - works for any element
        if el == "Ac":
            edge_feat = [row['a_edge (angstrom)'], row['alpha_ang (deg)'], 0, 0]
        elif el == "Al":
            edge_feat = [row['b_edge (angstrom)'], row['beta_ang (deg)'], 0, 0]
        elif el == "O":
            edge_feat = [row['c_edge (angstrom)'], row['gamma_ang (deg)'], 0, 0]
        else:
            # Default edge features for other elements
            edge_feat = [row['a_edge (angstrom)'], row['alpha_ang (deg)'], 0, 0]
        
        # Add bidirectional edges
        edge_index.append([center_idx, i])
        edge_index.append([i, center_idx])
        edge_attr.append(edge_feat)
        edge_attr.append(edge_feat)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # --- Global features ---
    crystal_systems = ['cubic','tetragonal','orthorhombic','monoclinic','triclinic','hexagonal','trigonal']
    crystal_sys_enc = [1.0 if row['crystal_system'] == cs else 0.0 for cs in crystal_systems]
    global_feat = [
        row['density (g/cc)'],
        row['total_magnetisation (bohr)'],
        row['volume (cubic-angstrom)']
    ]
    global_feat = torch.tensor(global_feat + crystal_sys_enc, dtype=torch.float)
    
    # --- Targets ---
    y = torch.tensor([
        row['energy_above_hull (eV/atom)'],
        row['band_gap (eV)']
    ], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=global_feat.unsqueeze(0), y=y, atom_list=expanded_atoms, center_idx=center_idx)


def process_csv(csv_file_path):
    """
    Process entire CSV file and return list of graph objects
    """
    df = pd.read_csv(csv_file_path)
    graphs = []
    
    for idx, row in df.iterrows():
        try:
            graph = build_graph(row)
            graphs.append(graph)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    return graphs


def plot_graph_with_features(graph, filename="graph_detailed.png"):
    """
    Visualize a PyG graph with detailed node & edge attributes and global features.
    Assumes `graph.atom_list` and `graph.center_idx` are included in the Data object.
    """
    # Convert to networkx
    G = to_networkx(graph, node_attrs=["x"], edge_attrs=["edge_attr"])
    
    # --- Node labels (element + features without fraction) ---
    labels = {}
    for i, el in enumerate(graph.atom_list):
        z, mass, eneg = graph.x[i].tolist()  # Only 3 features now
        labels[i] = f"{el}\nZ={int(z)}, M={mass:.1f}, χ={eneg:.2f}"
    
    # Layout (force center atom near center)
    pos = nx.spring_layout(G, seed=42)
    
    # Create figure with subplots for graph and global features
    fig = plt.figure(figsize=(14, 8))
    
    # --- Main graph plot ---
    ax1 = plt.subplot(121)  # Left subplot for graph
    nx.draw(G, pos, with_labels=False, node_size=2800,
            node_color="skyblue", edgecolors="black", ax=ax1)
    
    # Add node labels
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", ax=ax1)
    
    # --- Edge labels (custom features) ---
    edge_labels = {}
    for (u, v, d) in G.edges(data=True):
        edge_feat = d["edge_attr"]
        if hasattr(edge_feat, "tolist"):  # tensor -> list
            edge_feat = edge_feat.tolist()
        # Edge features: [param, angle, 0, 0]
        param, angle, _, _ = edge_feat
        edge_labels[(u, v)] = f"param={param:.2f}, angle={angle:.1f}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, ax=ax1)
    
    # Highlight the chosen center atom
    nx.draw_networkx_nodes(G, pos, nodelist=[graph.center_idx], 
                           node_color="orange", node_size=3000, edgecolors="black", ax=ax1)
    
    ax1.set_title("Material Graph Structure", fontsize=14, fontweight="bold")
    
    # --- Global features display ---
    ax2 = plt.subplot(122)  # Right subplot for global features
    ax2.axis('off')  # Remove axes for text display
    
    # Extract global features
    global_features = graph.u.squeeze().tolist()
    
    # Parse global features
    density = global_features[0]
    magnetization = global_features[1] 
    volume = global_features[2]
    
    # Crystal system one-hot encoding (last 7 values)
    crystal_systems = ['cubic','tetragonal','orthorhombic','monoclinic','triclinic','hexagonal','trigonal']
    crystal_encoding = global_features[3:]
    crystal_system = crystal_systems[crystal_encoding.index(1.0)] if 1.0 in crystal_encoding else "unknown"
    
    # Target values
    targets = graph.y.tolist()
    energy_above_hull = targets[0]
    band_gap = targets[1]
    
    # Create text display
    global_text = f"""GLOBAL FEATURES
    
Material Properties:
• Density: {density:.3f} g/cc
• Volume: {volume:.2f} Å³
• Magnetization: {magnetization:.3f} μB
• Crystal System: {crystal_system}

Target Properties:
• Energy Above Hull: {energy_above_hull:.4f} eV/atom
• Band Gap: {band_gap:.4f} eV

Graph Statistics:
• Total Atoms: {len(graph.atom_list)}
• Center Atom: {graph.atom_list[graph.center_idx]} (index {graph.center_idx})
• Number of Edges: {graph.edge_index.shape[1]}
• Atom Composition: {', '.join(graph.atom_list)}"""
    
    ax2.text(0.05, 0.95, global_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    ax2.set_title("Material Properties", fontsize=14, fontweight="bold")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graph with detailed features and global properties saved to {filename}")


# --- Example Usage with CSV file ---
if __name__ == "__main__":
    # Read from CSV file
    csv_file = "materials_data.csv"  # Replace with your CSV file path
    
    try:
        # Process all rows in CSV
        graphs = process_csv(csv_file)
        print(f"Successfully processed {len(graphs)} materials from CSV")
        
        # Example: visualize the first graph
        if graphs:
            plot_graph_with_features(graphs[0], "first_material_graph.png")
            
            # Print some info about the first graph
            first_graph = graphs[0]
            print(f"\nFirst material info:")
            print(f"Number of atoms: {len(first_graph.atom_list)}")
            print(f"Atom list: {first_graph.atom_list}")
            print(f"Center atom index: {first_graph.center_idx}")
            print(f"Center atom element: {first_graph.atom_list[first_graph.center_idx]}")
            print(f"Graph: {first_graph}")
            
    except FileNotFoundError:
        print(f"CSV file '{csv_file}' not found. Creating example with sample data...")
        
        # Fallback to example data
        df = pd.DataFrame([{
            "formula": "AcAlO3",
            "sites": 5,
            "composition": "Ac1 Al1 O3",
            "a_edge (angstrom)": 3.85863387,
            "b_edge (angstrom)": 2.85863387,
            "c_edge (angstrom)": 1.85863387,
            "alpha_ang (deg)": 90.0,
            "beta_ang (deg)": 90.0,
            "gamma_ang (deg)": 90.0,
            "crystal_system": "cubic",
            "space_group": "Pm-3m",
            "total_magnetisation (bohr)": 0.0,
            "energy_per_atom (eV/atom)": -8.232146192,
            "formation_energy (eV/atom)": -3.690019421,
            "energy_above_hull (eV/atom)": 0.0,
            "density (g/cc)": 8.728230082,
            "band_gap (eV)": 4.1024,
            "direct_bandgap": True,
            "volume (cubic-angstrom)": 57.45141324
        }])
        
        graph = build_graph(df.iloc[0])
        print(f"Example graph: {graph}")
        plot_graph_with_features(graph, "example_material_graph.png")
