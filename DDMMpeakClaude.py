import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_max_pool
import re
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
import os
warnings.filterwarnings('ignore')

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ChemicalFormulaParser:
    """Parse chemical formulas and extract elemental compositions"""
    
    def __init__(self):
        # Basic atomic properties (you can expand this with more features)
        self.atomic_properties = {
            'H': {'atomic_number': 1, 'atomic_mass': 1.008, 'electronegativity': 2.20, 'atomic_radius': 0.37, 'period': 1, 'group': 1},
            'He': {'atomic_number': 2, 'atomic_mass': 4.003, 'electronegativity': 0.0, 'atomic_radius': 0.32, 'period': 1, 'group': 18},
            'Li': {'atomic_number': 3, 'atomic_mass': 6.941, 'electronegativity': 0.98, 'atomic_radius': 1.34, 'period': 2, 'group': 1},
            'Be': {'atomic_number': 4, 'atomic_mass': 9.012, 'electronegativity': 1.57, 'atomic_radius': 0.90, 'period': 2, 'group': 2},
            'B': {'atomic_number': 5, 'atomic_mass': 10.811, 'electronegativity': 2.04, 'atomic_radius': 0.82, 'period': 2, 'group': 13},
            'C': {'atomic_number': 6, 'atomic_mass': 12.011, 'electronegativity': 2.55, 'atomic_radius': 0.77, 'period': 2, 'group': 14},
            'N': {'atomic_number': 7, 'atomic_mass': 14.007, 'electronegativity': 3.04, 'atomic_radius': 0.75, 'period': 2, 'group': 15},
            'O': {'atomic_number': 8, 'atomic_mass': 15.999, 'electronegativity': 3.44, 'atomic_radius': 0.73, 'period': 2, 'group': 16},
            'F': {'atomic_number': 9, 'atomic_mass': 18.998, 'electronegativity': 3.98, 'atomic_radius': 0.71, 'period': 2, 'group': 17},
            'Ne': {'atomic_number': 10, 'atomic_mass': 20.180, 'electronegativity': 0.0, 'atomic_radius': 0.69, 'period': 2, 'group': 18},
            'Na': {'atomic_number': 11, 'atomic_mass': 22.990, 'electronegativity': 0.93, 'atomic_radius': 1.54, 'period': 3, 'group': 1},
            'Mg': {'atomic_number': 12, 'atomic_mass': 24.305, 'electronegativity': 1.31, 'atomic_radius': 1.30, 'period': 3, 'group': 2},
            'Al': {'atomic_number': 13, 'atomic_mass': 26.982, 'electronegativity': 1.61, 'atomic_radius': 1.18, 'period': 3, 'group': 13},
            'Si': {'atomic_number': 14, 'atomic_mass': 28.086, 'electronegativity': 1.90, 'atomic_radius': 1.11, 'period': 3, 'group': 14},
            'P': {'atomic_number': 15, 'atomic_mass': 30.974, 'electronegativity': 2.19, 'atomic_radius': 1.06, 'period': 3, 'group': 15},
            'S': {'atomic_number': 16, 'atomic_mass': 32.065, 'electronegativity': 2.58, 'atomic_radius': 1.02, 'period': 3, 'group': 16},
            'Cl': {'atomic_number': 17, 'atomic_mass': 35.453, 'electronegativity': 3.16, 'atomic_radius': 0.99, 'period': 3, 'group': 17},
            'Ar': {'atomic_number': 18, 'atomic_mass': 39.948, 'electronegativity': 0.0, 'atomic_radius': 0.97, 'period': 3, 'group': 18},
            'K': {'atomic_number': 19, 'atomic_mass': 39.098, 'electronegativity': 0.82, 'atomic_radius': 1.96, 'period': 4, 'group': 1},
            'Ca': {'atomic_number': 20, 'atomic_mass': 40.078, 'electronegativity': 1.00, 'atomic_radius': 1.74, 'period': 4, 'group': 2},
            'Sc': {'atomic_number': 21, 'atomic_mass': 44.956, 'electronegativity': 1.36, 'atomic_radius': 1.44, 'period': 4, 'group': 3},
            'Ti': {'atomic_number': 22, 'atomic_mass': 47.867, 'electronegativity': 1.54, 'atomic_radius': 1.36, 'period': 4, 'group': 4},
            'V': {'atomic_number': 23, 'atomic_mass': 50.942, 'electronegativity': 1.63, 'atomic_radius': 1.25, 'period': 4, 'group': 5},
            'Cr': {'atomic_number': 24, 'atomic_mass': 51.996, 'electronegativity': 1.66, 'atomic_radius': 1.27, 'period': 4, 'group': 6},
            'Mn': {'atomic_number': 25, 'atomic_mass': 54.938, 'electronegativity': 1.55, 'atomic_radius': 1.39, 'period': 4, 'group': 7},
            'Fe': {'atomic_number': 26, 'atomic_mass': 55.845, 'electronegativity': 1.83, 'atomic_radius': 1.25, 'period': 4, 'group': 8},
            'Co': {'atomic_number': 27, 'atomic_mass': 58.933, 'electronegativity': 1.88, 'atomic_radius': 1.26, 'period': 4, 'group': 9},
            'Ni': {'atomic_number': 28, 'atomic_mass': 58.693, 'electronegativity': 1.91, 'atomic_radius': 1.21, 'period': 4, 'group': 10},
            'Cu': {'atomic_number': 29, 'atomic_mass': 63.546, 'electronegativity': 1.90, 'atomic_radius': 1.38, 'period': 4, 'group': 11},
            'Zn': {'atomic_number': 30, 'atomic_mass': 65.38, 'electronegativity': 1.65, 'atomic_radius': 1.31, 'period': 4, 'group': 12},
            'Ga': {'atomic_number': 31, 'atomic_mass': 69.723, 'electronegativity': 1.81, 'atomic_radius': 1.26, 'period': 4, 'group': 13},
            'Ge': {'atomic_number': 32, 'atomic_mass': 72.64, 'electronegativity': 2.01, 'atomic_radius': 1.22, 'period': 4, 'group': 14},
            'As': {'atomic_number': 33, 'atomic_mass': 74.922, 'electronegativity': 2.18, 'atomic_radius': 1.19, 'period': 4, 'group': 15},
            'Se': {'atomic_number': 34, 'atomic_mass': 78.96, 'electronegativity': 2.55, 'atomic_radius': 1.16, 'period': 4, 'group': 16},
            'Br': {'atomic_number': 35, 'atomic_mass': 79.904, 'electronegativity': 2.96, 'atomic_radius': 1.14, 'period': 4, 'group': 17},
            'Kr': {'atomic_number': 36, 'atomic_mass': 83.798, 'electronegativity': 3.00, 'atomic_radius': 1.10, 'period': 4, 'group': 18},
            'Y': {'atomic_number': 39, 'atomic_mass': 88.906, 'electronegativity': 1.22, 'atomic_radius': 1.62, 'period': 5, 'group': 3},
            'Zr': {'atomic_number': 40, 'atomic_mass': 91.224, 'electronegativity': 1.33, 'atomic_radius': 1.48, 'period': 5, 'group': 4},
            'Nb': {'atomic_number': 41, 'atomic_mass': 92.906, 'electronegativity': 1.6, 'atomic_radius': 1.37, 'period': 5, 'group': 5},
            'Mo': {'atomic_number': 42, 'atomic_mass': 95.96, 'electronegativity': 2.16, 'atomic_radius': 1.45, 'period': 5, 'group': 6},
            'Tc': {'atomic_number': 43, 'atomic_mass': 98.0, 'electronegativity': 1.9, 'atomic_radius': 1.56, 'period': 5, 'group': 7},
            'Ru': {'atomic_number': 44, 'atomic_mass': 101.07, 'electronegativity': 2.2, 'atomic_radius': 1.26, 'period': 5, 'group': 8},
            'Rh': {'atomic_number': 45, 'atomic_mass': 102.906, 'electronegativity': 2.28, 'atomic_radius': 1.35, 'period': 5, 'group': 9},
            'Pd': {'atomic_number': 46, 'atomic_mass': 106.42, 'electronegativity': 2.20, 'atomic_radius': 1.31, 'period': 5, 'group': 10},
            'Ag': {'atomic_number': 47, 'atomic_mass': 107.868, 'electronegativity': 1.93, 'atomic_radius': 1.53, 'period': 5, 'group': 11},
            'Cd': {'atomic_number': 48, 'atomic_mass': 112.411, 'electronegativity': 1.69, 'atomic_radius': 1.48, 'period': 5, 'group': 12},
            'In': {'atomic_number': 49, 'atomic_mass': 114.818, 'electronegativity': 1.78, 'atomic_radius': 1.44, 'period': 5, 'group': 13},
            'Sn': {'atomic_number': 50, 'atomic_mass': 118.71, 'electronegativity': 1.96, 'atomic_radius': 1.41, 'period': 5, 'group': 14},
            'Sb': {'atomic_number': 51, 'atomic_mass': 121.76, 'electronegativity': 2.05, 'atomic_radius': 1.38, 'period': 5, 'group': 15},
            'Te': {'atomic_number': 52, 'atomic_mass': 127.6, 'electronegativity': 2.1, 'atomic_radius': 1.35, 'period': 5, 'group': 16},
            'I': {'atomic_number': 53, 'atomic_mass': 126.904, 'electronegativity': 2.66, 'atomic_radius': 1.33, 'period': 5, 'group': 17},
            'Xe': {'atomic_number': 54, 'atomic_mass': 131.293, 'electronegativity': 2.60, 'atomic_radius': 1.30, 'period': 5, 'group': 18},
            'La': {'atomic_number': 57, 'atomic_mass': 138.905, 'electronegativity': 1.10, 'atomic_radius': 1.69, 'period': 6, 'group': 3},
            'Ce': {'atomic_number': 58, 'atomic_mass': 140.116, 'electronegativity': 1.12, 'atomic_radius': 1.65, 'period': 6, 'group': 101},
            'Pr': {'atomic_number': 59, 'atomic_mass': 140.908, 'electronegativity': 1.13, 'atomic_radius': 1.65, 'period': 6, 'group': 101},
            'Nd': {'atomic_number': 60, 'atomic_mass': 144.242, 'electronegativity': 1.14, 'atomic_radius': 1.64, 'period': 6, 'group': 101},
            'Pm': {'atomic_number': 61, 'atomic_mass': 145.0, 'electronegativity': 1.13, 'atomic_radius': 1.63, 'period': 6, 'group': 101},
            'Sm': {'atomic_number': 62, 'atomic_mass': 150.36, 'electronegativity': 1.17, 'atomic_radius': 1.62, 'period': 6, 'group': 101},
            'Eu': {'atomic_number': 63, 'atomic_mass': 151.964, 'electronegativity': 1.2, 'atomic_radius': 1.85, 'period': 6, 'group': 101},
            'Gd': {'atomic_number': 64, 'atomic_mass': 157.25, 'electronegativity': 1.20, 'atomic_radius': 1.61, 'period': 6, 'group': 101},
            'Tb': {'atomic_number': 65, 'atomic_mass': 158.925, 'electronegativity': 1.10, 'atomic_radius': 1.59, 'period': 6, 'group': 101},
            'Dy': {'atomic_number': 66, 'atomic_mass': 162.5, 'electronegativity': 1.22, 'atomic_radius': 1.59, 'period': 6, 'group': 101},
            'Ho': {'atomic_number': 67, 'atomic_mass': 164.930, 'electronegativity': 1.23, 'atomic_radius': 1.58, 'period': 6, 'group': 101},
            'Er': {'atomic_number': 68, 'atomic_mass': 167.259, 'electronegativity': 1.24, 'atomic_radius': 1.57, 'period': 6, 'group': 101},
            'Tm': {'atomic_number': 69, 'atomic_mass': 168.934, 'electronegativity': 1.25, 'atomic_radius': 1.56, 'period': 6, 'group': 101},
            'Yb': {'atomic_number': 70, 'atomic_mass': 173.054, 'electronegativity': 1.1, 'atomic_radius': 1.74, 'period': 6, 'group': 101},
            'Lu': {'atomic_number': 71, 'atomic_mass': 174.967, 'electronegativity': 1.27, 'atomic_radius': 1.56, 'period': 6, 'group': 3},
            'Hf': {'atomic_number': 72, 'atomic_mass': 178.49, 'electronegativity': 1.3, 'atomic_radius': 1.44, 'period': 6, 'group': 4},
            'Ta': {'atomic_number': 73, 'atomic_mass': 180.948, 'electronegativity': 1.5, 'atomic_radius': 1.34, 'period': 6, 'group': 5},
            'W': {'atomic_number': 74, 'atomic_mass': 183.84, 'electronegativity': 2.36, 'atomic_radius': 1.30, 'period': 6, 'group': 6},
            'Re': {'atomic_number': 75, 'atomic_mass': 186.207, 'electronegativity': 1.9, 'atomic_radius': 1.28, 'period': 6, 'group': 7},
            'Os': {'atomic_number': 76, 'atomic_mass': 190.23, 'electronegativity': 2.2, 'atomic_radius': 1.26, 'period': 6, 'group': 8},
            'Ir': {'atomic_number': 77, 'atomic_mass': 192.217, 'electronegativity': 2.20, 'atomic_radius': 1.27, 'period': 6, 'group': 9},
            'Pt': {'atomic_number': 78, 'atomic_mass': 195.084, 'electronegativity': 2.28, 'atomic_radius': 1.30, 'period': 6, 'group': 10},
            'Au': {'atomic_number': 79, 'atomic_mass': 196.967, 'electronegativity': 2.54, 'atomic_radius': 1.34, 'period': 6, 'group': 11},
            'Hg': {'atomic_number': 80, 'atomic_mass': 200.59, 'electronegativity': 2.00, 'atomic_radius': 1.49, 'period': 6, 'group': 12},
            'Tl': {'atomic_number': 81, 'atomic_mass': 204.383, 'electronegativity': 1.62, 'atomic_radius': 1.48, 'period': 6, 'group': 13},
            'Pb': {'atomic_number': 82, 'atomic_mass': 207.2, 'electronegativity': 2.33, 'atomic_radius': 1.47, 'period': 6, 'group': 14},
            'Bi': {'atomic_number': 83, 'atomic_mass': 208.980, 'electronegativity': 2.02, 'atomic_radius': 1.46, 'period': 6, 'group': 15},
            'Po': {'atomic_number': 84, 'atomic_mass': 209.0, 'electronegativity': 2.0, 'atomic_radius': 1.53, 'period': 6, 'group': 16},
            'At': {'atomic_number': 85, 'atomic_mass': 210.0, 'electronegativity': 2.2, 'atomic_radius': 1.43, 'period': 6, 'group': 17},
            'Rn': {'atomic_number': 86, 'atomic_mass': 222.0, 'electronegativity': 2.2, 'atomic_radius': 1.34, 'period': 6, 'group': 18},
            'Ac': {'atomic_number': 89, 'atomic_mass': 227.0, 'electronegativity': 1.1, 'atomic_radius': 1.88, 'period': 7, 'group': 3},
            'Th': {'atomic_number': 90, 'atomic_mass': 232.038, 'electronegativity': 1.3, 'atomic_radius': 1.65, 'period': 7, 'group': 102},
            'Pa': {'atomic_number': 91, 'atomic_mass': 231.036, 'electronegativity': 1.5, 'atomic_radius': 1.61, 'period': 7, 'group': 102},
            'U': {'atomic_number': 92, 'atomic_mass': 238.029, 'electronegativity': 1.38, 'atomic_radius': 1.58, 'period': 7, 'group': 102},
            'Np': {'atomic_number': 93, 'atomic_mass': 237.0, 'electronegativity': 1.36, 'atomic_radius': 1.55, 'period': 7, 'group': 102},
            'Pu': {'atomic_number': 94, 'atomic_mass': 244.0, 'electronegativity': 1.28, 'atomic_radius': 1.53, 'period': 7, 'group': 102},
        }
    
    def parse_formula(self, formula):
        """Parse chemical formula and return element counts"""
        # Remove spaces and split on capital letters
        elements = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
        composition = {}
        
        for element, count in elements:
            count = int(count) if count else 1
            composition[element] = composition.get(element, 0) + count
            
        return composition
    
    def get_element_features(self, element):
        """Get features for a given element"""
        if element in self.atomic_properties:
            props = self.atomic_properties[element]
            return [
                props['atomic_number'],
                props['atomic_mass'],
                props['electronegativity'],
                props['atomic_radius'],
                props['period'],
                props['group']
            ]
        else:
            # Default values for unknown elements
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

class MaterialsGraphDataset:
    """Dataset class for materials data"""

    def __init__(self, csv_path=None, data=None, target_col='band_gap'):
        self.parser = ChemicalFormulaParser()
        self.scaler = StandardScaler()
        self.crystal_encoder = LabelEncoder()
        self.space_group_encoder = LabelEncoder()
        self.target_col = target_col  # e.g., 'band_gap'

        if data is not None:
            self.df = data.copy()
        elif csv_path:
            self.df = pd.read_csv(csv_path)
        else:
            # Minimal sample (unchanged)
            sample_data = {
                'mp_id': ['mp-864606'],
                'formula': ['AcCuO3'],
                'sites': [5],
                'composition': ['Ac1 Cu1 O3'],
                'a_edge (angstrom)': [3.91333125],
                'b_edge (angstrom)': [3.91333125],
                'c_edge (angstrom)': [3.91333125],
                'alpha_ang (deg)': [90.0],
                'beta_ang (deg)': [90.0],
                'gamma_ang (deg)': [90.0],
                'crystal_system': ['cubic'],
                'space_group': ['Pm-3m'],
                'volume (cubic-angstrom)': [1.6e-06],
                'total_magnetisation (bohr)': [-7.035745362],
                'energy_per_atom (eV/atom)': [-2.422892424],
                'formation_energy (eV/atom)': [0.0],
                'energy_above_hull (eV/atom)': [0.0],
                'stable': [True],
                'band_gap (eV)': [9.380470938],
                'direct_bandgap': [False],
                'density (g/cc)': [59.92938666],
                'bulk_modulus (GPa)': [163.571],
                'shear_modulus (GPa)': [81.723],
            }
            self.df = pd.DataFrame(sample_data)

        self.prepare_data()

    def prepare_data(self):
        """Rename columns to a consistent internal schema, encode categoricals, and scale numeric features (no target leakage)."""

        # 1) Normalize column names from your CSV (unit-suffixed) to simple names
        rename_map = {
            'a_edge (angstrom)': 'a_edge',
            'b_edge (angstrom)': 'b_edge',
            'c_edge (angstrom)': 'c_edge',
            'alpha_ang (deg)': 'alpha_ang',
            'beta_ang (deg)': 'beta_ang',
            'gamma_ang (deg)': 'gamma_ang',
            'volume (cubic-angstrom)': 'volume',
            'total_magnetisation (bohr)': 'total_magnetisation',
            'energy_per_atom (eV/atom)': 'energy_per_atom',
            'formation_energy (eV/atom)': 'formation_energy',
            'energy_above_hull (eV/atom)': 'energy_above_hull',
            'band_gap (eV)': 'band_gap',
            'density (g/cc)': 'density',
            'bulk_modulus (GPa)': 'bulk_modulus',
            'shear_modulus (GPa)': 'shear_modulus',
        }
        self.df.rename(columns={k: v for k, v in rename_map.items() if k in self.df.columns}, inplace=True)

        # 2) Basic type fixes
        if 'stable' in self.df.columns:
            self.df['stable'] = self.df['stable'].astype(int)
        if 'direct_bandgap' in self.df.columns:
            # ensure 0/1 (some CSVs store True/False)
            self.df['direct_bandgap'] = self.df['direct_bandgap'].astype(int)

        # 3) Encode categoricals
        for col in ['crystal_system', 'space_group']:
            if col not in self.df.columns:
                # create placeholders if missing
                self.df[col] = 'Unknown'

        self.df['crystal_system_encoded'] = self.crystal_encoder.fit_transform(self.df['crystal_system'].astype(str))
        self.df['space_group_encoded'] = self.space_group_encoder.fit_transform(self.df['space_group'].astype(str))

        # 4) Decide target column (do NOT scale target; avoid leakage)
        if self.target_col not in self.df.columns:
            raise KeyError(f"Target column '{self.target_col}' not found. Available: {list(self.df.columns)}")

        # 5) Continuous feature columns to scale (exclude target and encoded categoricals)
        scale_cols = [c for c in [
            'a_edge', 'b_edge', 'c_edge', 'alpha_ang', 'beta_ang', 'gamma_ang',
            'volume', 'total_magnetisation', 'energy_per_atom', 'formation_energy',
            'energy_above_hull', 'density', 'bulk_modulus', 'shear_modulus'
        ] if c in self.df.columns]

        # Fit/transform scaler on available columns
        if scale_cols:
            self.df[scale_cols] = self.scaler.fit_transform(self.df[scale_cols])

        # 6) Define the exact material feature order to be concatenated to each node (NO target here)
        self.feature_cols = [c for c in [
            'sites', 'a_edge', 'b_edge', 'c_edge',
            'alpha_ang', 'beta_ang', 'gamma_ang',
            'crystal_system_encoded', 'space_group_encoded',
            'volume', 'total_magnetisation', 'energy_per_atom',
            'formation_energy', 'energy_above_hull', 'stable',
            'direct_bandgap', 'density', 'bulk_modulus', 'shear_modulus'
        ] if c in self.df.columns]  # include only those present

    def create_molecular_graph(self, formula, material_features):
        """Create a molecular graph from chemical formula (no target leakage)."""
        composition = self.parser.parse_formula(formula)

        # Node features = atomic properties
        node_features = []
        elements = list(composition.keys())
        for element, _count in composition.items():
            node_features.append(self.parser.get_element_features(element))

        # Center = highest electronegativity (keep your logic)
        center_idx = 0
        max_e = -1
        for i, el in enumerate(elements):
            e = self.parser.atomic_properties.get(el, {}).get('electronegativity', 0.0)
            if e > max_e:
                max_e = e
                center_idx = i

        # Star edges around center + self-loops
        edge_index = []
        n = len(elements)
        for i in range(n):
            if i != center_idx:
                edge_index.append([center_idx, i])
                edge_index.append([i, center_idx])
        for i in range(n):
            edge_index.append([i, i])
        if not edge_index:
            edge_index = [[0, 0]]

        # Tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Append MATERIAL-LEVEL features to every node (no target col)
        mf = torch.tensor(material_features, dtype=torch.float).unsqueeze(0).expand(n, -1)
        x = torch.cat([x, mf], dim=1)
        return Data(x=x, edge_index=edge_index)

    def create_graphs(self):
        """Create graphs for all materials in the dataset."""
        graphs = []
        for _, row in self.df.iterrows():
            formula = row['formula'] if 'formula' in self.df.columns else row['composition']
            material_features = [row[c] for c in self.feature_cols]
            g = self.create_molecular_graph(formula, material_features)

            # y = unscaled target (no leakage)
            g.y = torch.tensor([float(row[self.target_col])], dtype=torch.float)
            graphs.append(g)
        return graphs


class GCN(nn.Module):
    """Graph Convolutional Network"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.2):
        super(GCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index, batch):
        # Graph convolutions
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level representation
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x = torch.cat([x1, x2], dim=1)
        
        # Final prediction
        x = self.classifier(x)
        return x

class GraphSAGE(nn.Module):
    """GraphSAGE Network"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.2):
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, edge_index, batch):
        # Graph convolutions
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level representation
        x1 = global_mean_pool(x, batch)
        x2 = global_max_pool(x, batch)
        x = torch.cat([x1, x2], dim=1)
        
        # Final prediction
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001):
    """Training function"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    model.train()
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # Training
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.squeeze(), batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out.squeeze(), batch.y.squeeze())
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader):
    """Evaluation function"""
    model.eval()
    predictions = []
    targets = []
    total_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out.squeeze(), batch.y.squeeze())
            total_loss += loss.item()
            
            predictions.extend(out.squeeze().cpu().numpy())
            targets.extend(batch.y.squeeze().cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    
    # Calculate metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    if len(np.unique(targets)) > 1:  # Avoid division by zero
        r2 = 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))
    else:
        r2 = 0.0
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'avg_loss': avg_loss,
        'predictions': predictions,
        'targets': targets
    }

def main():
    """Main execution function"""
    print("Initializing Materials GNN Pipeline...")
    
    # Create dataset (using sample data - replace with your CSV path)
    print("Creating dataset...")
    dataset = MaterialsGraphDataset('cleaned_data_vrh.csv')  # Replace with: MaterialsGraphDataset('your_data.csv')
    
    # Create graphs
    print("Creating molecular graphs...")
    graphs = dataset.create_graphs()
    
    print(f"Created {len(graphs)} graphs")
    print(f"Sample graph - Nodes: {graphs[0].x.shape[0]}, Features: {graphs[0].x.shape[1]}")
    
    # Split data
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.2, random_state=42)
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    # Model parameters
    input_dim = graphs[0].x.shape[1]  # Number of node features
    hidden_dim = 128
    output_dim = 1  # Predicting band gap
    num_layers = 3
    dropout = 0.2
    
    print(f"Input dimension: {input_dim}")
    
    # Initialize models
    print("Initializing models...")
    gcn_model = GCN(input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)
    sage_model = GraphSAGE(input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)
    
    print(f"GCN Model Parameters: {sum(p.numel() for p in gcn_model.parameters())}")
    print(f"GraphSAGE Model Parameters: {sum(p.numel() for p in sage_model.parameters())}")
    
    # Train GCN
    print("\n" + "="*50)
    print("Training GCN Model...")
    print("="*50)
    gcn_train_losses, gcn_val_losses = train_model(gcn_model, train_loader, val_loader, num_epochs=100)
    
    # Train GraphSAGE
    print("\n" + "="*50)
    print("Training GraphSAGE Model...")
    print("="*50)
    sage_train_losses, sage_val_losses = train_model(sage_model, train_loader, val_loader, num_epochs=100)
    
    # Evaluate models
    print("\n" + "="*50)
    print("Evaluating Models...")
    print("="*50)
    
    gcn_results = evaluate_model(gcn_model, test_loader)
    sage_results = evaluate_model(sage_model, test_loader)
    
    print("GCN Results:")
    print(f"  MSE: {gcn_results['mse']:.4f}")
    print(f"  MAE: {gcn_results['mae']:.4f}")
    print(f"  R²: {gcn_results['r2']:.4f}")
    
    print("GraphSAGE Results:")
    print(f"  MSE: {sage_results['mse']:.4f}")
    print(f"  MAE: {sage_results['mae']:.4f}")
    print(f"  R²: {sage_results['r2']:.4f}")
    
    def save_training_curves(train_losses, val_losses, out_dir, model_name):
        plt.figure(figsize=(6, 4))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title(f"Training Curves - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"training_curves_{model_name}.jpeg"), dpi=300)
        plt.close()

    # --- Save predictions scatter ---
    def save_prediction_scatter(y_true, y_pred, out_dir, model_name, target_name="Property"):
        plt.figure(figsize=(5, 5))
        plt.scatter(y_true, y_pred, alpha=0.6)
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # y=x line
        plt.xlabel(f"True {target_name}")
        plt.ylabel(f"Predicted {target_name}")
        plt.title(f"Predictions vs True - {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"predictions_{model_name}.jpeg"), dpi=300)
        plt.close()

    save_prediction_scatter(gcn_results['targets'], gcn_results['predictions'], out_dir="outputs", model_name="GCN", target_name=dataset.target_col)
    save_prediction_scatter(sage_results['targets'], sage_results['predictions'], out_dir="outputs", model_name="GraphSAGE", target_name=dataset.target_col)

    # Save models
    print("\nSaving models...")
    torch.save(gcn_model.state_dict(), 'gcn_materials_model.pth')
    torch.save(sage_model.state_dict(), 'sage_materials_model.pth')
    
    print("Training complete!")
    
    return {
        'gcn_model': gcn_model,
        'sage_model': sage_model,
        'gcn_results': gcn_results,
        'sage_results': sage_results,
        'dataset': dataset
    }

# Example usage for inference
def predict_properties(model, formula, material_features, dataset):
    """Predict properties for a new material"""
    model.eval()
    
    # Create graph for the new material
    graph = dataset.create_molecular_graph(formula, material_features)
    
    # Create batch
    batch = Data(x=graph.x, edge_index=graph.edge_index, batch=torch.zeros(graph.x.shape[0], dtype=torch.long))
    batch = batch.to(device)
    
    with torch.no_grad():
        prediction = model(batch.x, batch.edge_index, batch.batch)
    
    return prediction.cpu().numpy()

if __name__ == "__main__":
    # Run the main pipeline
    results = main()
    
    # Example prediction (uncomment to use)
    # new_material_features = [5, 3.91, 3.91, 3.91, 90.0, 90.0, 90.0, 0, 0, 1.6e-06, 
    #                         -7.04, -2.42, 0.0, 0.0, 1, 9.38, 0.0, 59.93, 163.57, 81.72]
    # 
    # gcn_pred = predict_properties(results['gcn_model'], 'AcCuO3', new_material_features, results['dataset'])
    # sage_pred = predict_properties(results['sage_model'], 'AcCuO3', new_material_features, results['dataset'])
    # 
    # print(f"GCN Prediction: {gcn_pred[0]:.4f}")
    # print(f"GraphSAGE Prediction: {sage_pred[0]:.4f}")



# Additional utility functions
def visualize_training_curves(train_losses, val_losses, model_name):
    """Visualize training curves (requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Training Curves')
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("matplotlib not available for visualization")

def analyze_predictions(targets, predictions, model_name):
    """Analyze model predictions (requires matplotlib)"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, predictions, alpha=0.6)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name} Predictions vs Actual')
        plt.grid(True)
        plt.show()
    except ImportError:
        print("matplotlib not available for visualization")
