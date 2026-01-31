import h5py
from IEProtLib.py_utils.py_mol import PyPeriodicTable, PyProtein, PyProteinBatch
import numpy as np
import torch
import math
import warnings
import os
import pandas as pd
import dgl
import scipy.spatial as spa
from Bio.PDB import get_surface, PDBParser, ShrakeRupley
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import MolFromPDBFile, AllChem, GetPeriodicTable, rdDistGeom
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from scipy import spatial
from scipy.special import softmax
from scipy.spatial import distance
from tqdm import tqdm


mass_ = np.array([1.008, 4.002602, 6.94, 9.0121831, 10.81, 12.011, 14.007, 15.999, 18.998403163, 20.1797, 
        22.98976928, 24.305, 26.9815385, 28.085, 30.973761998, 32.06, 35.45, 39.948, 39.0983, 40.078, 
        44.955908, 47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546, 65.38, 
        69.723, 72.63, 74.921595, 78.971, 79.904, 83.798, 85.4678, 87.62, 88.90584, 91.224, 92.90637, 
        95.95, 97.0, 101.07, 102.9055, 106.42, 107.8682, 112.414, 114.818, 118.71, 121.76, 127.6, 
        126.90447, 131.293, 132.90545196, 137.327, 138.90547, 140.116, 140.90766, 144.242, 145.0, 
        150.36, 151.964, 157.25, 158.92535, 162.5, 164.93033, 167.259, 168.93422, 173.054, 174.9668, 
        178.49, 180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966569, 200.592, 204.38, 
        207.2, 208.9804, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.0377, 231.03588, 238.02891, 
        237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0, 258.0, 259.0, 262.0, 267.0, 270.0, 
        271.0, 270.0, 277.0, 276.0, 281.0, 282.0, 285.0, 285.0, 289.0, 288.0, 293.0, 294.0, 294.0])

covRadius_ = np.array([0.32, 0.28, 1.34, 0.96, 0.84, 0.72, 0.68, 0.68, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 
        1.036, 1.02, 1.02, 1.06, 2.03, 0.992, 1.70, 1.60, 1.53, 1.39, 1.19, 1.42, 1.11, 1.24, 1.32, 
        1.448, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 
        1.42, 1.39, 1.45, 1.668, 1.42, 1.39, 1.39, 1.38, 1.4, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 
        1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 
        1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 
        1.90, 1.87, 1.80, 1.69, 1.66, 1.68, 1.65, 1.67, 1.73, 1.76, 1.61, 1.57, 1.49, 1.43, 1.41, 1.34, 
        1.29, 1.28, 1.21, 1.22, 1.36, 1.43, 1.62, 1.75, 1.65, 1.57])

vdwRadius_ = np.array([1.2, 1.4, 1.82, 2.0, 2.0, 1.7, 1.55, 1.52, 1.47, 1.54, 2.27, 1.73, 2.0, 2.1, 1.8, 1.8, 
        1.75, 1.88, 2.75, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.63, 1.4, 1.39, 1.87, 2.0, 1.85, 1.9, 
        1.85, 2.02, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.63, 1.72, 1.58, 1.93, 2.17, 2.0, 2.06, 
        1.98, 2.16, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.75, 1.66, 1.55, 1.96, 2.02, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 
        2.0, 1.86, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])


hydrophobicity_index = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5,
    'CYS': 2.5, 'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4,
    'HIS': -3.2, 'ILE': 4.5, 'LEU': 3.8, 'LYS': -3.9,
    'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6, 'SER': -0.8,
    'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2
}

amino_acid_polarity = {
    'ALA': 0.0, 'ARG': 1.0, 'ASN': 0.5, 'ASP': -1.0, 'CYS': 0.0,
    'GLN': 0.5, 'GLU': -1.0, 'GLY': 0.0, 'HIS': 0.5, 'ILE': 0.0,
    'LEU': 0.0, 'LYS': 1.0, 'MET': 0.0, 'PHE': 0.0, 'PRO': 0.0,
    'SER': 0.5, 'THR': 0.5, 'TRP': 0.0, 'TYR': 0.5, 'VAL': 0.0
}

biopython_parser = PDBParser()
sr = ShrakeRupley(probe_radius=1.4,
                  n_points=100)

def compute_feat(hdf5_file):
    atom_mass, atom_covRadius, atom_vdwRadius, atom_hyd, atom_polar, atom_sasa, atom_bfactor = [], [], [], [], [], [], []
    periodicTable_ = PyPeriodicTable()
    curProtein = PyProtein(periodicTable_)
    # if 'positive' in hdf5_file:
    #     pdb_file = '/root/autodl-tmp/data/domain_pdb_p/' + hdf5_file.split('/')[-1].rstrip('hdf5') + 'pdb'
    # else:
    #     pdb_file = '/root/autodl-tmp/data/domain_pdb_n/' + hdf5_file.split('/')[-1].rstrip('hdf5') + 'pdb'
    pdb_file = '/root/autodl-tmp/moad_domain_pdb/' + hdf5_file.split('/')[-1].rstrip('hdf5') + 'pdb'
    
    # hdf5_file = '/root/autodl-tmp/domain_atom_hdf5/8prn_1_positive.hdf5'
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', pdb_file)
        rec = structure[0]
    sr.compute(rec, level="R")
    for residue in rec.get_residues():
        sasa = residue.sasa
        for atom in residue:
            atom_sasa.append(sasa)
            bfactor = atom.bfactor
            atom_bfactor.append(bfactor)   
    
    curProtein.load_hdf5(hdf5_file)
    atom_types = curProtein.atomTypes_
    atom_pos = curProtein.atomPos_
    atom_aminoid = curProtein.atomAminoIds_
    atom_residue_names = curProtein.atomResidueNames_
    
    adj_cov = curProtein.covBondList_
    adj_hb = curProtein.covBondListHB_
    
    x = curProtein.atomCovBondSIndicesHB_ 
    for atom_id in atom_types:
        atom_mass.append(mass_[atom_id])
        atom_covRadius.append(covRadius_[atom_id])
        atom_vdwRadius.append(vdwRadius_[atom_id])
    for atom_residue_name in atom_residue_names:
        atom_hyd.append(hydrophobicity_index[atom_residue_name])
        atom_polar.append(amino_acid_polarity[atom_residue_name])
    atom_aminoid = torch.tensor(atom_aminoid).view(-1, 1)
    atom_mass = torch.tensor(np.array(atom_mass)).view(-1, 1)#原子质量
    atom_covRadius = torch.tensor(np.array(atom_covRadius)).view(-1, 1)#共价键半径
    atom_vdwRadius = torch.tensor(np.array(atom_vdwRadius)).view(-1, 1)#范德华半径
    atom_hyd = torch.tensor(np.array(atom_hyd)).view(-1, 1)#亲水性
    atom_polar = torch.tensor(np.array(atom_polar)).view(-1, 1)#极性
    atom_sasa = torch.tensor(np.array(atom_sasa)).view(-1, 1)
    atom_bfactor = torch.tensor(np.array(atom_bfactor)).view(-1, 1) 
    h_feat = torch.cat((atom_aminoid, atom_mass, atom_covRadius, atom_vdwRadius, atom_hyd, atom_polar, atom_sasa, atom_bfactor), dim=1)
    x_feat = atom_pos.reshape(-1, 3)
    x_feat = torch.tensor(x_feat)
    
    return h_feat, x_feat



def create_graph(hdf5_file):
    periodicTable_ = PyPeriodicTable()
    curProtein = PyProtein(periodicTable_)
    curProtein.load_hdf5(hdf5_file)
    adj_cov = curProtein.covBondList_
    # print(adj_cov)
    adj_hb = curProtein.covBondListHB_
    atom_pos = curProtein.atomPos_
    atom_pos = atom_pos.reshape(-1, 3)
    dist_matrix = distance.cdist(atom_pos, atom_pos, 'euclidean')
    np.fill_diagonal(dist_matrix, np.inf)

    '''===============pos_adj==============='''
    src_pos = []
    dst_pos = []


    for i in range(len(dist_matrix)):
        # 找到每个点的三个最近邻
        neighbors = np.argsort(dist_matrix[i])[:3]
        for neighbor in neighbors:
            src_pos.append(i)
            dst_pos.append(neighbor)
    # 转换为Tensor
    src_pos = torch.tensor(src_pos)
    dst_pos = torch.tensor(dst_pos)
    
    '''===============cov_adj==============='''
    src_cov = adj_cov[:, 0]
    dst_cov = adj_cov[:, 1]
    src_cov = torch.tensor(src_cov)
    dst_cov = torch.tensor(dst_cov)
    # print(src_cov.size())
    # print(dst_cov.size())
    
    
    '''===============hb_adj==============='''
    src_hb = adj_hb[:, 0]
    dst_hb = adj_hb[:, 1]
    src_hb = torch.tensor(src_hb)
    dst_hb = torch.tensor(dst_hb)
    # print(src_hb.size())
    # print(dst_hb.size())
    
    h_feat, x_feat = compute_feat(hdf5_file)
    
    graph_data = {
    ('node', 'pos', 'node'): (src_pos, dst_pos),
    ('node', 'cov', 'node'): (src_cov, dst_cov),
    ('node', 'hb', 'node'): (src_hb, dst_hb),
    }

    hetero_graph = dgl.heterograph(graph_data)
    h_feat, x_feat = compute_feat(hdf5_file)
    hetero_graph.ndata['feat'] = h_feat
    hetero_graph.ndata['x'] = x_feat
    # print(graph_data)
    # print(hetero_graph)
    # print(hetero_graph.ndata['feat'].size())
    # print(hetero_graph.ndata['x'].size())
    return hetero_graph
    
def save_graph_to_hdf5(graph, filename):
    with h5py.File(filename, 'w') as f:
        # Save node features
        for key, value in graph.ndata.items():
            f.create_dataset(f'ndata/{key}', data=value.cpu().numpy())

        # Save edge features
        for etype in graph.canonical_etypes:
            src, dst = graph.edges(etype=etype)
            src, dst = src.cpu().numpy(), dst.cpu().numpy()
            f.create_dataset(f'edata/{etype}/src', data=src)
            f.create_dataset(f'edata/{etype}/dst', data=dst)
    
    
    
if __name__ == "__main__":
    
    for filename in tqdm(os.listdir('/root/autodl-tmp/domain_atom_hdf5/')):
        if filename.endswith('.hdf5'):
            try:
                hdf5_path = os.path.join('/root/autodl-tmp/domain_atom_hdf5/', filename)
                graph = create_graph(hdf5_path)
                save_graph_to_hdf5(graph, f'/root/autodl-tmp/domain_graph/{filename}')
            except:
                with open('./wrong_domain_graph.txt', 'a+') as f:
                    f.write(filename + '\n')
                    

    for filename in tqdm(os.listdir('/root/autodl-tmp/moad_domain_pdb/')):
        if filename.endswith('.pdb'):
            filename = filename.rstrip('pdb') + 'hdf5'
            try:
                hdf5_path = os.path.join('/root/autodl-tmp/moad_hdf5/', filename)
                graph = create_graph(hdf5_path)
                save_graph_to_hdf5(graph, f'/root/autodl-tmp/moad_hdf5/{filename}')
            except:
                with open('./moad_wrong_domain_graph.txt', 'a+') as f:
                    f.write(filename + '\n')
                    
    filename = '8prn_1_positive.hdf5'
    hdf5_path = os.path.join('/root/autodl-tmp/domain_atom_hdf5/', filename)
    print(hdf5_path)
    graph = create_graph(hdf5_path)
    print(graph)
    
    save_graph_to_hdf5(graph, f'/root/autodl-tmp/domain_graph/{filename}')
    
    filename = '4o79_1_negative.pdb'
    filename = filename.rstrip('pdb') + 'hdf5'
    hdf5_path = os.path.join('/root/autodl-tmp/moad_hdf5/', filename)
    graph = create_graph(hdf5_path)
    print(graph)
    save_graph_to_hdf5(graph, f'/root/autodl-tmp/moad_hdf5/{filename}')

    
    for i in os.listdir('/root/autodl-tmp/domain_atom_hdf5/')
    
    compute_feat('/root/autodl-tmp/domain_atom_hdf5/8prn_1_positive.hdf5')
    create_graph('/root/autodl-tmp/domain_atom_hdf5/8prn_1_positive.hdf5')