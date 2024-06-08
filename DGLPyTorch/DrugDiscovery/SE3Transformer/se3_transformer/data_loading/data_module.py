# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import torch.distributed as dist
from abc import ABC
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from dgl import DGLGraph
from torch import Tensor
from rdkit import Chem
import dgl
from torch.utils.data import Dataset
import torch
import pathlib
from torch.utils.data import DataLoader, random_split
from se3_transformer.data_loading.data_module import DataModule
from se3_transformer.runtime.utils import get_local_rank


def _get_dataloader(dataset: Dataset, shuffle: bool, **kwargs) -> DataLoader:
    # Classic or distributed dataloader depending on the context
    sampler = DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
    return DataLoader(dataset, shuffle=(shuffle and sampler is None), sampler=sampler, **kwargs)


class DataModule(ABC):
    """ Abstract DataModule. Children must define self.ds_{train | val | test}. """

    def __init__(self, **dataloader_kwargs):
        super().__init__()
        if get_local_rank() == 0:
            self.prepare_data()

        # Wait until rank zero has prepared the data (download, preprocessing, ...)
        if dist.is_initialized():
            dist.barrier(device_ids=[get_local_rank()])

        self.dataloader_kwargs = {'pin_memory': True, 'persistent_workers': dataloader_kwargs.get('num_workers', 0) > 0,
                                  **dataloader_kwargs}
        self.ds_train, self.ds_val, self.ds_test = None, None, None

    def prepare_data(self):
        """ Method called only once per node. Put here any downloading or preprocessing """
        pass

    def train_dataloader(self) -> DataLoader:
        return _get_dataloader(self.ds_train, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return _get_dataloader(self.ds_val, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return _get_dataloader(self.ds_test, shuffle=False, **self.dataloader_kwargs)



def _get_relative_pos(qm9_graph: DGLGraph) -> Tensor: # from qm9.py
    x = qm9_graph.ndata['pos']
    src, dst = qm9_graph.edges()
    rel_pos = x[dst] - x[src]
    return rel_pos

class LeashDataset(Dataset):
    def __init__(self, smiles_list, targets):
        self.smiles_list = smiles_list
        self.targets = targets
        self.atom_types = {6: [1, 0, 0, 0, 0],  # C
                           7: [0, 1, 0, 0, 0],  # N
                           8: [0, 0, 1, 0, 0],  # O
                           1: [0, 0, 0, 1, 0],  # H
                           9: [0, 0, 0, 0, 1]}  # F

        self.bond_types = {Chem.rdchem.BondType.SINGLE: [1, 0, 0, 0],
                           Chem.rdchem.BondType.DOUBLE: [0, 1, 0, 0],
                           Chem.rdchem.BondType.TRIPLE: [0, 0, 1, 0],
                           Chem.rdchem.BondType.AROMATIC: [0, 0, 0, 1]}

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        target = self.targets[idx] if self.targets is not None else None
        mol = Chem.MolFromSmiles(smiles)
        graph = self.mol_to_dgl_graph(mol)
        if target is not None:
            return graph, torch.tensor(target, dtype=torch.float32)
        else:
            return graph

    def mol_to_dgl_graph(self, mol):
        g = dgl.DGLGraph()
        g.add_nodes(mol.GetNumAtoms())

        # Node features
        node_features = []
        for atom in mol.GetAtoms():
            atom_type = self.atom_types.get(atom.GetAtomicNum(), [0, 0, 0, 0, 0])
            node_features.append(atom_type + [atom.GetAtomicNum()])
        g.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)

        # Edges and edge features
        src, dst, edge_features = [], [], []
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bond_type = self.bond_types.get(bond.GetBondType(), [0, 0, 0, 0])
            src.extend([u, v])
            dst.extend([v, u])
            edge_features.extend([bond_type, bond_type])
        g.add_edges(src, dst)
        g.edata['feat'] = torch.tensor(edge_features, dtype=torch.float32)
        
        return g

#### Continue from Update the DataModule Class: onward
#### edit so it gives positions

class CustomDataModule(DataModule):
    NODE_FEATURE_DIM = 6  # 5 (one-hot atom type) + 1 (number of protons)
    EDGE_FEATURE_DIM = 4  # One-hot-encoded bond type

    def __init__(self,
                 data_dir: pathlib.Path,
                 smiles_list,
                 targets,
                 batch_size: int = 240,
                 num_workers: int = 8,
                 **kwargs):
        self.data_dir = data_dir  # This needs to be before __init__ so that prepare_data has access to it
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=self._collate)
        self.smiles_list = smiles_list
        self.targets = targets
        self.batch_size = batch_size

        full_dataset = LeashDataset(smiles_list, targets)
        self.ds_train, self.ds_val, self.ds_test = random_split(full_dataset, self._get_split_sizes(full_dataset),
                                                                generator=torch.Generator().manual_seed(0))

    def prepare_data(self):
        # Prepare data if needed (e.g., download, preprocess)
        pass

    def _get_split_sizes(self, full_dataset):
        len_full = len(full_dataset)
        len_train = int(0.8 * len_full)
        len_val = int(0.1 * len_full)
        len_test = len_full - len_train - len_val
        return len_train, len_val, len_test
    
    def _collate(self, samples):
        if all(isinstance(sample, tuple) and len(sample) == 2 for sample in samples):
            graphs, targets = map(list, zip(*samples))
            batched_graph = dgl.batch(graphs)
            edge_feats = {'0': batched_graph.edata['feat']}
            batched_graph.edata['rel_pos'] = _get_relative_pos(batched_graph) # --- maybe add in positions later
            node_feats = {'0': batched_graph.ndata['feat']}
            targets = torch.stack(targets) # qm9.py uses torch.cat()
            return batched_graph, node_feats, edge_feats, targets
        else :
            graphs = samples
            batched_graph = dgl.batch(graphs)
            edge_feats = {'0': batched_graph.edata['feat']}
            batched_graph.edata['rel_pos'] = _get_relative_pos(batched_graph) # --- maybe add in positions later
            node_feats = {'0': batched_graph.ndata['feat']}
            return batched_graph, node_feats, edge_feats

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Custom dataset")
        # Add custom arguments if necessary
        return parent_parser

    def __repr__(self):
        return f'CustomDataModule()'
