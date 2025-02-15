import sys
sys.path.append('./')

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import e3nn
from functools import partial

print("PyTorch version {}".format(torch.__version__))
print("PyG version {}".format(torch_geometric.__version__))
print("e3nn version {}".format(e3nn.__version__))

from experiments.utils.plot_utils import plot_3d
from experiments.utils.train_utils import run_experiment
from models import SchNetModel, DimeNetPPModel, SphereNetModel, EGNNModel, GVPGNNModel, TFNModel, MACEModel

# Set the device
device = torch.device('cpu')
print(f"Using device: {device}")

def create_kchains(k):
    assert k >= 2

    dataset = []

    # Graph 0
    atoms = torch.LongTensor([0] + [0] + [0] * (k - 1) + [0])
    edge_index = torch.LongTensor([[i for i in range((k + 2) - 1)], [i for i in range(1, k + 2)]])
    pos = torch.FloatTensor(
        [[-4, -3, 0]] +
        [[0, 5 * i, 0] for i in range(k)] +
        [[4, 5 * (k - 1) + 3, 0]]
    )
    center_of_mass = torch.mean(pos, dim=0)
    pos = pos - center_of_mass
    y = torch.LongTensor([0])  # Label 0
    data1 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data1.edge_index = to_undirected(data1.edge_index)
    dataset.append(data1)

    # Graph 1
    atoms = torch.LongTensor([0] + [0] + [0] * (k - 1) + [0])
    edge_index = torch.LongTensor([[i for i in range((k + 2) - 1)], [i for i in range(1, k + 2)]])
    pos = torch.FloatTensor(
        [[4, -3, 0]] +
        [[0, 5 * i, 0] for i in range(k)] +
        [[4, 5 * (k - 1) + 3, 0]]
    )
    center_of_mass = torch.mean(pos, dim=0)
    pos = pos - center_of_mass
    y = torch.LongTensor([1])  # Label 1
    data2 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data2.edge_index = to_undirected(data2.edge_index)
    dataset.append(data2)

    return dataset

def create_pool_kchains(k):
    dataset = []

    # Graph 0
    atoms = torch.LongTensor([0] + [0] + [0])
    edge_index = torch.LongTensor([[0, 1, 2], [1, 2, 0]])
    pos = torch.FloatTensor(
        [[-8, -3, 0]] +
        [[0, 5 * i, 0] for i in range(k)] +
        [[8, 5 * (k - 1) + 3, 0]]
    )
    center_of_mass = torch.mean(pos, dim=0)
    pos = pos - center_of_mass
    pos = torch.cat([pos[0,:].unsqueeze(0),
                     torch.mean(pos[1:-1], dim=0).unsqueeze(0),
                     pos[-1,:].unsqueeze(0)],
                    dim=0)
    y = torch.LongTensor([0])  # Label 0
    data1 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data1.edge_index = to_undirected(data1.edge_index)
    dataset.append(data1)

    # Graph 1
    atoms = torch.LongTensor([0] + [0] + [0])
    edge_index = torch.LongTensor([[0, 1, 2], [1, 2, 0]])
    pos = torch.FloatTensor(
        [[8, -3, 0]] +
        [[0, 5 * i, 0] for i in range(k)] +
        [[8, 5 * (k - 1) + 3, 0]]
    )
    center_of_mass = torch.mean(pos, dim=0)
    pos = pos - center_of_mass
    pos = torch.cat([pos[0,:].unsqueeze(0),
                     torch.mean(pos[1:-1], dim=0).unsqueeze(0),
                     pos[-1,:].unsqueeze(0)],
                    dim=0)
    y = torch.LongTensor([1])  # Label 1
    data2 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data2.edge_index = to_undirected(data2.edge_index)
    dataset.append(data2)

    return dataset

def create_pool_kchains_new(k):
    dataset = []

    # Graph 0
    atoms = torch.LongTensor([0] + [0]*2 + [0])
    edge_index = torch.LongTensor([[0, 1, 2, 3], [1, 2, 3, 0]])

    pos = torch.FloatTensor(
        [[-8, -3, 0]] +
        [[0, 5 * i, 0] for i in range(k)] +
        [[8, 5 * (k - 1) + 3, 0]]
    )
    center_of_mass = torch.mean(pos, dim=0)
    pos = pos - center_of_mass
    center_of_node = k // 2
    pos = torch.cat([pos[0,:].unsqueeze(0),
                     torch.mean(pos[1:center_of_node], dim=0).unsqueeze(0),
                     torch.mean(pos[center_of_node:-1], dim=0).unsqueeze(0),
                     pos[-1,:].unsqueeze(0)],
                    dim=0)
    y = torch.LongTensor([0])  # Label 0
    data1 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data1.edge_index = to_undirected(data1.edge_index)
    dataset.append(data1)

    # Graph 1
    atoms = torch.LongTensor([0] + [0]*2 + [0])
    edge_index = torch.LongTensor([[0, 1, 2, 3], [1, 2, 3, 0]])

    pos = torch.FloatTensor(
        [[8, -3, 0]] +
        [[0, 5 * i, 0] for i in range(k)] +
        [[8, 5 * (k - 1) + 3, 0]]
    )
    center_of_mass = torch.mean(pos, dim=0)
    pos = pos - center_of_mass
    pos = torch.cat([pos[0,:].unsqueeze(0),
                     torch.mean(pos[1:center_of_node], dim=0).unsqueeze(0),
                     torch.mean(pos[center_of_node:-1], dim=0).unsqueeze(0),
                     pos[-1,:].unsqueeze(0)],
                    dim=0)
    y = torch.LongTensor([1])  # Label 1
    data2 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data2.edge_index = to_undirected(data2.edge_index)
    dataset.append(data2)

    return dataset



k = 4

# Create dataset
dataset = create_pool_kchains_new(k) #create_pool_kchains(k)
for data in dataset:
    plot_3d(data, lim=5*k)

# Create dataloaders
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(dataset, batch_size=2, shuffle=False)

model_name = "dimenet"

for num_layers in range(k // 2, k + 3):
    print(f"\nNumber of layers: {num_layers}")

    correlation = 2
    model = {
        "schnet": SchNetModel,
        "dimenet": DimeNetPPModel,
        "spherenet": SphereNetModel,
        "egnn": EGNNModel,
        "gvp": partial(GVPGNNModel, s_dim=32, v_dim=1),
        "tfn": TFNModel,
        "mace": partial(MACEModel, correlation=correlation),
    }[model_name](num_layers=num_layers, in_dim=1, out_dim=2)

    best_val_acc, test_acc, train_time = run_experiment(
        model,
        dataloader,
        val_loader,
        test_loader,
        n_epochs=100,
        n_times=10,
        device=device,
        verbose=False
    )