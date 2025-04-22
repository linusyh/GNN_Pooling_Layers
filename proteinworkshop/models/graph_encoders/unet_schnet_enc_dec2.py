from typing import Optional, Set, Union

import torch
import torch_scatter
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch
from torch_geometric.nn.models import SchNet
from torch.nn import  Linear, ModuleList, Module, ReLU
from proteinworkshop.types import EncoderOutput
from torch_geometric.nn import fps, MLP, GINConv
from proteinworkshop.utils.graph import compute_new_edges


class SimpleMLP(Module):
    def __init__(self, hidden_channels):
        super(SimpleMLP, self).__init__()
        self.fc1 = Linear(hidden_channels, hidden_channels)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_channels, hidden_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class UnetSchNetModelEncDec2(SchNet):
    def __init__(
        self,
        hidden_channels: int = 128,
        out_dim: int = 1,
        num_filters: int = 128,
        num_layers: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10,
        max_num_neighbors: int = 32,
        readout: str = "add",
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: Optional[torch.Tensor] = None,
    ):
        """
        Initializes an instance of the SchNetModel class with the provided
        parameters.

        :param hidden_channels: Number of channels in the hidden layers
            (default: ``128``)
        :type hidden_channels: int
        :param out_dim: Output dimension of the model (default: ``1``)
        :type out_dim: int
        :param num_filters: Number of filters used in convolutional layers
            (default: ``128``)
        :type num_filters: int
        :param num_layers: Number of convolutional layers in the model
            (default: ``6``)
        :type num_layers: int
        :param num_gaussians: Number of Gaussian functions used for radial
            filters (default: ``50``)
        :type num_gaussians: int
        :param cutoff: Cutoff distance for interactions (default: ``10``)
        :type cutoff: float
        :param max_num_neighbors: Maximum number of neighboring atoms to
            consider (default: ``32``)
        :type max_num_neighbors: int
        :param readout: Global pooling method to be used (default: ``"add"``)
        :type readout: str
        """
        super().__init__(
            hidden_channels,
            num_filters,
            num_layers*2,
            num_gaussians,
            cutoff,  # None, # Interaction graph is not used
            max_num_neighbors,
            readout,
            dipole,
            mean,
            std,
            atomref,
        )
        
        self.lin_transformations = ModuleList()
        for _ in range(num_layers//2 - 1):
            lin= Linear(hidden_channels, hidden_channels)
            self.lin_transformations.append(lin)
        
        self.readout = readout
        # Overwrite embbeding
        self.embedding = torch.nn.LazyLinear(hidden_channels)
        # Overwrite atom embedding and final predictor
        self.lin2 = torch.nn.LazyLinear(out_dim)

        self.reds = ModuleList()
        for _ in range(num_layers//2 - 1):
            red = SimpleMLP(hidden_channels=hidden_channels)
            self.reds.append(red)

    @property
    def required_batch_attributes(self) -> Set[str]:
        """
        Required batch attributes for this encoder.

        - ``x``: Node features (shape: :math:`(n, d)`)
        - ``pos``: Node positions (shape: :math:`(n, 3)`)
        - ``edge_index``: Edge indices (shape: :math:`(2, e)`)
        - ``batch``: Batch indices (shape: :math:`(n,)`)

        :return: Set of required batch attributes
        :rtype: Set[str]
        """
        return {"pos", "edge_index", "x", "batch"}

    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the SchNet encoder.

        Returns the node embedding and graph embedding in a dictionary.

        :param batch: Batch of data to encode.
        :type batch: Union[Batch, ProteinBatch]
        :return: Dictionary of node and graph embeddings. Contains
            ``node_embedding`` and ``graph_embedding`` fields. The node
            embedding is of shape :math:`(|V|, d)` and the graph embedding is
            of shape :math:`(n, d)`, where :math:`|V|` is the number of nodes
            and :math:`n` is the number of graphs in the batch and :math:`d` is
            the dimension of the embeddings.
        :rtype: EncoderOutput
        """
        prev_idx = torch.arange(batch.x.size(0), dtype=torch.long, device=batch.x.device)
        new_prev_idx = prev_idx
        stack_down_idx = []
        stack_down_edges = []
        h = self.embedding(batch.x)
        u, v = batch.edge_index
        edge_index = batch.edge_index
        edge_weight = (batch.pos[u] - batch.pos[v]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)
        h = h + self.interactions[0](h, batch.edge_index, edge_weight, edge_attr)
        for i in range(1,len(self.interactions)//2):
            if i % 2 == 0:
                idx = fps(batch.pos, batch.batch, 0.6)
                row, col = edge_index
                new_prev_idx = prev_idx[idx]
                node_features_aggregated = torch_scatter.scatter_add(h[row], col, dim=0, dim_size=h.size(0)) + \
                    torch_scatter.scatter_add(h[col], row, dim=0, dim_size=h.size(0))
                h[new_prev_idx] = self.reds[i//2-1](node_features_aggregated[new_prev_idx] + h[new_prev_idx])
                # h[new_prev_idx] = self.reds[i//2-1](h, batch.edge_index)[new_prev_idx]
                pos = batch.pos[new_prev_idx]
                graphs = batch.batch[new_prev_idx]
                edge_index, _ = compute_new_edges(pos, graphs, 'knn_16')
                u, v = edge_index
                edge_weight = (pos[u] - pos[v]).norm(dim=-1)
                edge_attr = self.distance_expansion(edge_weight)

            h[new_prev_idx] = h[new_prev_idx] + self.interactions[i](h[new_prev_idx], edge_index, edge_weight, edge_attr)
            if i % 2 == 0:
                h[new_prev_idx] = self.lin_transformations[i//2-1](h[new_prev_idx])
                h[new_prev_idx] = self.act(h[new_prev_idx])

            h[new_prev_idx] = self.lin_transformations[-1](h[new_prev_idx])
            h[new_prev_idx] = self.act(h[new_prev_idx])
            prev_idx = new_prev_idx
            stack_down_idx.append(prev_idx)
            stack_down_edges.append(edge_index)

        for i in range(len(self.interactions)//2,len(self.interactions)):
            idx = stack_down_idx.pop()
            edge_index = stack_down_edges.pop()
            pos = batch.pos[new_prev_idx]
            u, v = edge_index
            edge_weight = (pos[u] - pos[v]).norm(dim=-1)
            edge_attr = self.distance_expansion(edge_weight)
            h[new_prev_idx] = h[new_prev_idx] + self.interactions[i](h[new_prev_idx], edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        return EncoderOutput(
            {
                "node_embedding": h,
                "graph_embedding": torch_scatter.scatter(
                    h, batch.batch, dim=0, reduce=self.readout
                ),
            }
        )


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils
    from graphein.protein.tensor.data import get_random_protein

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(
        root / "configs" / "encoder" / "schnet.yaml"
    )
    print(cfg)
    encoder = hydra.utils.instantiate(cfg.schnet)
    print(encoder)
    batch = ProteinBatch().from_protein_list(
        [get_random_protein() for _ in range(4)], follow_batch=["coords"]
    )
    batch.batch = batch.coords_batch
    batch.edges("knn_8", cache="edge_index")
    batch.pos = batch.coords[:, 1, :]
    batch.x = batch.residue_type
    print(batch)
    out = encoder.forward(batch)
    print(out)
