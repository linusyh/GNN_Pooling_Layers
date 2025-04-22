from typing import Optional, Set, Union

import torch
import torch_scatter
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch
from torch_geometric.nn.models import SchNet
from torch_geometric.nn.models.schnet import InteractionBlock
from torch.nn import  Linear, ModuleList, Module, ReLU
from proteinworkshop.types import EncoderOutput
from torch_geometric.nn import fps, MLP, GINConv
from torch_geometric.nn.pool import nearest
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


class UnetSchNetModelEncDecAdd(SchNet):
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
        fps_prob: float = 0.6,
        sparse: bool = False,
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
            num_layers,
            num_gaussians,
            cutoff,  # None, # Interaction graph is not used
            max_num_neighbors,
            readout,
            dipole,
            mean,
            std,
            atomref,
        )
        
        num_up_downs = num_layers // 2
        self.fps_prob = fps_prob
        self.sparse = sparse
        
        self.readout = readout
        # Overwrite embbeding
        self.embedding = torch.nn.LazyLinear(hidden_channels)
        # Overwrite atom embedding and final predictor
        self.lin2 = torch.nn.LazyLinear(out_dim)

        self.reds = ModuleList(
            SimpleMLP(hidden_channels=hidden_channels)
            for _ in range(num_up_downs)
        )
            
        self.layers_up = torch.nn.ModuleList(
            InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            for _ in range(num_up_downs)
        )
        
        self.lin_transformations = ModuleList(
            Linear(hidden_channels, hidden_channels)
            for _ in range(num_up_downs)
        )
        
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
        # prev_idx = torch.arange(batch.x.size(0), dtype=torch.long, device=batch.x.device)
        # new_prev_idx = prev_idx
        # stack_down_idx = []
        # stack_down_edges = []
        # h = self.embedding(batch.x)
        # u, v = batch.edge_index
        # edge_index = batch.edge_index
        # edge_weight = (batch.pos[u] - batch.pos[v]).norm(dim=-1)
        # edge_attr = self.distance_expansion(edge_weight)
        # h = h + self.interactions[0](h, batch.edge_index, edge_weight, edge_attr)
        h = self.embedding(batch.x)
        stack_down_edges = []
        stack_down_h = []
        stack_down_batch = []
        stack_down_pos = []
        stack_down_idx = []
        
        idx = torch.arange(batch.x.size(0), dtype=torch.long, device=batch.x.device)
        pos = batch.pos
        graphs = batch.batch
        edge_index = batch.edge_index
        for i in range(len(self.interactions)):
            if i % 2 == 0:
                stack_down_edges.append(edge_index)
                stack_down_h.append(h)
                stack_down_batch.append(graphs)
                stack_down_pos.append(pos)
                stack_down_idx.append(idx)
                idx = fps(pos, graphs, self.fps_prob)
                
                row, col = edge_index
                if self.sparse:
                    mask = torch.ones(len(pos), dtype=torch.bool, device=idx.device)
                    mask[idx] = False
                    s = nearest(pos[mask], pos[idx], graphs[mask], graphs[idx])
                    h = torch_scatter.scatter_add(h[mask], s, dim=0, dim_size=h.size(0), out = h[idx])
                else:
                    # node_features_aggregated = (
                    #     torch_scatter.scatter_add(h[row], col, dim=0, dim_size=h.size(0)) + 
                    #     torch_scatter.scatter_add(h[col], row, dim=0, dim_size=h.size(0))
                    # )[idx]
                    node_features_aggregated = torch_scatter.scatter_add(h[row], col, dim=0, dim_size=h.size(0))[idx]
                    h = h[idx] + self.reds[i//2](node_features_aggregated)
                pos = pos[idx]
                graphs = graphs[idx]
                
                edge_index, _ = compute_new_edges(pos, graphs, 'knn_16')
                u, v = edge_index
                edge_weight = (pos[u] - pos[v]).norm(dim=-1)
                edge_attr = self.distance_expansion(edge_weight)

            h = h + self.interactions[i](h, edge_index, edge_weight, edge_attr)
            
            if i % 2 == 0:
                h = self.lin_transformations[i//2](h)
                h = self.act(h)

        for i, layer in enumerate(self.layers_up):
            # idx = stack_down_idx.pop()
            # edge_index = stack_down_edges.pop()
            # pos = batch.pos[new_prev_idx]
            # u, v = edge_index
            # edge_weight = (pos[u] - pos[v]).norm(dim=-1)
            # edge_attr = self.distance_expansion(edge_weight)
            # h[new_prev_idx] = h[new_prev_idx] + self.interactions[i](h[new_prev_idx], edge_index, edge_weight, edge_attr)
            h_new = stack_down_h.pop()
            h_new[idx] += h
            graphs = stack_down_batch.pop()
            pos = stack_down_pos.pop()
            idx = stack_down_idx.pop()
            edge_index = stack_down_edges.pop()
            
            u, v = edge_index
            edge_weight = (pos[u] - pos[v]).norm(dim=-1)
            edge_attr = self.distance_expansion(edge_weight)
            h = h_new + layer(h_new, edge_index, edge_weight, edge_attr)
        
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
