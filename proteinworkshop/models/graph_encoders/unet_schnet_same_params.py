from typing import Optional, Set, Union

import torch
import torch_scatter
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch
from torch_geometric.nn.models import SchNet
from proteinworkshop.types import EncoderOutput
from torch_geometric.nn import fps
import graphein.protein.tensor.edges as gp
import functools
from proteinworkshop.models.utils import get_aggregation
from torch_geometric.nn.pool import nearest

def compute_new_edges(pos, graphs, edge_type):
    edges = []
    edge_fn = functools.partial(gp.compute_edges, batch=graphs)
    edges.append(edge_fn(pos, edge_type))
    indxs = torch.cat(
        [
            torch.ones_like(e_idx[0, :]) * idx
            for idx, e_idx in enumerate(edges)
        ],
        dim=0,
    ).unsqueeze(0)
    edges = torch.cat(edges, dim=1)

    return edges, indxs

class UnetSchNetModelSameParams(SchNet):
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
        self.readout = get_aggregation(readout)
        # Overwrite embbeding
        self.embedding = torch.nn.LazyLinear(hidden_channels)
        # Overwrite atom embedding and final predictor
        self.lin2 = torch.nn.LazyLinear(out_dim)
        self.num_layers = num_layers
        self.fps_prob = fps_prob
        self.sparse = sparse
        self.num_ups = len(self.interactions) // 2
        # self.concat_merging_layers = torch.nn.ModuleList(
        #     torch.nn.Linear(2*hidden_channels, hidden_channels) for _ in range(self.num_ups)
        # )
        # self.concat_merging_layer = torch.nn.Linear(2*hidden_channels, hidden_channels, bias=False)

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
        stack_down_idx = []
        stack_down_edges = []
        stack_down_h = []
        stack_down_batch = []
        stack_down_pos = []
        
        h = self.embedding(batch.x)
        edge_index = batch.edge_index
        graphs = batch.batch
        pos = batch.pos
        idx = torch.arange(batch.x.size(0), dtype=torch.long, device=batch.x.device)

        for i in range(self.num_ups):
            stack_down_edges.append(edge_index)
            stack_down_h.append(h)
            stack_down_batch.append(graphs)
            stack_down_pos.append(pos)
            
            # Pooling: node selection
            idx = fps(pos, graphs, self.fps_prob)
            stack_down_idx.append(idx)
            mask = torch.ones(len(pos), dtype=torch.bool, device=idx.device)
            mask[idx] = False
            # Pooling
            if self.sparse:
                s = nearest(pos[mask], pos[idx], graphs[mask], graphs[idx])
                h = torch_scatter.scatter_add(h[mask], s, dim=0, dim_size=h.size(0), out = h[idx])
            else:
                row, col = edge_index
                h = torch_scatter.scatter_add(h[row], col, dim=0, dim_size=h.size(0), out=h)[idx]
            # Keep only pooled nodes
            pos = pos[idx]
            graphs = graphs[idx]
            edge_index, _ = compute_new_edges(pos, graphs, 'knn_16')
            u, v = edge_index
            edge_weight = (pos[u] - pos[v]).norm(dim=-1)
            edge_attr = self.distance_expansion(edge_weight)

            h = h + self.interactions[i](h, edge_index, edge_weight, edge_attr)
            
        for i in range(self.num_ups, 2*self.num_ups):
            idx = stack_down_idx.pop()
            pos = stack_down_pos.pop()
            graphs = stack_down_batch.pop()
            h_skip = stack_down_h.pop()
            edge_index = stack_down_edges.pop()

            u, v = edge_index
            edge_weight = (pos[u] - pos[v]).norm(dim=-1)
            edge_attr = self.distance_expansion(edge_weight)
            mask = torch.ones(h_skip.shape[0], dtype=torch.bool, device=h.device)
            mask[idx] = False

            h_new = h_skip  #  Adding skip connect
            h_new[idx] += h  #  
            h = h_new + self.interactions[i](h_new, edge_index, edge_weight, edge_attr)

        if self.num_ups % 2 == 1:
            h = h + self.interactions[-1](h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        return EncoderOutput(
            {
                "node_embedding": h,
                "graph_embedding": self.readout(
                    h, batch.batch
                )
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
