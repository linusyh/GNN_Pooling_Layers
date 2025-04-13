from typing import Optional, Set, Union

import torch
import torch_scatter
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch
from torch_geometric.nn.models import SchNet
from torch.nn import  Linear, ModuleList, ReLU, MaxUnpool1d, Upsample
from proteinworkshop.types import EncoderOutput
from torch_geometric.nn import fps, MLP, GINConv
import graphein.protein.tensor.edges as gp
import functools
from proteinworkshop.features.edges import compute_edges 
from torch_geometric.nn.models.schnet import InteractionBlock
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

class UnetSchNetModelConcat(SchNet):
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
        self.readout = readout
        # Overwrite embbeding
        self.embedding = torch.nn.LazyLinear(hidden_channels)
        # Overwrite atom embedding and final predictor
        self.lin2 = torch.nn.LazyLinear(out_dim)
        self.num_layers = num_layers

        
        self.up_interactions = ModuleList()
        for _ in range(num_layers-2):
            #block = InteractionBlock(hidden_channels*2, num_gaussians,
            #                         num_filters, cutoff)
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.up_interactions.append(block)

        
        '''
        self.lin = ModuleList()
        for _ in range(num_layers-2):
            self.lin.append(Linear(2*hidden_channels, hidden_channels))

        self.act_lin = ReLU()
        '''

        #self.unpool = MaxUnpool1d(kernel_size=2, stride=2)
        #self.upsample = Upsample(scale_factor=2, mode="nearest")

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
        idx = torch.arange(batch.x.size(0), dtype=torch.long, device=batch.x.device)
        stack_down_idx.append(idx)
        h = self.embedding(batch.x)
        u, v = batch.edge_index
        edge_index = batch.edge_index
        edge_weight = (batch.pos[u] - batch.pos[v]).norm(dim=-1)
        pos = batch.pos
        graphs = batch.batch
        edge_attr = self.distance_expansion(edge_weight)
        h = h + self.interactions[0](h, batch.edge_index, edge_weight, edge_attr)
        for i in range(1,len(self.interactions)):
            if i % 2 == 0:
                stack_down_edges.append(edge_index)
                stack_down_h.append(h)
                stack_down_batch.append(graphs)
                stack_down_pos.append(pos)
                idx = fps(pos, graphs, 0.6)
                stack_down_idx.append(idx)
                mask = torch.ones(len(pos), dtype=torch.bool)
                mask[idx] = False
                idx_neg = torch.arange(len(pos), device=idx.device)[mask]
                s = nearest(pos[idx_neg], pos[idx], graphs[idx_neg], graphs[idx])
                h = torch_scatter.scatter_add(h[idx_neg], s, dim=0, dim_size=h.size(0), out = h[idx])
                pos = pos[idx]
                graphs = graphs[idx]
                edge_index, _ = compute_new_edges(pos, graphs, 'knn_16')
                u, v = edge_index
                edge_weight = (pos[u] - pos[v]).norm(dim=-1)
                edge_attr = self.distance_expansion(edge_weight)

            h = h + self.interactions[i](h, edge_index, edge_weight, edge_attr)
            
        for i in range(0,self.num_layers-2):
            if i % 2 == 0:
                idx = stack_down_idx.pop()
                pos = stack_down_pos.pop()
                graphs = stack_down_batch.pop()
                h_skip = stack_down_h.pop()
                edge_index = stack_down_edges.pop()

                u, v = edge_index
                edge_weight = (pos[u] - pos[v]).norm(dim=-1)
                edge_attr = self.distance_expansion(edge_weight)
                h_zero = torch.zeros(h_skip.shape[0] - h.shape[0], self.hidden_channels, device=h.device)
                mask = torch.ones(h_skip.shape[0], dtype=torch.bool, device=h.device)
                mask[idx] = False

                h_new = torch.zeros(h_skip.shape[0], self.hidden_channels, device=h.device)
                h_new[idx] = torch.add(h, h_skip[idx])
                h_new[mask] = torch.add(h_zero, h_skip[mask])
                
                h = h_new

            h = h + self.up_interactions[i](h, edge_index, edge_weight, edge_attr)


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
