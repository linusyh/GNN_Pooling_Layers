from typing import Optional, Set, Union

import torch
import torch_scatter
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch
from torch_geometric.nn.models import SchNet
from torch.nn import  Linear, ModuleList, ReLU, Module
from proteinworkshop.types import EncoderOutput
from torch_geometric.nn import fps, MLP, GINConv

from proteinworkshop.features.edges import compute_edges 


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

        self.reds = ModuleList()
        for _ in range(num_layers//2 - 1):
            mlp = MLP([hidden_channels, hidden_channels, hidden_channels], act='relu', norm=None)
            self.reds.append(GINConv(nn=mlp, train_eps=False))

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
        h = self.embedding(batch.x)
        u, v = batch.edge_index
        edge_weight = (batch.pos[u] - batch.pos[v]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)
        h = h + self.interactions[0](h, batch.edge_index, edge_weight, edge_attr)
        #print("******* ======== ******** ======== ********")
        #print("******* h.shape: ", h.shape)
        #print("*** batch.shape: ", batch.batch.shape)
        Hs = [None] * len(self.interactions)
        Bs = [None] * len(self.interactions)
        # Initialize Bs and Hs with Struct elements filled with default (empty) tensors

        Hs[0] = h
        Bs[0] = batch
        #print("batch.batch: ", batch.batch[0])

        for i in range(1, len(self.interactions)):
            #print("***** within loop: i =", i)
            if i % 2 == 0:
                #print("**** i-1 =", i-1)
                #print("Bs[",i-1,"].pos.shape:", Bs[i-1].pos.shape)
                idx = fps(Bs[i-1].pos, Bs[i-1].batch, 0.6) #fps(batch.pos, batch.batch, 0.6)
                #ADD AGGREGATION
                #h[idx] = self.reds[i//2-1](h, batch.edge_index)[idx]
                Hs[i] = self.reds[i//2-1](Hs[i-1], Bs[i-1].edge_index)[idx]
                #print("Bs[",i-1,"].pos.shape:", Bs[i-1].pos.shape)
                #print("idx.shape:", idx.shape)
                Bs[i] = Bs[i-1].clone()
                Bs[i].pos = Bs[i-1].pos[idx]
                Bs[i].x = Bs[i-1].x[idx]
                Bs[i].batch = Bs[i-1].batch[idx]
                Bs[i].edge_index, Bs[i].edge_type = compute_edges(Bs[i], ['knn_16'])
                u, v = Bs[i].edge_index
                edge_weight = (Bs[i].pos[u] - Bs[i].pos[v]).norm(dim=-1)
                edge_attr = self.distance_expansion(edge_weight)
                Hs[i] = Hs[i] + self.interactions[i](Hs[i], Bs[i].edge_index, edge_weight, edge_attr)
            else:
                #print("**** i-1 =", i-1)
                Bs[i] = Bs[i-1].clone()
                Hs[i] = Hs[i-1] + self.interactions[i](Hs[i-1], Bs[i-1].edge_index, edge_weight, edge_attr)
            #print("***** within the loop: Bs[",i,"].shape:", Bs[i].batch.shape, "Hs[",i,"].shape:", Hs[i].shape)

            #h = h + self.interactions[i](h, Bs[i].edge_index, edge_weight, edge_attr)

            '''
            if i % 2 == 0:
                idx = fps(batch.pos, batch.batch, 0.6) #fps(batch.pos, batch.batch, 0.6)
                #ADD AGGREGATION
                #h[idx] = self.reds[i//2-1](h, batch.edge_index)[idx]
                h = self.reds[i//2-1](h, batch.edge_index)[idx]
                batch.pos = batch.pos[idx]
                batch.x = batch.x[idx]
                batch.batch = batch.batch[idx]
                batch.edge_index, batch.edge_type = compute_edges(batch, ['knn_16'])
                u, v = batch.edge_index
                edge_weight = (batch.pos[u] - batch.pos[v]).norm(dim=-1)
                edge_attr = self.distance_expansion(edge_weight)
                #h = h[idx]
            h = h + self.interactions[i](h, batch.edge_index, edge_weight, edge_attr)
            Hs.append(h)
            Bs.append(batch.clone())
            #print("within loop, i=", i, ", h.shape: ", h.shape)
            #print("within loop, i=", i, ", batch.shape: ", batch.batch.shape)
            '''

        #print("******* after loop, h.shape: ", h.shape)
        h = torch.nn.Parameter(torch.cat(Hs, dim=0))
        # batch concat
        bs = []
        for i in range(len(Bs)):
            bs.append(Bs[i].batch)
            #print("Bs[",i,"].shape: ", Bs[i].batch.shape)
        #print("******** bs: ", bs)
        batches = torch.cat(bs, dim=0)

        #print("after concat, h.shape: ", h.shape)
        #print("after concat, batch.shape: ", bs.shape)
        h = self.lin1(h)
        h = self.act(h)
        #print("******* after activation, h.shape: ", h.shape)
        h = self.lin2(h)
        #print("******* after lin2, h.shape: ", h.shape)

        return EncoderOutput(
            {
                "node_embedding": h,
                "graph_embedding": torch_scatter.scatter(
                    h, batches, dim=0, reduce=self.readout
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
