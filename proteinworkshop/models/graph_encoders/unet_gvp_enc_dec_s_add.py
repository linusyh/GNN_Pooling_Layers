from typing import Set, Union

import torch
import torch.nn.functional as F
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from torch_geometric.data import Batch

import proteinworkshop.models.graph_encoders.layers.gvp as gvp
from proteinworkshop.models.graph_encoders.components import blocks
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput
from torch_geometric.nn import fps, MLP, GINConv
from torch.nn import  Linear, ModuleList, ReLU, Module
from proteinworkshop.features.edges import compute_edges 
from torch_scatter import scatter_add
import graphein.protein.tensor.edges as gp
import functools
from torch_scatter import scatter_add
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


class UnetGVPGNNModel_Enc_Dec_S_Add(torch.nn.Module):
    def __init__(
        self,
        s_dim: int = 128,
        v_dim: int = 16,
        s_dim_edge: int = 32,
        v_dim_edge: int = 1,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        num_layers: int = 5,
        pool: str = "sum",
        residual: bool = True,
        fps_prob: float = 0.6,
    ):
        """
        Initializes an instance of the GVPGNNModel class with the provided
        parameters.

        :param s_dim: Dimension of the node state embeddings (default: ``128``)
        :type s_dim: int
        :param v_dim: Dimension of the node vector embeddings (default: ``16``)
        :type v_dim: int
        :param s_dim_edge: Dimension of the edge state embeddings
            (default: ``32``)
        :type s_dim_edge: int
        :param v_dim_edge: Dimension of the edge vector embeddings
            (default: ``1``)
        :type v_dim_edge: int
        :param r_max: Maximum distance for Bessel basis functions
            (default: ``10.0``)
        :type r_max: float
        :param num_bessel: Number of Bessel basis functions (default: ``8``)
        :type num_bessel: int
        :param num_polynomial_cutoff: Number of polynomial cutoff basis
            functions (default: ``5``)
        :type num_polynomial_cutoff: int
        :param num_layers: Number of layers in the model (default: ``5``)
        :type num_layers: int
        :param pool: Global pooling method to be used
            (default: ``"sum"``)
        :type pool: str
        :param residual: Whether to use residual connections
            (default: ``True``)
        :type residual: bool
        """
        super().__init__()
        _DEFAULT_V_DIM = (s_dim, v_dim)
        _DEFAULT_E_DIM = (s_dim_edge, v_dim_edge)
        self.r_max = r_max
        self.num_layers = num_layers
        self.fps_prob = fps_prob
        activations = (F.relu, None)
        
        num_up_downs = num_layers // 2

        # Node embedding
        self.emb_in = torch.nn.LazyLinear(s_dim)
        self.W_v = torch.nn.Sequential(
            gvp.LayerNorm((s_dim, 0)),
            gvp.GVP(
                (s_dim, 0),
                _DEFAULT_V_DIM,
                activations=(None, None),
                vector_gate=True,
            ),
        )
        # Edge embedding
        self.radial_embedding = blocks.RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        self.W_e_d = ModuleList()
        for _ in range(num_up_downs):
            w_e = torch.nn.Sequential(
                gvp.LayerNorm((self.radial_embedding.out_dim, 1)),
                gvp.GVP(
                    (self.radial_embedding.out_dim, 1),
                    _DEFAULT_E_DIM,
                    activations=(None, None),
                    vector_gate=True,
                ),)
            self.W_e_d.append(w_e)

        self.W_e_up = ModuleList()
        for _ in range(num_up_downs):
            w_e = torch.nn.Sequential(
                gvp.LayerNorm((self.radial_embedding.out_dim, 1)),
                gvp.GVP(
                    (self.radial_embedding.out_dim, 1),
                    _DEFAULT_E_DIM,
                    activations=(None, None),
                    vector_gate=True,
                ),)
            self.W_e_up.append(w_e)

        def _contruct_gvp_layer(v_dim=_DEFAULT_V_DIM, e_dim=_DEFAULT_E_DIM):
            return gvp.GVPConvLayer(
                v_dim,
                e_dim,
                activations=activations,
                vector_gate=True,
                residual=residual,
            )
            
        # Stack of GNN layers
        self.layers_d = torch.nn.ModuleList(
            _contruct_gvp_layer()
            for _ in range(num_layers)
        )
        self.layers_up = torch.nn.ModuleList(
            _contruct_gvp_layer()
            for _ in range(num_up_downs)
        )
        # Output GVP
        self.W_out = torch.nn.Sequential(
            gvp.LayerNorm(_DEFAULT_V_DIM),
            gvp.GVP(
                _DEFAULT_V_DIM,
                (s_dim, 0),
                activations=activations,
                vector_gate=True,
            ),
        )
        # Global pooling/readout function
        self.readout = get_aggregation(pool)

        self.reds = ModuleList()
        for _ in range(num_up_downs):
            mlp = MLP([s_dim, s_dim, s_dim], act='relu', norm=None)
            self.reds.append(GINConv(nn=mlp, train_eps=False))
        self.lin = ModuleList()

        self.lin_s = ModuleList()
        self.lin_v = ModuleList()
        for _ in range(2):
            self.lin_s.append(Linear(_DEFAULT_V_DIM[0], 
                                     _DEFAULT_V_DIM[0]))
            self.lin_v.append(Linear(_DEFAULT_V_DIM[1], 
                                     _DEFAULT_V_DIM[1]))
        """
        for _ in range(2):
            self.lin_s.append(Linear(_DEFAULT_V_DIM_UP[0], 
                                     _DEFAULT_V_DIM[0]))
            self.lin_v.append(Linear(_DEFAULT_V_DIM_UP[1], 
                                     _DEFAULT_V_DIM[1]))
        """

        self.act_lin = ReLU()

    @property
    def required_batch_attributes(self) -> Set[str]:
        """Required batch attributes for this encoder.

        - ``edge_index`` (shape ``[2, num_edges]``)
        - ``pos`` (shape ``[num_nodes, 3]``)
        - ``x`` (shape ``[num_nodes, num_node_features]``)
        - ``batch`` (shape ``[num_nodes]``)

        :return: _description_
        :rtype: Set[str]
        """
        return {"edge_index", "pos", "x", "batch"}

    @jaxtyped(typechecker=typechecker)
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the GVP-GNN encoder.

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
        # Edge features
        stack_down_idx = []
        stack_down_h_V = []
        stack_down_edges = []
        stack_down_batch = []
        stack_down_pos = []
        idx = torch.arange(batch.x.size(0), dtype=torch.long, device=batch.x.device)
        stack_down_idx.append(idx)
        vectors = (
            batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
        )  # [n_edges, 3]
        lengths = torch.linalg.norm(
            vectors, dim=-1, keepdim=True
        )  # [n_edges, 1]

        h_V = self.emb_in(batch.x)
        h_E = (
            self.radial_embedding(lengths),
            torch.nan_to_num(torch.div(vectors, lengths)).unsqueeze_(-2),
        )

        h_V = self.W_v(h_V)
        h_E = self.W_e_d[0](h_E)

        pos = batch.pos
        graphs = batch.batch
        edge_index = batch.edge_index
        for i, layer in enumerate(self.layers_d):
            if i % 2 == 1:
                stack_down_h_V.append(h_V)
                stack_down_batch.append(graphs)
                stack_down_pos.append(pos)
                stack_down_edges.append(edge_index)
                stack_down_idx.append(idx)
                idx = fps(pos, graphs, self.fps_prob)
                
                mask = torch.ones(len(pos), dtype=torch.bool)
                mask[idx] = False
                idx_neg = torch.arange(len(pos), device=idx.device)[mask]
                s = nearest(pos[idx_neg], pos[idx], graphs[idx_neg], graphs[idx])
                h_s = scatter_add(h_V[0][idx_neg], s, dim=0, dim_size=h_V[0].size(0), out = h_V[0][idx])
                h_v = scatter_add(h_V[1][idx_neg], s, dim=0, dim_size=h_V[1].size(0), out = h_V[1][idx])
                h_V = (h_s, h_v)
                pos = pos[idx]
                graphs = graphs[idx]

                edge_index, _ = compute_new_edges(pos, graphs, 'knn_16')

                # Edge features
                vectors = (
                    pos[edge_index[0]] - pos[edge_index[1]])  # [n_edges, 3]
                lengths = torch.linalg.norm(
                    vectors, dim=-1, keepdim=True)  # [n_edges, 1]
                h_E = (
                    self.radial_embedding(lengths),
                    torch.nan_to_num(torch.div(vectors, lengths)).unsqueeze_(-2),
                    )
                h_E = self.W_e_d[i//2-1](h_E)

            h_V = layer(h_V, edge_index, h_E)
        for i, layer in enumerate(self.layers_up):
            h_V_skip = stack_down_h_V.pop()
            h_s, h_v = h_V_skip
            h_s[idx] = h_s[idx] + h_V[0]
            h_v[idx] = h_v[idx] + h_V[1]
            h_V = (h_s, h_v)
            graphs = stack_down_batch.pop()
            pos = stack_down_pos.pop()
            idx = stack_down_idx.pop()
            edge_index = stack_down_edges.pop()
            vectors = (
                pos[edge_index[0]] - pos[edge_index[1]])  # [n_edges, 3]
            lengths = torch.linalg.norm(
                vectors, dim=-1, keepdim=True)  # [n_edges, 1]
            h_E = (
                self.radial_embedding(lengths),
                torch.nan_to_num(torch.div(vectors, lengths)).unsqueeze_(-2),
                )
            h_E = self.W_e_up[i](h_E)
            
            h_V = layer(h_V, edge_index, h_E)

        out = self.W_out(h_V)

        return EncoderOutput(
            {
                "node_embedding": out,
                "graph_embedding": self.readout(
                    out, batch.batch
                ),  # (n, d) -> (batch_size, d)
                # "pos": pos  # TODO it is possible to output pos with GVP if needed
            }
        )


if __name__ == "__main__":
    import hydra
    import omegaconf

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(
        constants.SRC_PATH / "config" / "encoder" / "gvp.yaml"
    )
    enc = hydra.utils.instantiate(cfg)
    print(enc)
