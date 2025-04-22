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
from torch_geometric.nn import fps, MLP, GINConv, nearest
from torch.nn import Linear, ModuleList, ReLU, Module
from proteinworkshop.features.edges import compute_edges
from torch_scatter import scatter_add
import graphein.protein.tensor.edges as gp
import functools

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

class UnifiedUnetGVPGNNModel(torch.nn.Module):
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
        pooling_strategy: str = "fps",  # "fps", "nearest", "both"
    ):
        super().__init__()
        _DEFAULT_V_DIM = (s_dim, v_dim)
        _DEFAULT_E_DIM = (s_dim_edge, v_dim_edge)
        self.r_max = r_max
        self.num_layers = num_layers
        self.pooling_strategy = pooling_strategy
        activations = (F.relu, None)

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
        self.W_e = ModuleList()
        for _ in range(num_layers // 2):
            w_e = torch.nn.Sequential(
                gvp.LayerNorm((self.radial_embedding.out_dim, 1)),
                gvp.GVP(
                    (self.radial_embedding.out_dim, 1),
                    _DEFAULT_E_DIM,
                    activations=(None, None),
                    vector_gate=True,
                ),
            )
            self.W_e.append(w_e)
        # Stack of GNN layers
        self.layers = torch.nn.ModuleList(
            gvp.GVPConvLayer(
                _DEFAULT_V_DIM,
                _DEFAULT_E_DIM,
                activations=activations,
                vector_gate=True,
                residual=residual,
            )
            for _ in range(num_layers)
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
        for _ in range(num_layers // 2 + 1):
            mlp = MLP([s_dim, s_dim, s_dim], act='relu', norm=None)
            self.reds.append(GINConv(nn=mlp, train_eps=False))

        self.lin_s = ModuleList()
        self.lin_v = ModuleList()
        self.act_lin = ReLU()

    @property
    def required_batch_attributes(self) -> Set[str]:
        return {"edge_index", "pos", "x", "batch"}

    @jaxtyped(typechecker=typechecker)
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
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
        h_E = self.W_e[0](h_E)

        for i, layer in enumerate(self.layers):
            if (i + 1) % 2 == 0:
                if self.pooling_strategy == "fps":
                    idx = fps(batch.pos, batch.batch, 0.6)
                elif self.pooling_strategy == "nearest":
                    mask = torch.ones(len(batch.pos), dtype=torch.bool)
                    mask[idx] = False
                    idx_neg = torch.arange(len(batch.pos), device=idx.device)[mask]
                    s = nearest(batch.pos[idx_neg], batch.pos[idx], batch.batch[idx_neg], batch.batch[idx])
                    h_s = scatter_add(h_V[0][idx_neg], s, dim=0, dim_size=h_V[0].size(0), out=h_V[0][idx])
                    h_v = scatter_add(h_V[1][idx_neg], s, dim=0, dim_size=h_V[1].size(0), out=h_V[1][idx])
                    h_V = (h_s, h_v)
                elif self.pooling_strategy == "both":
                    idx = fps(batch.pos, batch.batch, 0.6)
                    mask = torch.ones(len(batch.pos), dtype=torch.bool)
                    mask[idx] = False
                    idx_neg = torch.arange(len(batch.pos), device=idx.device)[mask]
                    s = nearest(batch.pos[idx_neg], batch.pos[idx], batch.batch[idx_neg], batch.batch[idx])
                    h_s = scatter_add(h_V[0][idx_neg], s, dim=0, dim_size=h_V[0].size(0), out=h_V[0][idx])
                    h_v = scatter_add(h_V[1][idx_neg], s, dim=0, dim_size=h_V[1].size(0), out=h_V[1][idx])
                    h_V = (h_s, h_v)

                batch.pos = batch.pos[idx]
                batch.x = batch.x[idx]
                batch.batch = batch.batch[idx]
                batch.edge_index, batch.edge_type = compute_edges(batch, ['knn_16'])

                vectors = (
                    batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
                )  # [n_edges, 3]
                lengths = torch.linalg.norm(
                    vectors, dim=-1, keepdim=True
                )  # [n_edges, 1]
                h_E = (
                    self.radial_embedding(lengths),
                    torch.nan_to_num(torch.div(vectors, lengths)).unsqueeze_(-2),
                )
                h_E = self.W_e[(i + 1) // 2 - 1](h_E)

            h_V = layer(h_V, batch.edge_index, h_E)

        out = self.W_out(h_V)

        return EncoderOutput(
            {
                "node_embedding": out,
                "graph_embedding": self.readout(
                    out, batch.batch
                ),
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
