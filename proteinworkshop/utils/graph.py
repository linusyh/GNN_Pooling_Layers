import torch
import functools
import graphein.protein.tensor.edges as gp


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
