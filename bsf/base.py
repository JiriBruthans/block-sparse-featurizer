"""Shared base for the three block-sparse featurizers.

A *concept* is a block of `group_size` latent dims; a featurizer carves the
`d`-dim activation space into `n_groups` such blocks. All three featurizers
share the same linear decoder -- stack the gated codes and multiply by the
decoder atoms (one (group_size, d) block per concept) -- so the only things a
featurizer defines are how it `encode`s an input into gated per-block codes and
what `loss` it optimises.
"""
import torch
import torch.nn as nn


def group_topk(group_norms, l0):
    """Per-sample block TopK (the projection Pi_l): in each row keep the `l0`
    blocks of largest norm and zero the rest. Returns a {0, 1} mask (N, G)."""
    idx = group_norms.topk(l0, dim=-1).indices
    return torch.zeros_like(group_norms).scatter_(1, idx, 1.0)


def unit_blocks(W, n_groups, group_size, eps=1e-8):
    """Scale each (group_size, d) decoder block to unit Frobenius norm."""
    Wb = W.reshape(n_groups, group_size, -1)
    norm = Wb.flatten(1).norm(dim=1).clamp_min(eps).reshape(n_groups, 1, 1)
    return (Wb / norm).reshape(W.shape)


class BSF(nn.Module):
    """Block-Sparse Featurizer.

    Subclasses implement `encode(x) -> (N, n_groups, group_size)` and
    `loss(x, target) -> (scalar, info)`. They expose their decoder either as a
    `W_dec` parameter (the default) or by overriding `decoder_atoms`.
    """

    def __init__(self, d, n_groups, group_size):
        super().__init__()
        self.d = d
        self.n_groups = n_groups
        self.group_size = group_size

    # provided by subclasses
    def encode(self, x):
        raise NotImplementedError

    def loss(self, x, target=None):
        raise NotImplementedError

    # (n_groups * group_size, d) decoder matrix
    def decoder_atoms(self):
        return self.W_dec

    def decode(self, z):
        return z.reshape(z.shape[0], -1) @ self.decoder_atoms()

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    # per-concept decoder atoms (n_groups, group_size, d) for the viz
    def atoms(self):
        return self.decoder_atoms().detach().reshape(
            self.n_groups, self.group_size, self.d)

    # unit-Frobenius the decoder blocks (no-op for an orthonormal decoder)
    @torch.no_grad()
    def normalize_decoder(self):
        if hasattr(self, 'W_dec'):
            self.W_dec.data.copy_(
                unit_blocks(self.W_dec.data, self.n_groups, self.group_size))
