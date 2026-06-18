"""Group-lasso BSF (soft-threshold slot, learned threshold).

A free linear encoder produces signed per-block codes (no ReLU). A block fires
when its L2 norm clears a per-block threshold theta, and the full *signed* code
is kept -- a threshold on the block norm, like the group-lasso soft-threshold,
but hard (no magnitude shrinkage, which collapses under a norm-constrained
decoder). theta is learned by a straight-through estimator (the block analogue
of JumpReLU): the hard gate has no gradient w.r.t. theta, so a rectangle-kernel
pseudo-derivative carries it. `gain` gives theta a fast effective learning rate
under the shared single-LR trainer, and an L0 (active-block) penalty `coef` sets
the sparsity level. theta is initialised from the first training batch so that
~`target_l0` blocks fire.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BSF, unit_blocks


class SoftThreshold(torch.autograd.Function):
    """Hard block gate H(||a_g|| - theta) with a straight-through pseudo-derivative
    for theta (rectangle kernel). Passes no gradient to ||a_g|| -- the encoder
    learns through the magnitude path z = a * gate."""

    @staticmethod
    def forward(ctx, gn, theta, bandwidth):
        ctx.save_for_backward(gn, theta)
        ctx.bw = float(bandwidth)
        return (gn > theta).to(gn.dtype)

    @staticmethod
    def backward(ctx, g):
        gn, theta = ctx.saved_tensors
        K = ((gn - theta).abs() <= ctx.bw / 2).to(gn.dtype) / ctx.bw
        return None, -(g * K).sum(0), None


class GroupLassoBSF(BSF):
    def __init__(self, d, n_groups, group_size=3, coef=1e-2, target_l0=16, gain=10.0):
        super().__init__(d, n_groups, group_size)
        self.coef = coef
        self.target_l0 = target_l0
        self.gain = gain
        W = unit_blocks(torch.randn(n_groups * group_size, d), n_groups, group_size)
        self.W_dec = nn.Parameter(W)
        # tied init
        self.W_enc = nn.Parameter(W.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(n_groups * group_size))
        self.raw_theta = nn.Parameter(torch.zeros(n_groups))
        self.register_buffer('bandwidth', torch.ones(()))
        self.register_buffer('inited', torch.zeros((), dtype=torch.bool))

    def theta(self):
        return F.softplus(self.gain * self.raw_theta)

    def _preact(self, x):
        return (x @ self.W_enc + self.b_enc).reshape(-1, self.n_groups, self.group_size)

    @torch.no_grad()
    def _init_theta(self, gn):
        """Place theta in the group-norm distribution so ~target_l0 blocks fire
        (the JumpReLU cold-start fix: theta must sit inside the STE kernel)."""
        q = 1.0 - self.target_l0 / self.n_groups
        thr = torch.quantile(gn.flatten(), q).clamp_min(1e-3)
        # undo gain
        self.raw_theta.copy_(torch.log(torch.expm1(thr)) / self.gain)
        self.bandwidth.copy_(gn.std() * 0.5 + 1e-6)
        self.inited.fill_(True)

    def _gate(self, gn):
        if self.training and not bool(self.inited):
            self._init_theta(gn)
        return SoftThreshold.apply(gn, self.theta(), float(self.bandwidth))

    def encode(self, x):
        a = self._preact(x)
        return a * self._gate(a.norm(dim=-1)).unsqueeze(-1)

    def loss(self, x, target=None):
        # target != x -> denoising
        target = x if target is None else target
        a = self._preact(x)
        gate = self._gate(a.norm(dim=-1))
        z = a * gate.unsqueeze(-1)
        recon = (target - self.decode(z)).pow(2).mean()
        # active blocks / token
        l0 = gate.sum(-1).mean()
        return recon + self.coef * l0, {'recon': recon.item(), 'l0': l0.item()}
