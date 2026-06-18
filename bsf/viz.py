"""Plotting the learned concepts.

`plot_concepts` draws one band per concept: on the left its firing cloud as a 3D
manifold (PCA of the contributions m_i, coloured by PCA coords -> RGB, point
size = norm, halo + shadowed square floor), and on the right a grid of the
images it fires hardest on -- each patch tinted by its PCA coordinate (hue =
where on the manifold it lies) with alpha = its norm.

The codes `z` (N, G, K) and decoder atoms (G, K, d) are passed in directly:
compute them in the notebook with `model.encode(x)` and `model.atoms()`.
"""
import numpy as np
import einops
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# registers the 3d projection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _pca_fit(c, k=3):
    """Mean and top-`k` principal axes of the contribution cloud `c` (n, d)."""
    mean = c.mean(0)
    comps = np.linalg.svd(c - mean, full_matrices=False)[2][:k]
    return mean, comps


def _make_colorize(proj, per_axis, clip=(0.0, 99.0), saturation=1.0):
    """Return a function mapping PCA coordinates -> RGB in [0, 1].

    per_axis=True : clip each axis to its `clip` percentiles (default min/99th),
                    then min/max-normalise per axis. `saturation` pushes colours
                    away from gray (1.0 = none).
    per_axis=False: one global scale centred at 0 (gray = zero).
    """
    if per_axis:
        lo = np.percentile(proj, clip[0], axis=0)
        hi = np.percentile(proj, clip[1], axis=0)

        def f(p):
            rgb = ((p - lo) / np.maximum(hi - lo, 1e-8)).clip(0, 1)
            return (0.5 + (rgb - 0.5) * saturation).clip(0, 1)
        return f
    scale = float(np.abs(proj).max()) + 1e-8
    return lambda p: (0.5 + 0.5 * p / scale * saturation).clip(0, 1)


def _manifold_ax(ax, proj, rgb, norm):
    """3D scatter: square floor below the cloud + shadow, no axes, halo behind
    each point, point size proportional to norm."""
    # small points, size = norm
    n = norm / max(norm.max(), 1e-8)
    s = 0.2 + 2.5 * n
    # floor clearly below the lowest point (margin scaled to the cloud size)
    span = np.ptp(proj, axis=0).max()
    floor = proj[:, 2].min() - 0.3 * span

    # square floor plane centred on the cloud
    cx, cy = proj[:, 0].mean(), proj[:, 1].mean()
    r = max(np.ptp(proj[:, 0]), np.ptp(proj[:, 1])) / 2 * 1.05 + 1e-8
    xx, yy = np.meshgrid([cx - r, cx + r], [cy - r, cy + r])
    ax.plot_surface(xx, yy, np.full_like(xx, floor), color='0.93', alpha=0.6,
                    shade=False, zorder=0)
    # shadow on the floor
    ax.scatter(proj[:, 0], proj[:, 1], np.full(len(proj), floor),
               c='0.35', s=s, alpha=0.05, edgecolors='none', depthshade=False)
    # halo: a 2x larger translucent twin of each point
    ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=rgb, s=s * 2.0, alpha=0.2,
               edgecolors='none', depthshade=False)
    # the points
    ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=rgb, s=s, alpha=1.0,
               edgecolors='none', depthshade=False)
    try:
        ax.axis('equal')
    except Exception:
        pass
    ax.view_init(elev=16, azim=-60)
    ax.set_axis_off()


def _overlay(ax, image, z_patch, atoms_g, mean, comps, colorize, grid):
    """Tint each patch by the PCA->RGB of its contribution; alpha = its norm."""
    # (P, d)
    c = z_patch @ atoms_g
    # (P, 3)
    proj = (c - mean) @ comps.T
    rgb = colorize(proj)
    norm = np.linalg.norm(c, axis=1)
    alpha = (norm / max(norm.max(), 1e-8)).clip(0, 1)
    rgba = np.concatenate([rgb, alpha[:, None]], 1).reshape(grid, grid, 4)
    h, w = image.shape[:2]
    ax.imshow(image)
    ax.imshow(rgba, extent=(0, w, h, 0), interpolation='bicubic')
    ax.set_xticks([]); ax.set_yticks([])


def plot_concepts(z, atoms, images, concepts, grid, n_img=10, ncol_img=5,
                  per_axis_rgb=True, clip=(0.0, 99.0), saturation=1.0,
                  drop_low_norm=0.0, max_points=5000, seed=0):
    """One band per concept: 3D manifold (left) + an `n_img` grid of overlays.

    z       (N, G, K)  gated codes (model.encode(x))
    atoms   (G, K, d)  decoder atoms (model.atoms())
    images  (n_imgs, H, W, 3)
    concepts           iterable of group indices to show
    drop_low_norm      fraction of lowest-norm firing points to discard before
                       fitting/plotting (cleans the gray center).
    """
    P = grid * grid
    # (N, G)
    heat = np.linalg.norm(z, axis=-1)
    zr = einops.rearrange(z, '(n p) g k -> n p g k', p=P)
    heat_img = einops.rearrange(heat, '(n p) g -> n p g', p=P)
    rng = np.random.default_rng(seed)

    # image rows per concept
    nrow_img = int(np.ceil(n_img / ncol_img))
    # manifold spans 2 columns
    mcol = 2
    ncols = mcol + ncol_img
    gs = GridSpec(nrow_img * len(concepts), ncols)
    fig = plt.figure(figsize=(1.7 * ncols, 1.7 * nrow_img * len(concepts)))

    for r, g in enumerate(concepts):
        r0 = r * nrow_img
        ax = fig.add_subplot(gs[r0:r0 + nrow_img, 0:mcol], projection='3d')
        idx = np.where(heat[:, g] > 1e-6)[0]
        # drop the lowest-norm firing points
        if drop_low_norm > 0 and idx.size > 8:
            keep = heat[idx, g] >= np.quantile(heat[idx, g], drop_low_norm)
            idx = idx[keep]
        if idx.size < 8:
            ax.set_axis_off()
            continue
        sub = idx if idx.size <= max_points else rng.choice(idx, max_points, replace=False)
        # (n, d) contributions, then PCA-project to 3D
        c = z[sub, g, :] @ atoms[g]
        mean, comps = _pca_fit(c, 3)
        proj = (c - mean) @ comps.T
        colorize = _make_colorize(proj, per_axis_rgb, clip, saturation)
        _manifold_ax(ax, proj, colorize(proj), np.linalg.norm(c, axis=1))

        top_imgs = np.argsort(-(heat_img[:, :, g] ** 2).sum(1))[:n_img]
        for j, ii in enumerate(top_imgs):
            ax2 = fig.add_subplot(gs[r0 + j // ncol_img, mcol + j % ncol_img])
            _overlay(ax2, images[ii], zr[ii, :, g, :], atoms[g], mean, comps, colorize, grid)

    fig.tight_layout()
    return fig
