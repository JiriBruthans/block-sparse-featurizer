# bsf — block-sparse featurizers

Three tiny featurizers that learn *concepts* in vision-model activations, where a
concept is a small group of latent dimensions rather than a single neuron. Demo'd
on DINOv3 patch activations of 300 rabbit images.

| featurizer | idea |
|---|---|
| `VanillaBSF` | free encoder, per-sample block TopK (keep the `l0` blocks of largest norm) |
| `GroupLassoBSF` | free encoder, thresholded block gate (signed codes); threshold `theta` learned by a straight-through estimator (block JumpReLU), L0 penalty for sparsity |
| `GrassmannianBSF` | tied orthonormal frame per concept + block TopK, learned scale `gamma` |

All three subclass `bsf.BSF` and share one interface (`encode` / `loss` /
decoder), so a **single trainer** (`bsf.train`) and a **single visualiser**
(`bsf.viz`) work for any of them. With `group_size=3` each concept is a 3D
subspace, which is what makes the manifold plots meaningful.

## Layout

```
bsf/
  base.py          abstract BSF (decode / forward / atoms / block ops)
  vanilla.py       VanillaBSF
  group_lasso.py   GroupLassoBSF
  grassmannian.py  GrassmannianBSF
  train.py         featurizer-agnostic trainer
  viz.py           concept manifolds (3D) + PCA->RGB overlays
  data.py          rabbit images -> DINOv3 activations
  pos_mean.npy     DINOv3 per-position mean over ImageNet (196, 768)
rabbit.npz         300 x 224 x 224 x 3 rabbit images
starters/
  01_grassmannian.ipynb
  02_group_lasso.ipynb
  03_vanilla.ipynb
make_notebooks.py  regenerates the three notebooks
```

## Use

```python
import bsf, numpy as np, torch, einops
from bsf import data

images = data.load_rabbit_images('rabbit.npz')
acts = data.dino_activations(images)                 # (300, 196, 768), needs a GPU

# centre by the ImageNet per-position mean, flatten, scale to ||x|| ~ sqrt(d)
acts = acts - data.POS_MEAN
x = einops.rearrange(acts, 'n p d -> (n p) d')
x = x / np.sqrt((x ** 2).sum(1).mean()) * np.sqrt(x.shape[1])
grid = data.patch_grid(acts.shape[1])                # 14

model = bsf.VanillaBSF(d=768, n_groups=256, group_size=3, l0=16)
bsf.train(model, x, epochs=60)                       # snr=0.1 denoising by default

# encode every patch, rank concepts by energy, plot
z = model.encode(torch.as_tensor(x, dtype=torch.float32, device='cuda')).detach().cpu().numpy()
atoms = model.atoms().cpu().numpy()
heat = np.linalg.norm(z, axis=-1)
fire, energy = (heat > 1e-6).sum(0), (heat ** 2).sum(0)
top = [g for g in np.argsort(-energy) if fire[g] >= 100][:20]
bsf.viz.plot_concepts(z, atoms, images, top, grid, n_img=10)
```

Each notebook in `starters/` runs this end-to-end for one featurizer. They locate
the repo root automatically, so they work from anywhere.

## Requirements

`torch`, `numpy`, `scikit-learn`, `matplotlib`, `einops`, `transformers` (DINOv3).
Computing activations needs a GPU; training and plotting run on CPU.
