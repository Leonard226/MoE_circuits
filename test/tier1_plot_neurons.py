"""Per-neuron version of panel (a): routing-influence variance by receiving layer
for a handful of named super-neurons, against a gray cloud of random neurons.
"""
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)
art = os.path.join(config["result_path"], "tier1_dynamic")

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

C_M1E9 = "#d62728"
C_M2E30 = "#2ca02c"
C_M4E14 = "#1f77b4"
C_M14E60 = "#ff7f0e"
C_RAND = "#cccccc"

T_dyn = torch.load(os.path.join(art, "T_dyn.pt"))  # [L_send, n_experts, d_ffn, L_recv]
N_LAYERS, N_EXPERTS, D_FFN, _ = T_dyn.shape

# Top dominant neuron of each named expert (from panels b/c).
NAMED_NEURONS = [
    ("M1E9, z=915",  (1,  9, 915), C_M1E9,  "o"),
    ("M2E30, z=742", (2, 30, 742), C_M2E30, "^"),
    ("M4E14, z=391", (4, 14, 391), C_M4E14, "s"),
    ("M14E60, z=357",(14, 60, 357),C_M14E60,"D"),
]
named_set = {coord for _, coord, _, _ in NAMED_NEURONS}

# Sample random non-super neurons for the gray cloud.
N_RAND = 500
rng = np.random.default_rng(42)
rand_neurons = []
while len(rand_neurons) < N_RAND:
    c = int(rng.integers(0, N_LAYERS - 1))
    j = int(rng.integers(0, N_EXPERTS))
    z = int(rng.integers(0, D_FFN))
    if (c, j, z) in named_set:
        continue
    if T_dyn[c, j, z, :].sum().item() <= 0:
        continue
    rand_neurons.append((c, j, z))

fig, ax = plt.subplots(figsize=(7.0, 4.0))

# Gray cloud of random neurons.
for (c, j, z) in rand_neurons:
    for l in range(c + 1, N_LAYERS):
        v = T_dyn[c, j, z, l].item()
        if v > 0:
            ax.scatter(l, v, color=C_RAND, alpha=0.25, s=5,
                       edgecolor="none", zorder=1)

# Named super-neurons.
for name, (c, j, z), col, marker in NAMED_NEURONS:
    ls = list(range(c + 1, N_LAYERS))
    vals = [T_dyn[c, j, z, l].item() for l in ls]
    ax.scatter(ls, vals, color=col, marker=marker,
               s=44, label=name, zorder=3, edgecolor=col, linewidth=0)

ax.scatter([], [], color=C_RAND, alpha=0.45, s=8, label=f"Other Neurons ({N_RAND} random)")

# Determine y-range from named neurons.
named_max = max(T_dyn[c, j, z, l].item()
                for _, (c, j, z), _, _ in NAMED_NEURONS
                for l in range(c + 1, N_LAYERS))

ax.set_xlabel("Receiving Layer")
ax.set_ylabel("Variance")
ax.set_title("Per-neuron routing influence by receiving layer")
ax.set_xticks([0, 5, 10, 15])
ax.set_xlim(-0.5, N_LAYERS - 0.5)
ax.set_ylim(-named_max * 0.04, named_max * 1.10)
ax.grid(True, alpha=0.3, lw=0.4)
ax.set_axisbelow(True)
ax.legend(loc="upper left", framealpha=0.95, fontsize=8.5)

plt.tight_layout()

out_pdf = os.path.join(art, "tier1_super_neurons_recv.pdf")
out_png = os.path.join(art, "tier1_super_neurons_recv.png")
fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
fig.savefig(out_png, format="png", bbox_inches="tight", dpi=180)
plt.close(fig)

print(f"Wrote {out_pdf}")
print(f"Wrote {out_png}")

# Also dump the per-neuron, per-recv-layer values for the named neurons (sanity / use in caption).
print()
print("Per-neuron variance by receiving layer  (named super-neurons):")
print(f"{'neuron':<14} " + " ".join(f"l={l:2d}" for l in range(N_LAYERS)))
for name, (c, j, z), _, _ in NAMED_NEURONS:
    row = []
    for l in range(N_LAYERS):
        v = T_dyn[c, j, z, l].item()
        row.append(f"{v:5.3f}" if l > c else "  -  ")
    print(f"{name:<14} " + " ".join(row))
