import matplotlib.pyplot as plt
import numpy as np

# Data preparation
ctx_len = [128, 256, 512, 1024, 2048]
mus = [32, 64, 128, 256]

# Latency data for KV Transfer, Attention, and MoE FFN
data_kv = {
    32: [0.00141, 0.00274, 0.00546, 0.0109, 0.0218],
    64: [0.00274, 0.00546, 0.01090, 0.02176, 0.04351],
    128: [0.00546, 0.01092, 0.02178, 0.04350, 0.08699],
    256: [0.01089, 0.02177, 0.04349, 0.08697, 0.17388]
}

data_attn = {
    32: [0.00044, 0.00067, 0.00173, 0.003, 0.00594],
    64: [0.00074, 0.00148, 0.00298, 0.00589, 0.0116],
    128: [0.00157, 0.00298, 0.00632, 0.0114, 0.0232],
    256: [0.00371, 0.00673, 0.01178, 0.02206, 0.04458]
}

data_moe = {
    32: 0.0122,
    64: 0.0123,
    128: 0.0125,
    256: 0.013
}

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.ravel()

# Selecting colorblind-friendly colors that are also distinct in grayscale
colors_kv = '#377eb8'  # Blue color
colors_attn = '#e41a1c'  # Red color

# Set font sizes
title_fontsize = 18
label_fontsize = 18
tick_fontsize = 14
legend_fontsize = 18  # Larger legend font size

for i, mu in enumerate(mus):
    ax = axs[i]
    spacing = range(len(ctx_len))
    ax.bar([x - 0.2 for x in spacing], data_kv[mu], width=0.4, color=colors_kv, label='KV Transfer' if i == 0 else "")
    ax.bar([x + 0.2 for x in spacing], data_attn[mu], width=0.4, color=colors_attn, label='CPU Attention' if i == 0 else "")
    ax.axhline(y=data_moe[mu], color='black', linestyle='--', label='MoE FFN' if i == 0 else "")

    ax.set_title(f'Micro Batch Size: {mu}', fontsize=title_fontsize, fontweight='bold')
    ax.set_xlabel('Context Length', fontsize=label_fontsize, fontweight='bold')
    ax.set_ylabel('Latency (seconds)', fontsize=label_fontsize, fontweight='bold')
    ax.set_xticks(spacing)
    ax.set_xticklabels(ctx_len, fontsize=tick_fontsize)  # Ticks not bold
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

# Set a single legend at the top with a larger font
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make space for legend
plt.savefig('ablation-2.pdf', dpi=400, bbox_inches='tight')
plt.show()
