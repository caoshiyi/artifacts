import numpy as np
import matplotlib.pyplot as plt
from utils import attention_flops, attention_bytes
from scipy.interpolate import interp1d

label_fontsize = 20
tick_fontsize = 16
legend_fontsize = 14  # Larger legend font size
anno_fontsize = 18

normal_linewidth = 3

# System specifications
cpu_mem_bandwidth = 100  # GB/s
gpu_mem_bandwidth = 300  # GB/s
cpu_gpu_mem_bandwidth = 16  # GB/s
cpu_peak_flops = 1.3  # TFLOPS/s
gpu_peak_flops = 242  # TFLOPS/s
gpu_mem = 24  # GB
cpu_mem = 192  # GB

# Convert TFLOPS/s to GFLOPS/s for plotting
cpu_peak_flops *= 1000
gpu_peak_flops *= 1000

# Define the range of operational intensity (FLOP/Byte)
oi = np.logspace(-1, 4, 500)  # from 0.1 to 1000 FLOP/Byte

# Compute ceilings
cpu_ceiling = np.minimum(cpu_mem_bandwidth * oi, cpu_peak_flops)
gpu_ceiling = np.minimum(gpu_mem_bandwidth * oi, gpu_peak_flops)
cpu_gpu_ceiling = np.minimum(cpu_gpu_mem_bandwidth * oi, gpu_peak_flops)

# Calculate intersection point
oi_intersection = cpu_peak_flops / cpu_gpu_mem_bandwidth

# attention
num_q_head = 32
num_kv_head = 8
head_dim = 128
seq_len = [512]
batch_size = 1
data_types = ['int4', 'f16']
attn_ois = [[attention_flops(batch_size, sl, num_q_head, head_dim) / attention_bytes(batch_size, sl, num_q_head, num_kv_head, head_dim, data_type) for sl in seq_len]
            for data_type in data_types]
attn_ideal_performance_values = [[min(gpu_mem_bandwidth * oi, gpu_peak_flops) for oi in attn_oi] for attn_oi in attn_ois]

# Setup plot
plt.figure(figsize=(10, 7))
plt.loglog(oi, cpu_ceiling, label='CPU Mem Bdw', linestyle='-', color='blue', linewidth=normal_linewidth)
plt.loglog(oi, gpu_ceiling, label='GPU Mem Bdw', linestyle='-', color='green', linewidth=normal_linewidth)
plt.loglog(oi, cpu_gpu_ceiling, label='CPU-GPU Mem Bdw', linestyle='-', color='red', linewidth=normal_linewidth)

# Add horizontal lines for peak FLOPS
plt.axhline(y=cpu_peak_flops, color='blue', linestyle='--', label='CPU Peak FLOPS', linewidth=normal_linewidth)
plt.axhline(y=gpu_peak_flops, color='green', linestyle='--', label='GPU Peak FLOPS', linewidth=normal_linewidth)

plt.vlines(x=oi_intersection, ymin=10, ymax=cpu_peak_flops, color='grey', linestyle='--', linewidth=normal_linewidth)
plt.scatter(oi_intersection, cpu_peak_flops, color='red', zorder=5, s=100)
plt.annotate('P1',
             xy=(oi_intersection, cpu_peak_flops), 
             xytext=(oi_intersection * 0.9, cpu_peak_flops * 1.6),
             horizontalalignment='right',
             verticalalignment='top',
             fontsize=anno_fontsize, fontweight='bold')
plt.fill_between(oi[oi <= oi_intersection], 10, cpu_ceiling[oi <= oi_intersection], color='blue', alpha=0.1)
plt.plot(oi[oi <= oi_intersection], cpu_ceiling[oi <= oi_intersection], color='blue', linewidth=4)

cpu_gpu_ceiling_interpolator = interp1d(oi, cpu_gpu_ceiling, kind='linear', fill_value='extrapolate')

# Attention
# for i, data_type in enumerate(data_types):
#     plt.scatter(attn_ois[i], attn_ideal_performance_values[i], label=f'Attention ({data_type}) x Context Length', zorder=5)
#     for j, txt in enumerate(seq_len):
#         plt.annotate(txt, (attn_ois[i][j], attn_ideal_performance_values[i][j]), 
#                      textcoords="offset points", xytext=(0,10), 
#                      ha='center', 
#                      fontsize=anno_fontsize, fontweight='bold')


for i, attn_oi_list in enumerate(attn_ois):
    for j, attn_oi_value in enumerate(attn_oi_list):
        max_intersection = np.max(np.array([np.interp(attn_oi_value, oi, roof) for roof in [cpu_ceiling, cpu_gpu_ceiling]]))   
        plt.vlines(x=attn_oi_value, ymin=10, ymax=max_intersection, color='purple', linestyle='--', label=f'Attention x DataType' if i == 0 else "", linewidth=normal_linewidth)
        plt.annotate(f'{data_types[i]}', (attn_oi_value, cpu_peak_flops),
                    xytext=(attn_oi_value*0.72,max_intersection*0.25), ha='center', va='bottom', color='black',
                    fontsize=anno_fontsize, fontweight='bold')
        # Find and plot intersection points
        plt.scatter([attn_oi_value], max_intersection, color='black', zorder=5)

# Add labels and title
plt.ylim(bottom=10, top=1e6)
plt.xlabel('Operational Intensity (FLOPs/Byte)', fontsize=label_fontsize)
plt.ylabel('Performance (GFLOPS/s)', fontsize=label_fontsize)
plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
# plt.title('Roofline Model', fontsize=28)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=legend_fontsize)
plt.grid(True)

# Save the plot
plt.savefig('roofline-attn-new.pdf', dpi=400, bbox_inches='tight')
# Show the plot
# plt.show()
