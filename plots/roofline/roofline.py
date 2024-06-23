import numpy as np
import matplotlib.pyplot as plt
from utils import MLP_flops, MLP_bytes, attention_flops, attention_bytes
from scipy.interpolate import interp1d


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

# MLP
hidden_size1 = 4096
hidden_size2 = 14336
batch_sizes = [32, 128, 1024, 1024*16]
n_experts = 8
topk = 2

mlp_oi = [MLP_flops(hidden_size1, hidden_size2, bs, topk) / (MLP_bytes(hidden_size1, hidden_size2, bs, n_experts)) for bs in batch_sizes]
mlp_ideal_performance_values = [min(gpu_mem_bandwidth * oi, gpu_peak_flops) for oi in mlp_oi]

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
plt.loglog(oi, cpu_ceiling, label='CPU Mem Bdw', linestyle='-', color='blue')
plt.loglog(oi, gpu_ceiling, label='GPU Mem Bdw', linestyle='-', color='green')
plt.loglog(oi, cpu_gpu_ceiling, label='CPU-GPU Mem Bdw', linestyle='-', color='red')

# Add horizontal lines for peak FLOPS
plt.axhline(y=cpu_peak_flops, color='blue', linestyle='--', label='CPU Peak FLOPS')
plt.axhline(y=gpu_peak_flops, color='green', linestyle='--', label='GPU Peak FLOPS')

plt.vlines(x=oi_intersection, ymin=10, ymax=cpu_peak_flops, color='grey', linestyle='--')
plt.scatter(oi_intersection, cpu_peak_flops, color='red', zorder=5, label='Important Intersections')
plt.annotate('P1',
             xy=(oi_intersection, cpu_peak_flops), 
             xytext=(oi_intersection * 0.9, cpu_peak_flops * 1.6),
             horizontalalignment='right',
             verticalalignment='top',
             fontsize=13)

cpu_gpu_ceiling_interpolator = interp1d(oi, cpu_gpu_ceiling, kind='linear', fill_value='extrapolate')
# MLP
plt.scatter(mlp_oi[1:2], mlp_ideal_performance_values[1:2], color='orange', label='MLP x ubs', zorder=5)
plt.annotate(f'ubs {batch_sizes[1]}',
             xy=(mlp_oi[1], mlp_ideal_performance_values[1]), 
             xytext=(mlp_oi[1] * 0.9, mlp_ideal_performance_values[1] * 1.6),
             horizontalalignment='right',
             verticalalignment='top',
             fontsize=13)

y_intersections = []
# Draw horizontal lines and find intersections
for mlp_oi_value, mlp_performance_value in zip(mlp_oi[1:2], mlp_ideal_performance_values[1:2]):
    # Interpolate to find the closest oi that matches the performance value along the ceiling
    x_intersection = np.min(oi[np.where(cpu_gpu_ceiling >= mlp_performance_value)])
    y_intersection = cpu_gpu_ceiling_interpolator(x_intersection)
    y_intersections.append(y_intersection)
    plt.scatter(x_intersection, y_intersection, color='red', zorder=5)
    plt.annotate('P2',
             xy=(x_intersection, y_intersection), 
             xytext=(x_intersection * 0.9, y_intersection * 1.6),
             horizontalalignment='right',
             verticalalignment='top',
             fontsize=13)
    # Draw horizontal line to the intersection point
    plt.hlines(y=mlp_performance_value, xmin=mlp_oi_value, xmax=oi[-1], color='orange', zorder=4)
    
    # Draw vertical line from intersection point down
    plt.vlines(x=x_intersection, ymin=10, ymax=y_intersection, color='gray', linestyle='--', zorder=4)

for i, mlp_oi_value in enumerate(mlp_oi):
    max_intersection = min(np.max(np.array([np.interp(mlp_oi_value, oi, roof) for roof in [cpu_ceiling, cpu_gpu_ceiling]])), y_intersections[0])
    print(max_intersection)   
    # Draw vertical line up to the intersection point
    plt.vlines(x=mlp_oi_value, ymin=10, ymax=max_intersection, color='purple', linestyle='--', label='MLP x bs' if i == 0 else "")
    plt.annotate(f'bs {batch_sizes[i]}', (mlp_oi_value, cpu_peak_flops),
                 xytext=(mlp_oi_value,cpu_peak_flops*0.05), ha='center', va='bottom', color='black',
                 fontsize=12)
    # Find and plot intersection points
    plt.scatter([mlp_oi_value], max_intersection, color='black', zorder=5)


# # Attention
# for i, data_type in enumerate(data_types):
#     plt.scatter(attn_ois[i], attn_ideal_performance_values[i], label=f'Attention ({data_type}) x Context Length', zorder=5)
    # for j, txt in enumerate(seq_len):
    #     plt.annotate(txt, (attn_ois[i][j], attn_ideal_performance_values[i][j]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=13)


# for i, attn_oi_list in enumerate(attn_ois):
#     for j, attn_oi_value in enumerate(attn_oi_list):
#         max_intersection = np.max(np.array([np.interp(attn_oi_value, oi, roof) for roof in [cpu_ceiling, cpu_gpu_ceiling]]))   
#         plt.vlines(x=attn_oi_value, ymin=10, ymax=max_intersection, color='purple', linestyle='--', label=f'Attention x DataType' if i == 0 else "")
#         plt.annotate(f'{data_types[i]}', (attn_oi_value, cpu_peak_flops),
#                     xytext=(attn_oi_value*0.75,cpu_peak_flops*0.03), ha='center', va='bottom', color='black',
#                     fontsize=12)
#         # Find and plot intersection points
#         plt.scatter([attn_oi_value], max_intersection, color='black', zorder=5)

# Add labels and title
plt.ylim(bottom=10, top=1e6)
plt.xlabel('Operational Intensity (FLOPs/Byte)', fontsize=22)
plt.ylabel('Performance (GFLOPS/s)', fontsize=22)
plt.tick_params(axis='both', which='major', labelsize=12)
# plt.title('Roofline Model', fontsize=28)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=12)
plt.grid(True)

# Save the plot
plt.savefig('roofline-mlp.pdf', dpi=400, bbox_inches='tight')
# Show the plot
plt.show()
