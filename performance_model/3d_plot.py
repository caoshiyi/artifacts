import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from cost_model import get_throughput
from model_config import ModelConfig
import os

# Initial setup
config = ModelConfig("mistralai/Mixtral-8x7B-Instruct-v0.1")
num_cpu_nodes = np.arange(1, 11)
cpu_gpu_bdw = np.linspace(100, 500, 5)

data_paths = {
    'weights_on_cpu': 'weights_on_cpu.npy',
    'cpu_attention_enabled': 'cpu_attention_enabled.npy',
    'kv_cache_cpu': 'kv_cache_cpu.npy'
}
data_exists = all(os.path.exists(data_paths[key]) for key in data_paths)
print(data_exists)

if not data_exists:
    # Data matrices initialization
    weights_on_cpu = np.zeros((5, 10))
    cpu_attention_enabled = np.zeros((5, 10))
    kv_cache_cpu = np.zeros((5, 10))

    # Simulation loop
    for i, cpu_node in enumerate(num_cpu_nodes):
        for j, bdw in enumerate(cpu_gpu_bdw):
            cpu_mem = 200 * cpu_node
            gpu_mem = 160  # Assuming fixed two GPUs
            cpu_bw = 100 * cpu_node
            gpu_bw = 4000  # Assuming fixed bandwidth for two GPUs
            cpu_flops = 1.6 * cpu_node
            gpu_flops = 600  # Assuming fixed flops for two GPUs

            gattn_throughput, policy_gpu = get_throughput(config, cpu_mem, gpu_mem, cpu_bw, gpu_bw, cpu_flops, gpu_flops, bdw, True)
            cattn_throughput, policy_cpu = get_throughput(config, cpu_mem, gpu_mem, cpu_bw, gpu_bw, cpu_flops, gpu_flops, bdw, False)
            policy = policy_gpu if gattn_throughput > cattn_throughput else policy_cpu
            weights_on_cpu[j, i] = 1 - policy.wg
            cpu_attention_enabled[j, i] = 1 if gattn_throughput < cattn_throughput else 0
            kv_cache_cpu[j, i] = policy.cc

    np.save('weights_on_cpu.npy', weights_on_cpu)
    np.save('cpu_attention_enabled.npy', cpu_attention_enabled)
    np.save('kv_cache_cpu.npy', kv_cache_cpu)
else:
    weights_on_cpu = np.load('weights_on_cpu.npy')
    cpu_attention_enabled = np.load('cpu_attention_enabled.npy')
    kv_cache_cpu = np.load('kv_cache_cpu.npy')


# single plot
# # Plotting
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# X, Y = np.meshgrid(num_cpu_nodes, cpu_gpu_bdw)
# Z = kv_cache_cpu

# # 3D Surface plot
# surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# # Overlay Heatmap as a texture on the X-Y plane
# heatmap_z = Z.min() - 0.1
# ax.contourf(X, Y, cpu_attention_enabled, zdir='z', offset=heatmap_z, cmap='coolwarm', alpha=0.5)

# ax.set_xlabel('Number of CPU Nodes')
# ax.set_ylabel('CPU-GPU Bandwidth')
# ax.set_zlabel('Normalized Weights on CPU')
# ax.set_title('CPU Attention Enabled')
# plt.savefig('3d_plot.pdf', dpi=400, bbox_inches='tight')
# plt.show()


# 3d subplots
# Plotting
# fig = plt.figure(figsize=(12, 6))
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122, projection='3d')

# X, Y = np.meshgrid(num_cpu_nodes, cpu_gpu_bdw)

# # Weights on GPU 3D Surface plot
# ax1.plot_surface(X, Y, weights_on_cpu, cmap='viridis', alpha=0.7)
# # Overlay Heatmap as a texture on the X-Y plane
# heatmap_z = weights_on_cpu.min() - 0.1
# ax1.contourf(X, Y, cpu_attention_enabled, zdir='z', offset=heatmap_z, cmap='coolwarm', alpha=0.5)
# ax1.set_title('Weights on CPU')
# ax1.set_xlabel('Number of CPU Nodes')
# ax1.set_ylabel('CPU-GPU Bandwidth')
# ax1.set_zlabel('Weights')

# # KV Cache on CPU 3D Surface plot
# ax2.plot_surface(X, Y, kv_cache_cpu, cmap='plasma', alpha=0.7)
# # Overlay Heatmap as a texture on the X-Y plane
# heatmap_z = weights_on_cpu.min() - 0.1
# ax2.contourf(X, Y, cpu_attention_enabled, zdir='z', offset=heatmap_z, cmap='coolwarm', alpha=0.5)
# ax2.set_title('KV Cache on CPU')
# ax2.set_xlabel('Number of CPU Nodes')
# ax2.set_ylabel('CPU-GPU Bandwidth')
# ax2.set_zlabel('Cache Ratio')
# plt.tight_layout()
# plt.savefig('3d_plot.pdf', dpi=400, bbox_inches='tight')
# plt.show()

title_fontsize = 18
label_fontsize = 18
tick_fontsize = 14
legend_fontsize = 18  # Larger legend font size

# contour plot
fig, axs = plt.subplots(1, 2, figsize=(13, 6))
X, Y = np.meshgrid(num_cpu_nodes, cpu_gpu_bdw)
# Contour plot for Weights on GPU
contour1 = axs[0].contourf(X, Y, weights_on_cpu, cmap='viridis', levels=20)
cbar0 = fig.colorbar(contour1, ax=axs[0])
# set colorbar's fontsize
cbar0.ax.tick_params(labelsize=tick_fontsize)
axs[0].set_title('Ratio of Weights on CPU', fontsize=title_fontsize, fontweight='bold')
axs[0].set_xlabel('CPU Scaling Ratio', fontsize=label_fontsize, fontweight='bold')
axs[0].set_ylabel('CPU-GPU Bandwidth', fontsize=label_fontsize, fontweight='bold')
axs[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)

# Overlay GPU attention enabled points
for i in range(len(num_cpu_nodes)):
    for j in range(len(cpu_gpu_bdw)):
        if cpu_attention_enabled[j, i] == 1:
            axs[0].plot(num_cpu_nodes[i], cpu_gpu_bdw[j], 'ro')  # 'ro' for red circle

# Contour plot for KV Cache on CPU
contour2 = axs[1].contourf(X, Y, kv_cache_cpu, cmap='plasma', levels=20)
cbar1 = fig.colorbar(contour2, ax=axs[1])
cbar1.ax.tick_params(labelsize=tick_fontsize)
axs[1].set_title('Ratio of KV Cache on CPU', fontsize=title_fontsize, fontweight='bold')
axs[1].set_xlabel('CPU Scaling Ratio', fontsize=label_fontsize, fontweight='bold')
axs[1].set_ylabel('CPU-GPU Bandwidth', fontsize=label_fontsize, fontweight='bold')
axs[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
# Overlay GPU attention enabled points
for i in range(len(num_cpu_nodes)):
    for j in range(len(cpu_gpu_bdw)):
        if cpu_attention_enabled[j, i] == 1:
            axs[1].plot(num_cpu_nodes[i], cpu_gpu_bdw[j], 'ro')  # 'ro' for red circle
    


plt.savefig('contour_plot.pdf', dpi=400, bbox_inches='tight')
plt.tight_layout()
plt.show()

