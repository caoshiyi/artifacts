import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_e2e_single(dataset, model, device):
    # Data
    generation_lengths = [32, 64, 128, 256]
    # throughputs = {
    #     'FlexGen': [29.15, 34.94, 37.18, 28.82],
    #     'FlexGen(c)': [17.50, 18.92, 20.00, 15.87],
    #     'MoE-Lightning(p)': [96.73, 110.9, 116.03, 97.331],
    #     'MoE-Lightning': [203.69, 289.02, 285.96, 181.60]
    # }
    if device == "L4":
        throughputs = {
            'FlexGen': [29.15, 34.94, 37.18, 28.82],
            'FlexGen(c)': [17.50, 18.92, 20.00, 15.87],
            'MoE-Lightning(p)': [53.71, 67.37, 78.97, 78.6],
            'MoE-Lightning': [203.00, 294.54, 217.54, 167.9]
        }
    elif device == "T4":
        throughputs = {
            'FlexGen': [12.14, 12.32, 9.50, 9.63],
            'FlexGen(c)': [9.81, 9.363, 7.164, 6.782],
            'MoE-Lightning(p)': [15.61, 24, 30.12, 33.94],
            'MoE-Lightning': [63.04, 101.33, 108.62, 96.7]
        }

    x = np.arange(len(generation_lengths))  # the label locations
    width = 0.2  # the width of the bars
    n_baselines = len(throughputs)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate the maximum non-MoE throughput for each generation length
    max_non_moe_throughput = np.maximum.reduce([throughputs[key] for key in throughputs if "MoE-Lightning" not in key])

    # Create bars using a loop and annotate
    for i, (label, values) in enumerate(throughputs.items()):
        offset = (i - n_baselines / 2) * width + width / 2
        rects = ax.bar(x + offset, values, width, label=label)
        
        # Annotate bars
        for j, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
            # Calculate and annotate speedup for MoE-Lightning(p)
            if label == 'MoE-Lightning(p)':
                speedup = height / max_non_moe_throughput[j]
                ax.annotate(f'{speedup:.1f}x',
                            xy=(rect.get_x() + rect.get_width() / 2, height+4),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', 
                            color='red', fontweight='bold')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Generation Length')
    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_xticks(x)
    ax.set_xticklabels(generation_lengths)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)

    # Adding a grid for better readability
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    plt.tight_layout()
    # save
    plt.savefig(f'{dataset}_{device}_{model}_throughput.pdf', dpi=400, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mtbench")
    parser.add_argument("--model", type=str, default="Mixtral 8x7B")
    parser.add_argument("--device", type=str, default="L4")
    args = parser.parse_args()

    all_datasets = ['mtbench']
    if args.dataset != "all":
        plot_e2e_single(args.dataset, args.model, args.device)

