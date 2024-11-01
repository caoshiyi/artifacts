import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_e2e_single(dataset, model, device):
    label_fontsize = 32
    tick_fontsize = 28
    legend_fontsize = 24  # Larger legend font size
    anno_fontsize = 18

    # Data
    generation_lengths = [32, 64, 128, 256]
    # throughputs = {
    #     'FlexGen': [29.15, 34.94, 37.18, 28.82],
    #     'FlexGen(c)': [17.50, 18.92, 20.00, 15.87],
    #     'MoE-Lightning(p)': [96.73, 110.9, 116.03, 97.331],
    #     'MoE-Lightning': [203.69, 289.02, 285.96, 181.60]
    # }
    if device == "s2":
        throughputs = {
            'FlexGen': [29.2, 34.9, 37.2, 28.8],
            'FlexGen(c)': [17.50, 18.9, 20.00, 15.9],
            'DeepSpeed-Zero': [12.7,13.3,12.1,11.8],
            'MoE-Lightning(p)': [53.7, 67.4, 79.0, 78.6],
            'MoE-Lightning': [203.0, 294.5, 217.5, 167.9]
        }
    elif device == "s1":
        throughputs = {
            'FlexGen': [12.1, 12.3, 9.5, 9.6],
            'FlexGen(c)': [9.8, 9.4, 7.2, 6.8],
            'DeepSpeed-Zero': [7.1,7.6,7.8,6.7],
            'MoE-Lightning(p)': [15.6, 24, 30.1, 33.9],
            'MoE-Lightning': [63.0, 101.3, 97.73, 96.7]
        }
    elif device == "s6":
        throughputs = {
            'FlexGen': [4.25, 4.4, 4.77, 3.66],
            'FlexGen(c)': [2.7,2.86,3.44,3.09],
            'DeepSpeed-Zero': [0.56,0.59,0.61, 0.62],
            'MoE-Lightning(p)': [5.38,7.33,7.75,9.13],
        }
    elif device == "s7":
        throughputs = {
            'FlexGen': [4.97, 5.31, 4.36, 2.96],
            'FlexGen(c)': [1.78,0.97,1.02,0.67],
            'DeepSpeed-Zero': [0.9,1.0,1.2,1.3],
            'MoE-Lightning(p)': [14.9, 22.4, 26.2, 25.8],
        }
    elif device == "s8_9":
        throughputs = {
            '2xT4': [34.04, 36.24, 29.67, 25.86],
            '4xT4': [71.54, 83.58, 82.98, 59.45],
        }

    x = np.arange(len(generation_lengths))  # the label locations
    n_baselines = len(throughputs)
    group_width = 0.85  # Total width of each group, a good default is 0.8 to 0.9
    width = group_width / n_baselines  # the width of the bars

    fig, ax = plt.subplots(figsize=(14, 7))

    # Calculate the maximum non-MoE throughput for each generation length
    max_non_moe_throughput = np.maximum.reduce([throughputs[key] for key in throughputs if "MoE-Lightning" not in key])
    max_throughput = max(max(values) for values in throughputs.values())*1.2

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
                        ha='center', va='bottom', fontsize=anno_fontsize)
            
            # Calculate and annotate speedup for MoE-Lightning(p)
            if label == 'MoE-Lightning(p)':
                speedup = height / max_non_moe_throughput[j]
                ax.annotate(f'{speedup:.1f}x',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 19),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', 
                            color='red', fontweight='bold',fontsize=anno_fontsize)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Generation Length', fontsize=label_fontsize, fontweight='bold')
    if device == "s8_9" or device == "s1" or device == "s6":
        ax.set_ylabel('Throughput (tokens/s)', fontsize=label_fontsize, fontweight='bold')
    # ax.set_ylabel('Throughput (tokens/s)', fontsize=label_fontsize, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    plt.ylim(0, max_throughput)
    ax.set_xticks(x)
    ax.set_xticklabels(generation_lengths)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=legend_fontsize)

    # Adding a grid for better readability
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    plt.tight_layout()
    # save
    plt.savefig(f'{dataset}_{device}.pdf', dpi=400, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mtbench")
    parser.add_argument("--model", type=str, default="Mixtral 8x7B")
    parser.add_argument("--device", type=str, default="L4")
    args = parser.parse_args()

    all_datasets = ['mtbench']
    if args.dataset != "all":
        plot_e2e_single(args.dataset, args.model, args.device)

