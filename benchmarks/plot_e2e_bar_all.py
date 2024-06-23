import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_e2e_all(dataset, model, devices):
    label_fontsize = 33
    tick_fontsize = 30
    legend_fontsize = 30  # Larger legend font size
    anno_fontsize = 20
    generation_lengths = [32, 64, 128, 256]
    
    device_throughputs = {
        "S2": {
            'FlexGen': [29.2, 34.9, 37.2, 28.8],
            'FlexGen(c)': [17.50, 18.9, 20.00, 15.9],
            'DeepSpeed-Zero': [12.7,13.3,12.1,11.8],
            'MoE-Lightning(p)': [53.7, 67.4, 79.0, 78.6],
            'MoE-Lightning': [203.0, 294.5, 217.5, 167.9]
        },
        "S1": {
            'FlexGen': [12.1, 12.3, 9.5, 9.6],
            'FlexGen(c)': [9.8, 9.4, 7.2, 6.8],
            'DeepSpeed-Zero': [7.1,7.6,7.8,6.7],
            'MoE-Lightning(p)': [15.6, 24, 30.1, 33.9],
            'MoE-Lightning': [63.0, 101.3, 97.73, 96.7]
        },
        "S6": {
            'FlexGen': [4.25, 4.4, 4.77, 3.66],
            'FlexGen(c)': [2.7,2.86,3.44,3.09],
            'DeepSpeed-Zero': [0.56,0.59,0.61, 0.62],
            'MoE-Lightning(p)': [5.38,7.33,7.75,9.13]
        },
        "S7": {
            'FlexGen': [4.97, 5.31, 4.36, 2.96],
            'FlexGen(c)': [1.78,0.97,1.02,0.67],
            'DeepSpeed-Zero': [0.9,1.0,1.2,1.3],
            'MoE-Lightning(p)': [14.9, 22.4, 26.2, 25.8]
        }
    }

    fig, axs = plt.subplots(2, 2, figsize=(28, 14), sharex=True)
    axs = axs.flatten()

    for idx, device in enumerate(devices):
        throughputs = device_throughputs[device]
        x = np.arange(len(generation_lengths))  # the label locations
        n_baselines = len(throughputs)
        group_width = 0.85  # Total width of each group
        width = group_width / n_baselines  # the width of the bars

        max_throughput = max(max(values) for values in throughputs.values())*1.2
        max_non_moe_throughput = np.maximum.reduce([throughputs[key] for key in throughputs if "MoE-Lightning" not in key])

        for i, (label, values) in enumerate(throughputs.items()):
            offset = (i - n_baselines / 2) * width + width / 2
            rects = axs[idx].bar(x + offset, values, width, label=label)
            
            for j, rect in enumerate(rects):
                height = rect.get_height()
                axs[idx].annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=anno_fontsize)

                # Calculate and annotate speedup for MoE-Lightning(p)
                if label == 'MoE-Lightning(p)':
                    speedup = height / max_non_moe_throughput[j]
                    axs[idx].annotate(f'{speedup:.1f}x',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 20),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', 
                                color='red', fontweight='bold',fontsize=anno_fontsize)
            

        axs[idx].set_title(f'MTBench @ {device}', fontsize=label_fontsize)
        axs[idx].set_xticks(x)
        axs[idx].set_xticklabels(generation_lengths if idx >= 2 else [""] * len(generation_lengths))
        axs[idx].tick_params(axis='both', which='major', labelsize=tick_fontsize)
        axs[idx].legend().set_visible(False)
        axs[idx].set_ylim(0, max_throughput)
        if idx == 0 or idx == 2:  # Add y-label only to the left column
            axs[idx].set_ylabel('Throughput (tokens/s)', fontsize=label_fontsize, fontweight='bold')
        if idx == 2 or idx == 3:  # Add y-label only to the left column
            axs[idx].set_xlabel('Generation Length', fontsize=label_fontsize, fontweight='bold')
    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=5, fontsize=legend_fontsize)
    
    plt.tight_layout()
    plt.savefig(f'{dataset}_all.pdf', dpi=400, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mtbench")
    parser.add_argument("--model", type=str, default="Mixtral 8x7B")
    parser.add_argument("--device", type=str, nargs='+', default=["S1", "S2", "S6", "S7"])
    args = parser.parse_args()

    plot_e2e_all(args.dataset, args.model, args.device)
