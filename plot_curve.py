import matplotlib.pyplot as plt
import json 
import os
import numpy as np
import seaborn as sns

save_folder = os.path.join('eval/', 'COCO')

def plot_loss_and_lr(train_loss, learning_rate):
    try:
        x = list(range(len(train_loss)))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss, 'r', label='loss')
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title("Train Loss and lr")
        plt.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(x, learning_rate, label='lr')
        ax2.set_ylabel("learning rate")
        ax2.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况
        fig.savefig('./loss_and_lr.png')
        plt.close()
        print("successful save loss curve! ")
    except Exception as e:
        print(e)


def plot_map(ap_stats, metrics, legend):
    style = 'dark'
    sns.set_style(style)
    out = os.path.join(save_folder, 'ap.png')
    title = None

    i = 0
    num_metrics = len(metrics)
    epochs = ap_stats['epoch']
    for j, metric in enumerate(metrics):
        xs  = np.asarray(epochs)
        ys = ap_stats[metric]
        ax = plt.gca()
        ax.set_xticks(xs)
        plt.xlabel('epoch')
        plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')

        plt.legend()
    if title is not None:
        plt.title(title)
    
    print(f'save curve to: {out}')
    plt.savefig(out)
    plt.show()
    plt.cla()


if __name__ == '__main__':
    ap_stats = {'ap50': [0.003985194073006601, 0.6851529061645676], 
            'ap_small': [0.0, 0.0673558810299327], 
            'ap_medium': [0.0012955153247929476, 0.34469121736332514], 
            'ap_large': [0.005240430664568138, 0.30883813438445296], 
            'epoch': [100, 110]}
            
    res_file = None
    res_file = os.path.join(save_folder, 'ap_stats.json')
    if res_file:
        print('Writing ap stats json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(ap_stats, fid)
        with open(res_file) as f:
            ap_stats = json.load(f)

    metrics = ['ap50', 'ap_small', 'ap_medium', 'ap_large']
    legend  = ['ap50', 'ap_small', 'ap_medium', 'ap_large']
    plot_map(ap_stats, metrics, legend)
