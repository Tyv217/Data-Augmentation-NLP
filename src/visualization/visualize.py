from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_and_compare_emb(embeddings1, embeddings2, fig_name):
    pca = PCA(n_components=2)
    reduced1 = pca.fit_transform(embeddings1)
    reduced2 = pca.fit_transform(embeddings2)
    reduced1 = reduced1.transpose()
    reduced2 = reduced2.transpose()
    plt.scatter(reduced1[0], reduced1[1], s=5, c='blue')
    plt.scatter(reduced2[0], reduced2[1], s=5, c='orange')

    ax = plt.gca()
    abs_max = max(abs(max(ax.get_ylim(), key=abs)), abs(max(ax.get_xlim(), key=abs)))
    ax.set_ylim(ymin=-abs_max, ymax=abs_max)
    ax.set_xlim(xmin=-abs_max, xmax=abs_max)
    
    plt.savefig(fig_name)
    plt.show()
    
def plot_emb(embeddings, fig_name, datapoints):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    reduced = reduced.transpose()
    x_mean = np.mean(reduced[0])
    y_mean = np.mean(reduced[1])

    dist = np.sqrt(np.square(reduced[0] - x_mean) + np.square(reduced[1] - y_mean))
    dist1 = np.copy(dist)
    dist1 = np.sort(dist1)
    num_samples = min(datapoints, len(dist1))
    cutoff = dist1[len(dist1) - num_samples]

    x_coords = np.array(reduced[0])[dist >= cutoff]
    y_coords = np.array(reduced[1])[dist >= cutoff]

    x_mean = np.mean(x_coords)
    y_mean = np.mean(y_coords)
    x_std = np.std(x_coords)
    y_std = np.std(y_coords)

    plt.scatter(x_coords, y_coords, s=5, c='green')
    plt.scatter(x_mean, y_mean, s = 10, c='red')
    plt.errorbar(x_mean, y_mean, xerr = x_std, yerr = y_std, fmt = 'o', color = 'red')
    text = "Mean: ({x_mean:.2f},{y_mean:.2f})\nstd: ({x_std:.2f},{y_std:.2f})"\
        .format(x_mean = x_mean, y_mean = y_mean, x_std = x_std, y_std = y_std)

    ax = plt.gca()
    
    abs_max = 0.75
    ax.set_ylim(ymin=-abs_max, ymax=abs_max)
    ax.set_xlim(xmin=-abs_max, xmax=abs_max)
    
    plt.text(0.49,0.49,text,
        horizontalalignment='right',
        verticalalignment='top')
    plt.savefig(fig_name)
    plt.show()

def plot_saliency_scores(words, saliency_scores, fig_name):

    plt.bar(words, saliency_scores)
    plt.ylim([0, 1]) # Set the y-axis limit to 0 and 1
    plt.xlabel('Words')
    plt.ylabel('Saliency Scores')
    plt.title('Word Saliency Scores')


    plt.savefig(fig_name)
    plt.show()
    
def plot_results(args):
    task = args.task
    if task == "classify" or task == "classify_saliency":
        task = args.dataset + task.replace("classify", "")
    pretrain = "pretrain" if args.pretrain else "no_pretrain"
    default_aug_params = "default_aug_params" if args.use_default_augmentation_params != 0 else "no_default_aug_params"
    filename = task + "_" + pretrain + "_" + default_aug_params
    file = filename + ".csv"
    error_file = filename + "_error_bars.csv"
    results = pd.read_csv("reports/data_for_plotting/" + file)
    error_bars = pd.read_csv("reports/data_for_plotting/" + error_file)
    first_col = results.columns[0]
    fig = plt.figure(figsize=(8,5))
    for col in results.columns[1:]:
        plt.plot(results[first_col], results[col], label=col)
        plt.errorbar(results[first_col], results[col], error_bars[col], marker='x', mfc='red',
         mec='red', ms=0, mew=1, capsize=10, elinewidth=2)
    plt.title(filename.replace("_", " "))
    plt.xlabel('Dataset Percentage')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("reports/figures/results/" +filename + ".png")
