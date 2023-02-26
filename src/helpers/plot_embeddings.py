from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

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
    
def plot_emb(embeddings, fig_name):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    reduced = reduced.transpose()
    x_mean = np.mean(reduced[0])
    y_mean = np.mean(reduced[1])
    x_std = np.std(reduced[0])
    y_std = np.std(reduced[1])
    plt.scatter(reduced[0], reduced[1], s=5, c='green')
    plt.scatter(x_mean, y_mean, s = 10, c='red')
    plt.errorbar(x_mean, y_mean, xerr = x_std, yerr = y_std, fmt = 'o', color = 'red')
    text = "Mean: ({x_mean:.2f},{y_mean:.2f})\nstd: ({x_std:.2f},{y_std:.2f})"\
        .format(x_mean = x_mean, y_mean = y_mean, x_std = x_std, y_std = y_std)

    ax = plt.gca()
    abs_max = 0.5
    ax.set_ylim(ymin=-abs_max, ymax=abs_max)
    ax.set_xlim(xmin=-abs_max, xmax=abs_max)
    
    plt.text(0.5,0.5,text,
        horizontalalignment='right',
        verticalalignment='top')
    plt.savefig(fig_name)
    plt.show()
    