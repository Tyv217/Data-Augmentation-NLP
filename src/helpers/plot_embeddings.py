from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_and_compare_emb(embeddings1, embeddings2, fig_name):
    pca = PCA(n_components=2)
    reduced1 = pca.fit_transform(embeddings1)
    reduced2 = pca.fit_transform(embeddings2)
    reduced1 = reduced1.transpose()
    reduced2 = reduced2.transpose()
    plt.scatter(reduced1[0], reduced1[1], c='blue')
    plt.scatter(reduced2[0], reduced2[1], c='orange')
    plt.savefig(fig_name)
    plt.show()
    
