import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_pca(vectors, labels, title="PCA Visualization"):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    plt.figure(figsize=(8,6))
    for i, label in enumerate(labels):
        x, y = reduced[i]
        plt.scatter(x, y, label=label)
        plt.text(x+0.01, y+0.01, label, fontsize=9)
    plt.title(title)
    plt.legend()
    plt.show()