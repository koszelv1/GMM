from GMM import GMM
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

from numpy import random
from pandas import DataFrame

def main():

    choice = True

    # random.seed(234)
    random.seed(31)
    x, _ = make_blobs(n_samples=430, centers=5, cluster_std=1.84)

    if choice:

        plt.figure(figsize=(8, 6))
        plt.scatter(x[:, 0], x[:, 1])
        plt.show()

        gmm_custom = GMM(n_components=5)
        gmm_custom.fit(x)
        centers = gmm_custom.mean_vector
        print(centers)

        plt.figure(figsize=(8, 6))
        plt.scatter(x[:, 0], x[:, 1], label="data")
        x_coords = [center[0] for center in centers]
        y_coords = [center[1] for center in centers]
        plt.scatter(x_coords, y_coords, c='r', label="centers")
        plt.legend()
        plt.show()

        pred = gmm_custom.predict(x)

        df = DataFrame({'x': x[:, 0], 'y': x[:, 1], 'label': pred})
        groups = df.groupby('label')

        ig, ax = plt.subplots(figsize=(8, 6))
        for name, group in groups:
            ax.scatter(group.x, group.y, label=name)

        ax.legend()
        plt.show()

        f = plt.figure(figsize=(8, 6), dpi=80)
        f.add_subplot(2, 2, 1)

        fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=80)
        for i in range(2, 6):
            gm = GaussianMixture(n_components=i).fit(x)
            pred = gm.predict(x)
            df = DataFrame({'x': x[:, 0], 'y': x[:, 1], 'label': pred})
            groups = df.groupby('label')
            ax = axes[(i - 2) // 2, (i - 2) % 2]
            for name, group in groups:
                ax.scatter(group.x, group.y, label=name, s=8)
                ax.set_title("Cluster size:" + str(i))
                ax.legend()
        plt.tight_layout()
        plt.show()
    else:

        plt.figure(figsize=(8, 6))
        plt.scatter(x[:, 0], x[:, 1])
        plt.show()

        gm = GaussianMixture(n_components=5).fit(x)
        centers = gm.means_
        print(centers)

        plt.figure(figsize=(8, 6))
        plt.scatter(x[:, 0], x[:, 1], label="data")
        plt.scatter(centers[:, 0], centers[:, 1], c='r', label="centers")
        plt.legend()
        plt.show()

        pred = gm.predict(x)

        df = DataFrame({'x': x[:, 0], 'y': x[:, 1], 'label': pred})
        groups = df.groupby('label')

        ig, ax = plt.subplots(figsize=(8, 6))
        for name, group in groups:
            ax.scatter(group.x, group.y, label=name)

        ax.legend()
        plt.show()

        f = plt.figure(figsize=(8, 6), dpi=80)
        f.add_subplot(2, 2, 1)

        for i in range(2, 6):
            gm = GaussianMixture(n_components=i).fit(x)
            pred = gm.predict(x)
            df = DataFrame({'x': x[:, 0], 'y': x[:, 1], 'label': pred})
            groups = df.groupby('label')
            f.add_subplot(2, 2, i - 1)
            for name, group in groups:
                plt.scatter(group.x, group.y, label=name, s=8)
                plt.title("Cluster size:" + str(i))
                plt.legend()

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    main()