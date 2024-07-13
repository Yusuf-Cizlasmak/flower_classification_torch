# * Vektörizasyon işlemi

import logging
import os
import shutil
from itertools import compress

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from sklearn.decomposition import PCA
from tqdm import tqdm

from pre_processing import create_dir

# Boyut azaltma fonksiyonu


def calcuate_pca(embeddings, dim=16):
    pca = PCA(n_components=dim)

    pca_embeddings = pca.fit_transform(embeddings.squeeze())

    logging.info(f"PCA is done. New shape is {pca_embeddings.shape}")

    return pca_embeddings, pca.explained_variance_ratio_


def calcuate_kmeans(embeddings, k):
    """
    data: Kümeleme yapılacak veri noktalarını içeren dizi.
    k: Küme sayısı
    minit: Küme merkezlerinin başlangıçta nasıl seçileceğini belirleyen parametre.
    """

    centroid, labels = kmeans2(data=embeddings, k=k, minit="points")
    counts = np.bincount(labels)

    return centroid, labels


def get_embeddings(file_path):
    embeddings = pd.read_csv(file_path)

    file_paths = embeddings["filepaths"]
    embeddings = embeddings.drop("filepaths", axis=1)

    return embeddings.values, file_paths


# EMBEDDINGSLERI KULLANARAK BOYUT AZALTMA VE KÜMELEME İŞLEMİ


classes = ["Lilly", "Lotus", "Orchid", "Sunflower", "Tulip"]

for class_ in classes:
    embeddings, image_paths = get_embeddings(f"./embeddings/{class_}.csv")

    pca_dims = 50
    k = 10

    # Calculate PCA embeddings
    pca_embeddings, ratio = calcuate_pca(embeddings=embeddings, dim=pca_dims)

    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(ratio), marker="o")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title(f"PCA for {class_}")

    os.makedirs("./pca_plots", exist_ok=True)
    plt.savefig(f"./pca_plots/{class_}.png")
    plt.close()

    # Calculate distortions for different cluster numbers
    distortions = []
    for i in range(1, k + 1):
        centroid, labels = calcuate_kmeans(embeddings=pca_embeddings, k=i)
        distortions.append(
            sum(np.min(np.square(pca_embeddings - centroid[labels]), axis=1))
            / pca_embeddings.shape[0]
        )

    # Plot the elbow curve
    plt.plot(range(1, k + 1), distortions, marker="o")
    plt.xlabel("Number of clusters")
    plt.axvline(x=6, color="g", linestyle="--")
    plt.ylabel("Distortion")
    plt.title(f"Elbow Curve for {class_}")
    plt.legend(classes, loc="upper right")  # Add legend with class names
    os.makedirs("./elbow_plots", exist_ok=True)
    plt.savefig(f"./elbow_plots/{class_}.png")

    for label_number in tqdm(range(6)):
        label_mask = labels == label_number

        path_images = list(compress(image_paths, label_mask))

        target_dir = f"./clustered_images/{class_}/cluster_{label_number}"

        create_dir(target_dir)

        for image_path in path_images:
            shutil.copy(image_path, target_dir)
