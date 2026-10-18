# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:40:01 2025

@author: maboorai279
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Pour la visualisation en 3D

# Charger une image
image_path = "val2.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir BGR → RGB  cv::COLOR_BGR2GRAY
image = cv2.resize(image, (400, 200))

# Transformer l’image en un tableau de pixels en forme (nb_pixels, 3 couleurs)
pixels = image.reshape((-1, 3))

# Calcule la distance euclidienne entre deux points
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Initialisation aléatoire des centroïdes
def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

# Assigne chaque point au centroïde le plus proche
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = np.argmin(distances)  # Trouve l'indice du centroïde le plus proche
        clusters.append(cluster)
    return np.array(clusters)

# Recalcule les centroïdes
def update_centroids(data, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = data[clusters == i]  
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
        else:
            new_centroid = data[np.random.randint(0, len(data))]  # Si cluster vide, re-init
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

# K-Means avec visualisation en 3D

def kmeans_visualize_3D_1(data, k, max_iters=10):
    centroids = initialize_centroids(data, k)


    for i in range(max_iters):
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        # Palette de couleurs pour chaque cluster
        colors = plt.cm.get_cmap("tab10", k)

        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        ax1.clear()  # Nettoyer le graphique pour la nouvelle itération
        ax1.set_xlabel("Canal Rouge (R)")
        ax1.set_ylabel("Canal Vert (G)")
        ax1.set_zlabel("Canal Bleu (B)")
        ax1.set_title(f"Iteration {i+1} de K-Means")

        # Afficher les points de l'image en fonction de leur cluster
        for j in range(k):
            cluster_points = data[clusters == j]
            if len(cluster_points) > 0:
                ax1.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                           color=colors(j), label=f"Cluster {j+1}", alpha=0.5, s=2)

        # Afficher les centroïdes
        ax1.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
                   color="black", marker="x", s=100, label="Centroïdes")

        ax1.legend()
                
        ax2.clear()  # Nettoyer l'affichage précédent
        segmented_pixels = np.array([centroids[label] for label in clusters])
        segmented_image = segmented_pixels.reshape(image.shape)
        ax2.imshow(segmented_image.astype(np.uint8))
        ax2.set_title(f"Segmentation - Itération {i+1}")
        ax2.axis("off")

        #plt.pause(0.3)  # Pause pour voir l'évolution des centroïdes
        
        # Vérification de convergence
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break

        centroids = new_centroids

    plt.show()    

    

def kmeans_visualize_3D_2(data, k, max_iters=10):
    centroids = initialize_centroids(data, k)

    for i in range(max_iters):
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        # Assigner chaque pixel à un cluster
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        ax1.clear()  # Nettoyer le graphique pour la nouvelle itération
        ax1.set_xlabel("Canal Rouge (R)")
        ax1.set_ylabel("Canal Vert (G)")
        ax1.set_zlabel("Canal Bleu (B)")
        ax1.set_title(f"Iteration {i+1} de K-Means")

        # Afficher les points en 3D avec la couleur correspondant à leur cluster
        for j in range(k):
            cluster_points = data[clusters == j]
            if len(cluster_points) > 0:
                ax1.scatter(
                    cluster_points[:, 0], 
                    cluster_points[:, 1], 
                    cluster_points[:, 2], 
                    color=[centroids[j] / 255],  # Utiliser les couleurs des centroïdes normalisées
                    label=f"Cluster {j+1}", 
                    alpha=0.5, 
                    s=2
                )

        # Afficher les centroïdes en noir
        ax1.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
                   color="black", marker="x", s=100, label="Centroïdes")

        ax1.legend()

        # Afficher l'image segmentée
        ax2.clear()
        segmented_pixels = np.array([centroids[label] for label in clusters])
        segmented_image = segmented_pixels.reshape(image.shape)
        ax2.imshow(segmented_image.astype(np.uint8))
        ax2.set_title(f"Segmentation - Itération {i+1}")
        ax2.axis("off")

        # Vérifier si l'algorithme a convergé
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break

        centroids = new_centroids  # Mettre à jour les centroïdes

    plt.show()

def kmeans_visualize_3D_3(data, k, max_iters=10):
    centroids = initialize_centroids(data, k)
    palette = np.random.randint(0, 255, (k, 3))  # Palette de fausses couleurs

    for i in range(max_iters):
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)

        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        ax1.clear()
        ax1.set_xlabel("Canal Rouge (R)")
        ax1.set_ylabel("Canal Vert (G)")
        ax1.set_zlabel("Canal Bleu (B)")
        ax1.set_title(f"Iteration {i+1} de K-Means")

        for j in range(k):
            cluster_points = data[clusters == j]
            if len(cluster_points) > 0:
                ax1.scatter(
                    cluster_points[:, 0], 
                    cluster_points[:, 1], 
                    cluster_points[:, 2], 
                    color=[centroids[j] / 255], 
                    label=f"Cluster {j+1}", 
                    alpha=0.5, 
                    s=2
                )

        ax1.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
                   color="black", marker="x", s=100, label="Centroïdes")

        ax1.legend()

        # Appliquer fausses couleurs
        segmented_pixels = np.array([palette[label] for label in clusters])
        segmented_image = segmented_pixels.reshape(image.shape)

        ax2.clear()
        ax2.imshow(segmented_image.astype(np.uint8))
        ax2.set_title(f"Segmentation - Itération {i+1} (Fausses couleurs)")
        ax2.axis("off")

        if np.allclose(centroids, new_centroids, atol=1e-4):
            break

        centroids = new_centroids  

    plt.show()


def histogramme(image):
    h = np.zeros([256, 1])  
    NL, NC = image.shape 
    for i in range(NL):
        for j in range(NC):
            val = image[i, j]  
            h[val] += 1 
    return h
    
    
h = histogramme(image[:, :, 2])
plt.plot(h)

# Nombre de clusters
k = 6

# Lancer la visualisation combinée 3D + 2D
kmeans_visualize_3D_3(pixels, k, max_iters=30)