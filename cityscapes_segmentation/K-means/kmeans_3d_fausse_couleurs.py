# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:56:28 2025

@author: maboorai279
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Charger une image
image_path = "test.png"  # Utilisez le chemin correct pour l'image téléchargée
image = cv2.imread(image_path)

if image is None:
    print(f"Erreur de chargement de l'image depuis {image_path}. Vérifiez le chemin du fichier.")
else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir BGR → RGB
    image = cv2.resize(image, (400, 200))

    # Transformer l’image en un tableau de pixels en forme (nb_pixels, 3 couleurs)
    pixels = image.reshape((-1, 3))

    # Calcule la distance euclidienne entre deux points
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # Initialisation aléatoire des centroïdes parmi les pixels existants.
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

    # Recalcule les centroïdes en prenant la moyenne des points dans chaque cluster
    def update_centroids(data, clusters, k):
        new_centroids = []
        for i in range(k):
            cluster_points = data[clusters == i]  # Récupérer tous les pixels du cluster i
            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points, axis=0)
            else:
                new_centroid = data[np.random.randint(0, len(data))]  # Si cluster vide, ré-initialisation aléatoire
            new_centroids.append(new_centroid)
        return np.array(new_centroids)

    # Implémentation complète de K-Means
    def kmeans(data, k, max_iters=10):
        centroids = initialize_centroids(data, k)

        for i in range(max_iters):
            clusters = assign_clusters(data, centroids)
            new_centroids = update_centroids(data, clusters, k)

            # Vérification de convergence
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break

            centroids = new_centroids

        return centroids, clusters

    # Nombre de segments (clusters) à utiliser
    k = 10

    # Appliquer K-means pour obtenir les centroïdes et les clusters
    centroids, clusters = kmeans(pixels, k)

    # Définir des couleurs spécifiques pour chaque cluster (associées à chaque classe)
    cluster_colors = [
        (255, 0, 0),   # Bleu pour le Cluster 1
        (0, 255, 0),   # Vert pour le Cluster 2
        (255, 255, 0), # Jaune pour le Cluster 3
        (0, 255, 255), # Cyan pour le Cluster 4
        (255, 0, 255), # Magenta pour le Cluster 5
        (255, 0, 100),   # Bleu pour le Cluster 1
        (255, 255, 255),   # Vert pour le Cluster 2
        (255, 255, 150), # Jaune pour le Cluster 3
        (0, 115, 255), # Cyan pour le Cluster 4
        (255, 0, 112), # Magenta pour le Cluster 5
    ]

    # Normaliser les couleurs (convertir les valeurs de 0-255 à 0-1)
    cluster_colors_normalized = [np.array(color) / 255.0 for color in cluster_colors]

    # Recréer l'image segmentée avec ces couleurs spécifiques
    segmented_pixels = np.array([cluster_colors_normalized[label] for label in clusters])  # Remplacer chaque pixel par son centroïde
    segmented_image = segmented_pixels.reshape(image.shape)  # Reformer l’image

    # Création de l'image segmentée
    fig = plt.figure(figsize=(15, 7))

    # Affichage de l'image segmentée
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(segmented_image)
    ax1.set_title('Image segmentée')
    ax1.axis("off")

    # Création d'un sous-graphe 3D pour afficher les clusters
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # Affichage des points dans l'espace 3D (chaque cluster avec une couleur différente)
    for i in range(k):
        cluster_points = pixels[clusters == i]
        ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                    c=[cluster_colors_normalized[i]]*len(cluster_points), label=f"Cluster {i+1}", s=2)

    # Affichage des centroïdes
    ax2.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', marker='x', s=100, label='Centroïdes')

    # Ajouter des labels et un titre
    ax2.set_xlabel('R')
    ax2.set_ylabel('G')
    ax2.set_zlabel('B')
    ax2.set_title('Segmentation des classes en 3D')

    # Légende
    ax2.legend()

    plt.show()