# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:40:01 2025

@author: maboorai279
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger une image
image_path = "car.png"
image = cv2.imread(image_path)
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
            new_centroid = data[np.random.randint(0, len(data))]  # Si cluster vide, re-initialisation aléatoire
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

# Test avec plusieurs initialisations séquentielles (10 itérations)
def kmeans_multiple_sequential(data, k, num_trials=10, max_iters=10):
    # Créer un graphique avec 2 lignes et 5 colonnes pour afficher les résultats
    fig, ax = plt.subplots(2, 5, figsize=(15, 10))

    # Liste pour stocker les résultats
    centroids_list = []
    clusters_list = []

    for trial in range(num_trials):
        centroids, clusters = kmeans(data, k, max_iters)

        # Sauvegarder les résultats dans la liste
        centroids_list.append(centroids)
        clusters_list.append(clusters)

        # Visualiser les résultats de chaque tirage
        segmented_pixels = np.array([centroids[label] for label in clusters])  # Remplacer chaque pixel par son centroïde
        segmented_image = segmented_pixels.reshape(image.shape)  # Reformer l’image
        
        ax.flat[trial].imshow(segmented_image.astype(np.uint8))
        ax.flat[trial].set_title(f"Segmentation - Tirage {trial+1}")
        ax.flat[trial].axis("off")

    plt.tight_layout()  # Ajuster la disposition pour éviter les chevauchements
    plt.show()

# Choisir un nombre de clusters (k)
k = 10

# Exécuter plusieurs tirages avec affichage des résultats séquentiels
kmeans_multiple_sequential(pixels, k, num_trials=10, max_iters=10)
