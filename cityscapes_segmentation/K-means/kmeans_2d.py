import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Charger une image
image_path = "test2.png"
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
        cluster = np.argmin(distances) # Trouve l'indice du centroïde le plus proche
        clusters.append(cluster)
    return np.array(clusters)

# Recalcule les centroïdes en prenant la moyenne des points dans chaque cluster
def update_centroids(data, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = data[clusters == i] # Récupérer tous les pixels du cluster i
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
        else:
            new_centroid = data[np.random.randint(0, len(data))]  # Si cluster vide, re-initialisation aléatoire
        new_centroids.append(new_centroid)
    return np.array(new_centroids)

def visualize_segmentation_pca(data, clusters, centroids, iteration):
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(data)
    transformed_centroids = pca.transform(centroids)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=clusters, cmap='viridis', s=1)
    plt.scatter(transformed_centroids[:, 0], transformed_centroids[:, 1], c='red', marker='x', s=100, label="Centroïdes")
    plt.colorbar(scatter)
    plt.title(f"Segmentation en 2D avec PCA - Itération {iteration}")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.legend()
    plt.show()
    
# Modifier la boucle dans kmeans pour ajouter cette visualisation
def kmeans_with_pca(data, k, max_iters=10):
    centroids = initialize_centroids(data, k)
    
    for i in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        # Vérification de convergence
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break

        centroids = new_centroids
        
        # Visualiser l’évolution de la segmentation en 2D
        visualize_segmentation_pca(data, clusters, centroids, i+1)

    return centroids, clusters

# Nombre de segments optimal (à choisir après analyse de la courbe)
k = 10

# Appliquer le K-means avec PCA
centroids, labels = kmeans_with_pca(pixels, k)
