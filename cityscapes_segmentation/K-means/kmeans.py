import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Implémentation complète de K-Means
# K-Means avec affichage après chaque itération
def kmeans(data, k, max_iters=10):
    centroids = initialize_centroids(data, k)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Image originale")
    ax[0].axis("off")


    for i in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        # Vérification de convergence
        if np.allclose(centroids, new_centroids, atol=1e-4):
            break

        centroids = new_centroids
        
        # Affichage de l'image segmentée après chaque itération
        segmented_pixels = np.array([centroids[label] for label in clusters])  # Remplace chaque pixel par son centroïde
        segmented_image = segmented_pixels.reshape(image.shape)  # Reformer l’image

        ax[1].clear()
        ax[1].imshow(segmented_image.astype(np.uint8))
        ax[1].set_title(f"Segmentation après {i+1} itérations (5 classes)")
        ax[1].axis("off")

        plt.pause(0.5)  # Pause pour voir l'évolution

    plt.show()
    return centroids, clusters

# Méthode du coude pour déterminer le k optimal
def elbow_method(data, max_k=20):
    k_values = list(range(1, max_k + 1))
    inertias = []

    for k in k_values:
        centroids = initialize_centroids(data, k)
        for _ in range(10):
            clusters = assign_clusters(data, centroids)
            new_centroids = update_centroids(data, clusters, k)
            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids

        # Calculer l'inertie
        inertia = sum(np.sum((data[clusters == i] - centroids[i]) ** 2) for i in range(k))
        inertias.append(inertia)

    # Tracer la courbe du coude
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker='o', linestyle='-', color='b')
    plt.xlabel("Nombre de clusters (k)")
    plt.ylabel("Inertie intra-cluster")
    plt.title("Méthode du coude pour déterminer k optimal")
    plt.grid(True)
    plt.show()

# Appliquer la méthode du coude
elbow_method(pixels, max_k=20)

# Nombre de segments optimal (à choisir après analyse de la courbe)
k = 10
centroids, labels = kmeans(pixels, k)
