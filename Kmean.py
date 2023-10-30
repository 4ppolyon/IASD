import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import random


# Define a function to calculate the distance between two points
def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


# Define a function to check if the centroids have changed
def centroids_changed(centroids, old_centroids):
    if len(old_centroids) == 0:
        return True
    for i in range(3):
        if not np.array_equal(centroids[i]["point"], old_centroids[i]):
            return True
    return False


# Define a function to calculate the new centroids as a point at the mean of associated points
def new_centroids(centroids):
    # For each centroid, calculate the mean of its associated points
    for i in range(3):
        if centroids[i]["nb_points"] > 0:
            centroids[i]["point"] = np.mean(centroids[i]["points"], axis=0)
        centroids[i]["points"] = []
        centroids[i]["nb_points"] = 0


# Define a function to assign each point to the closest centroid
def assign_points(centroids, points):
    for point in points:
        distances = [distance(point, centroids[x]["point"]) for x in
                     range(3)]  # Calculate the distance between the point and each centroid and store them in a list
        min_index = distances.index(min(distances))  # Find the index of the closest centroid
        centroids[min_index]["points"].append(point)
        centroids[min_index]["nb_points"] += 1
    return centroids


# Define a function to display points and centroids with their respective colors
def display(centroids):
    for centroid in range(3):
        plt.scatter(centroids[centroid]["point"][1], centroids[centroid]["point"][0], c=centroids[centroid]["color"],
                    marker='x',
                    s=100)  # for each centroid, display it with a cross
        for point in centroids[centroid][
            "points"]:  # for each point associated to the centroid, display it with the centroid's color
            plt.scatter(point[1], point[0], c=centroids[centroid]["color"])
    plt.show()


def verif(centroids, points):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=0).fit(points)
    centroid_verif = kmeans.cluster_centers_
    # on calcul les distances entre les centroides de sklearn et les centroides de notre algo pour toutes les combinaisons
    # et on affiche les distances minimales pour chaque centroides avec les coordonnées des deux algorithmes
    for i in range(3):
        distances = [distance(centroid_verif[i], centroids[x]["point"]) for x in range(3)]
        min_index = distances.index(min(distances))
        # print("Distance min pour le centroid de sklearn", i, ":", min(distances), "avec le centroid de notre algo",
        #       min_index +1)
        print("Centroid", i + 1, ":", centroid_verif[i])
        print("Centroid", i + 1, ":", centroids[min_index]["point"])
        print("Distance between them:", min(distances))
    # on display les points avec les centroides de sklearn
    for i in range(3):
        plt.scatter(centroid_verif[i][1], centroid_verif[i][0], c=centroids[i]["color"], marker='x', s=100)
    plt.show()


# Implement the k-means algorithm that stops when centroids no longer change or after 100 iterations
def k_means(centroids, old_centroids, points):
    k = 0
    while k < 100:
        centroids = assign_points(centroids, points)
        display(centroids)
        if not centroids_changed(centroids, old_centroids):
            break
        # Copie des valeurs de centroids dans old_centroids
        old_centroids = []
        for x in range(3):
            old_centroids.append(centroids[x]["point"].copy())
        new_centroids(centroids)
        k += 1
    verif(centroids, points)
    print("Number of iterations:", k + 1)


# Loading the iris dataset
iris = datasets.load_iris()
points = iris.data
# mettre points en 2D
# points = points[:, [0, 3]]

# Define a list of dictionaries called centroids to store centroid information
centroids = []
old_centroids = []
# Firstly, choose 3 random points from the dataset and add them to centroids with different colors
colors = ['r', 'g', 'b']
for i in range(3):
    # Choose a random point
    point = points[random.randint(0, len(points) - 1)]
    # Check if the point is already in centroids
    while any(np.array_equal(point, centroid["point"]) for centroid in centroids):
        point = points[random.randint(0, len(points) - 1)]
    # Add the point to centroids with a different color
    centroid = {
        "point": point,
        "points": [],
        "nb_points": 0,
        "color": colors[i]
    }
    centroids.append(centroid)
k_means(centroids, old_centroids, points)
