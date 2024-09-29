import math
import matplotlib.pyplot as plt

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

# Function to assign each data point to the nearest centroid
def assign_clusters(data, centroids):
    clusters = {centroid: [] for centroid in centroids}
    
    for point in data:
        # Find the closest centroid for each data point
        closest_centroid = min(centroids, key=lambda centroid: euclidean_distance(point, centroid))
        clusters[closest_centroid].append(point)
    
    return clusters

# Function to update centroids by calculating the mean of points in each cluster
def update_centroids(clusters):
    new_centroids = []
    
    for points in clusters.values():
        # Calculate the new centroid as the mean of the points in the cluster
        new_centroid = tuple(sum(dim) / len(points) for dim in zip(*points))
        new_centroids.append(new_centroid)
    
    return new_centroids

# Function to plot the final clusters
def plot_final_clusters(clusters, centroids):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']  # Define some colors for clusters
    plt.figure()
    
    # Plot each cluster
    for i, (centroid, points) in enumerate(clusters.items()):
        points = list(zip(*points))  # Unzipping points to get x and y coordinates separately
        plt.scatter(points[0], points[1], color=colors[i % len(colors)], label=f'Cluster {i+1}')
        plt.scatter(*centroid, color='k', marker='x', s=200, label=f'Centroid {i+1}')

    plt.title(f'Final K-Means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

# Main K-Means algorithm function (no plotting during iterations, only final plot)
def k_means(data, centroids, max_iterations=100):
    for _ in range(max_iterations):
        # Assign points to clusters based on the nearest centroid
        clusters = assign_clusters(data, centroids)
        
        # Update the centroids by calculating the mean of each cluster
        new_centroids = update_centroids(clusters)
        
        # If centroids don't change, the algorithm has converged
        if set(new_centroids) == set(centroids):
            break
        
        # Update centroids for the next iteration
        centroids = new_centroids
    
    # After convergence, plot the final clusters
    plot_final_clusters(clusters, centroids)
    
    return centroids, clusters

# Function to get input from the user
def get_user_input():
    data = []
    
    # Get the number of data points from the user
    num_points = int(input("Enter the number of data points: "))
    
    # Get each data point from the user
    for i in range(num_points):
        point = tuple(map(float, input(f"Enter point {i+1} (format: x, y): ").split(',')))
        data.append(point)
    
    # Get the number of clusters (K) from the user
    k = int(input("Enter the value of K (number of clusters): "))
    
    # Get the initial centroids for each cluster from the user
    centroids = []
    for i in range(k):
        centroid = tuple(map(float, input(f"Enter initial centroid {i+1} (format: x, y): ").split(',')))
        centroids.append(centroid)
    
    return data, centroids

# Main function to run the program
def main():
    # Get data points and user-defined initial centroids from the user
    data, centroids = get_user_input()
    
    # Run the K-Means algorithm using the user-provided centroids
    final_centroids, clusters = k_means(data, centroids)
    
    # Output the final centroids and clusters
    print("\nFinal Centroids:", final_centroids)
    print("\nClusters:")
    for centroid, points in clusters.items():
        print(f"Centroid {centroid}: {points}")

# Entry point of the program
if __name__ == "__main__":
    main()
