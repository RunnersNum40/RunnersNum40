import numpy as np

def J(mu, clusters):
    """Objective function of the clustering that should provide a method of determining quality of fit.
    Sum of the squared distances between all points and their centeroids."""
    return sum(sum(distance_sq(centroid, point) for point in cluster) for centroid, cluster in zip(mu, clusters))

def distance_sq(x_1, x_2):
    """Return the euclidan distance squared between points x_1 and x_2 in real space.
    Because sqrt is an increasing function we do not need to apply it to sort by distance"""
    assert x_1.shape==x_2.shape, "Points muct be of equal dimensions"
    return sum((d_1-d_2)**2 for d_1, d_2 in zip(x_1, x_2))

def k_means(k, x):
    """Preform k-means clustering on n data points x with k centeroids. Requires that k <= n and k is int"""
    assert k <= len(x), "k must be less than or equal to the number of training points"
    assert k > 0, "k must be > 0"
    # select k random training points as the beginning centroids
    index = np.random.choice(x.shape[0], k, replace=False)
    mu = list(map(np.array, x[index]))

    for _ in range(100):
        # create arrays to use as cluster assignment
        clusters = [[] for _ in mu]

        # assign points to centroids
        for point in x:
            # find the closest centroid to each point
            closest, distance = 0, np.inf
            for i, centroid in enumerate(mu):
                test_dist = distance_sq(centroid, point)
                if test_dist < distance:
                    distance = test_dist
                    closest = i
            # add each point to the cluster of the closest centroid
            clusters[closest].append(point)

        # move centroids to mean of their clusters
        clusters = list(map(np.array, clusters))
        new_mu = [np.mean(cluster, axis=0) for cluster in clusters]

        # if no change has occured the algorithm has converged
        if all(np.array_equal(new, old) for new, old in zip(new_mu, mu)):
            break
        else:
            # other than nans updted the centroids
            for i, n_i in enumerate(new_mu):
                if not np.any(np.isnan(n_i)):
                    mu[i] = n_i

    return np.array(mu), clusters

def optimize(k, x, n=None):
    """Test several trials of k-means and return the one with the lowest objective function"""
    if n is None: n = 2*(k+1)
    trials = []
    for _ in range(n):
        trials.append(k_means(k, x))

    objectives = [J(mu, clusters) for mu, clusters in trials]
    return trials[objectives.index(min(objectives))]

def find_k(x, max_k=10):
    """Test different values of k and return their objective function results"""
    trials = []
    for k in range(1, min(len(x)-1, max_k)):
        trials.append(optimize(k, x))

    objectives = [J(mu, clusters) for mu, clusters in trials]
    return objectives


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    def generate_cluster(x, y, n=10, scale=1):
        """Return a guassian cluster of size n centered on (x, y) in the form of a n*2 np array"""
        return np.random.normal(0, scale, size=(n, 2))+np.array([x, y])

    def plot_k_clusters(mu, clusters):
        """Plot clusters in different colors"""
        colors = list(mcolors.TABLEAU_COLORS.values())[:len(mu)]

        plt.scatter(mu[:,0], mu[:,1], c=colors, marker="+", s=100)
        for color, cluster in zip(colors, clusters):
            plt.scatter(cluster[:,0], cluster[:,1], c=color)
        plt.show()

    data = np.concatenate((generate_cluster(0, 0, 330, 2), 
                           generate_cluster(10, 0, 330, 2), 
                           generate_cluster(0, 10, 330, 2), 
                           generate_cluster(10, 10, 330, 2)))

    k = 4
    mu, clusters = optimize(k, data)

    plot_k_clusters(mu, clusters)

    # fits = find_k(data)
    # plt.plot(range(len(fits)), fits)
    # plt.show()