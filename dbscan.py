import numpy as np


class DBSCAN(object):
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset
    def fit(self):
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        You should iterate through each of your points one at a time and keep track of your points' cluster assignments.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        """
        visited = []
        c = 0
        cluster_idx = np.zeros((self.dataset.shape[0],))
        for i in range(0,self.dataset.shape[0]):
            if i not in visited:
                visited.append(i)
                neighbors = self.regionQuery(i)
                if neighbors.shape[0] <= self.minPts:
                    cluster_idx[i] = -1
                else:
                    c = c + 1
                    self.expandCluster(i,neighbors,c , cluster_idx, visited)
                    cluster_idx[i] = -1

        return cluster_idx



    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly

        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hint: np.concatenate(), np.unique(), and np.take() may be helpful here
        """
        for neighbor_i in neighborIndices:
            if not neighbor_i in visitedIndices:
                visitedIndices.append(neighbor_i)
                neighborIndices[neighbor_i] = self.regionQuery(neighbor_i)
                
                if len(neighborIndices[neighbor_i]) >= self.minPts:
                    self.expandCluster(neighbor_i, neighborIndices[neighbor_i], C, cluster_idx,visitedIndices)
                   
                else:
                    cluster_idx[neighbor_i] = C
                    




            
    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        indices = np.empty((0,))
        for i in range(0, len(self.dataset)):
            if (np.linalg.norm(self.dataset[pointIndex] - self.dataset[i]) < self.eps):
                indices = np.append(indices,i)


        return indices