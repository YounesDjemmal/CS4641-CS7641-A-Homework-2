'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


class KMeans(object):

    def __init__(self):  # No need to implement
        pass
 
    def _init_centers(self, points, K, **kwargs):  # [5 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        """
        n,d = np.shape(points)
        centers = np.zeros((K,d))
        for i in range(K) : 
            centers[i] = points[np.random.randint(0,n)]
        return centers
            

    def _update_assignment(self, centers, points):  # [10 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point

        Hint: You could call pairwise_dist() function.
        """
        dists = pairwise_dist(points,centers) 
        n,_ = np.shape(points)
        cluster_idx = np.zeros((n,1))
        cluster_idx = [np.argmin(dists[i]) for i in range(n)]
        cluster_idx = np.array(cluster_idx)
        return cluster_idx


    def _update_centers(self, old_centers, cluster_idx, points):  # [10 pts]
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, a new K x D numpy array, where K is the number of clusters, and D is the dimension.

        HINT: If you need to reduce the number of clusters when there are 0 points for a center, then do so.
        """
        K, D = np.shape(old_centers)
        centroids = np.copy(old_centers)
        cluster_idx = np.array(cluster_idx)
        i= 0
        counter = 0
        for k in range(K):
            assignment_k = np.argwhere(cluster_idx == k)
            average_K = 0
            if np.size(assignment_k) > 0:
                average_K = np.mean(points[assignment_k,:], axis= 0)
                centroids[i] = average_K[0]
                i+=1
            else:
                counter+=1
        for i in range(counter): 
            centroids = np.delete(centroids[-1])
        return centroids


    def _get_loss(self, centers, cluster_idx, points):  # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans.
        """
        K, D = np.shape(centers)
        n,_ = np.shape(points)
        loss = np.zeros(K)
        for i in range(n):
            loss[cluster_idx[i]] = loss[cluster_idx[i]] + (np.linalg.norm(centers[cluster_idx[i]] - points[i]))**2 
        return np.sum(loss)



    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if ((diff < abs_tol) and ((diff / prev_loss) < rel_tol)):
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))
        return cluster_idx, centers, loss

def find_optimal_num_clusters(data, max_K=15):  # [10 pts]
    np.random.seed(1)
    """Plots loss values for different number of clusters in K-Means

    Args:
        image: input image of shape(H, W, 3)
        max_K: number of clusters
    Return:
        losses: vector of loss values (also plot loss values against number of clusters but do not return this)
    """
    losses = np.empty((0,1))
    K = range(1,  max_K+1)
    kmeans = KMeans()
    for k in K:
        cluster_idx, centers, loss = kmeans.__call__(points = data,K = k, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False)
        losses = np.append(losses, [loss])
    plt.figure(figsize=(16,8))
    plt.plot(K, losses, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    return losses
    



def pairwise_dist(x, y):  # [5 pts]
    np.random.seed(1)
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
        dist: N x M array, where dist2[i, j] is the euclidean distance between
        x[i, :] and y[j, :]
    """
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)

