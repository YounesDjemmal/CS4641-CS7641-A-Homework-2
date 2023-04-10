import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
from kmeans import KMeans

SIGMA_CONST = 1e-6
LOG_CONST = 1e-32

class GMM(object):
    def __init__(self, X, K, max_iters = 100): # No need to change
        """
        Args: 
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters
        
        self.N = self.points.shape[0]        #number of observations
        self.D = self.points.shape[1]        #number of features
        self.K = K                           #number of components/clusters

    #Helper function for you to implement
    def softmax(self, logit): # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        """
        exp_logit = np.exp(logit - np.max(logit,axis= -1, keepdims= True))
        
        return exp_logit/np.sum(exp_logit,axis = -1, keepdims= True)

        

    def logsumexp(self, logit): # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        """
        exp_logit = np.exp(logit - np.max(logit,axis= -1, keepdims= True))
        
        return np.log(np.sum(exp_logit,axis = -1, keepdims= True)) + np.max(logit,axis=-1,keepdims=True)
        

    #for undergraduate student
    def normalPDF(self, logit, mu_i, sigma_i): #[5pts]
        """
        Args: 
            logit: N x D numpy array
            mu_i: 1xD numpy array, the center for the ith gaussian.
            sigma_i: 1xDxD numpy array, the covariance matrix of the ith gaussian.  
        Return:
            pdf: 1xN numpy array, the probability distribution of N data for the ith gaussian
            
        Hint: 
            np.diagonal() should be handy.
        """
        N, D = np.shape(logit)
        if len(np.shape(sigma_i)) == 2:
            sigma_i = np.diagonal(sigma_i)
        else:
            sigma_i = sigma_i.reshape(1,-1)
            sigma_i = np.diagonal(sigma_i * sigma_i.T)
        pdf = np.ones((1, N))
        for i in range(D):
            exp = ((-2 * sigma_i[i] )**-1) * np.square(logit[:, i] - mu_i[i])
            pdf *= ((2 * np.pi * sigma_i[i])**-0.5) * np.exp(exp)
        return pdf.reshape((logit.shape[0],))      
    
    #for grad students
    def multinormalPDF(self, logits, mu_i, sigma_i):  #[5pts]
        """
        Args: 
            logit: N x D numpy array
            mu_i: 1xD numpy array, the center for the ith gaussian.
            sigma_i: 1xDxD numpy array, the covariance matrix of the ith gaussian.  
        Return:
            normal_pdf: 1xN numpy array, the probability distribution of N data for the ith gaussian
            
        Hint: 
            np.linalg.det() and np.linalg.inv() should be handy.
        """
        raise NotImplementedError
    
    
    def _init_components(self, **kwargs): # [5pts]
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case
        """
        pi = np.full((self.K,),1/self.K )
        sigma = np.zeros((self.K, self.D, self.D)) 
        mu = np.zeros((self.K, self.D))
        for k in range(self.K):
            mu[k] = self.points[int(np.random.uniform(0,self.K))]
            sigma[k] = np.eye(self.D)
        
        return pi, mu, sigma
    
    def _ll_joint(self, pi, mu, sigma, full_matrix = False, **kwargs): # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
            
        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        ll = np.zeros((self.K,self.N))
        for k in range(self.K):
            ll[k] = np.log(pi[k]+ 10**(-32)) + np.log(self.normalPDF(self.points, mu[k], sigma[k]) + 10**(-32))

        return ll.T


    def _E_step(self, pi, mu, sigma, **kwargs): # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            
        Hint: 
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above. 
        """
        ll = self._ll_joint(pi, mu, sigma)
        gamma = self.softmax(ll)
        return gamma
        

    def _M_step(self, gamma, full_matrix = False, **kwargs): # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian. 
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            
        Hint:  
            There are formulas in the slide and in the above description box.
        """
        N, D = np.shape(self.points)
        K = np.shape(gamma)[1]
        N_k = np.sum(gamma, axis=0)
        mu = np.zeros((self.K, self.D))

        
        pi = N_k / N
        mu = (np.matmul(gamma.T, self.points).T / N_k).T
        
        sigma = np.zeros((K, D, D))

        for k in range(K):
            temp = np.matmul((self.points - mu[k]).reshape((N, D, 1)), (self.points - mu[k]).reshape((N, 1, D)))
            sigma[k] = np.dot(np.transpose(temp, (1, 2, 0)), gamma[:,k])
            sigma[k] /= N_k[k]
            sigma[k] = sigma[k]*np.eye(self.D)

        return pi, mu, sigma

    
    def __call__(self, full_matrix = False, abs_tol=1e-16, rel_tol=1e-16, **kwargs): # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)       
        
        Hint: 
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters. 
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))
        
        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma)
            
            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)
            
            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)