# coding=utf8
"""
Graph-Based Semi-Supervised Learning (GBSSL) implementation.

"""

# Authors: Yuto Yamaguchi <yamaguchi.yuto@aist.go.jp>
# Lisence: MIT

import numpy as np
from scipy import sparse
from abc import ABCMeta, abstractmethod

class Base():
    __metaclass__ = ABCMeta
    def __init__(self,graph,max_iter=30):
        self.max_iter = max_iter
        self.graph = graph

    @abstractmethod
    def _build_propagation_matrix(self):
        raise NotImplementedError("Propagation matrix construction must be implemented to fit a model.")

    @abstractmethod
    def _build_base_matrix(self):
        raise NotImplementedError("Base matrix construction must be implemented to fit a model.")

    def fit(self,x,y):
        """Fit a graph-based semi-supervised learning model

        All the input data is provided array X (labeled samples only)
        and corresponding label array y.

        Parameters
        ----------
        x : array_like, shape = [n_labeled_samples]
            Node IDs of labeled samples
        y : array_like, shape = [n_labeled_samples]
            Label IDs of labeled samples

        Returns
        -------
        self : returns an instance of self.
        """
        self.x_ = x
        self.y_ = y

        n_samples = self.graph.shape[0]
        n_classes = self.y_.max()+1
        self.F_ = np.zeros((n_samples,n_classes))

        P = self._build_propagation_matrix()
        B = self._build_base_matrix()

        remaining_iter = self.max_iter
        while remaining_iter > 0:
            self.F_ = P.dot(self.F_) + B
            remaining_iter -= 1

        return self

    def predict(self,x):
        """Performs prediction based on the fitted model

        Parameters
        ----------
        x : array_like, shape = [n_samples]
            Node IDs

        Returns
        -------
        y : array_like, shape = [n_samples]
            Predictions for input node IDs
        """
        probas = self.predict_proba(x)
        return np.argmax(probas,axis=1)

    def predict_proba(self,x):
        """Predict probability for each possible label

        Parameters
        ----------
        x : array_like, shape = [n_samples]
            Node IDs

        Returns
        -------
        probabilities : array_like, shape = [n_samples, n_classes]
            Probability distributions across class labels
        """
        return (self.F_[x].T / np.sum(self.F_[x], axis=1)).T


class LGC(Base):
    """Local and Global Consistency (LGC) for GBSSL

    Parameters
    ----------
    alpha : float
      clamping factor
    max_iter : float
      maximum number of iterations allowed

    Attributes
    ----------
    x_ : array, shape = [n_samples]
        Input array of node IDs.

    Examples
    --------
    <<<

    References
    ----------
    Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004).
    Learning with local and global consistency.
    Advances in neural information processing systems, 16(16), 321-328.
    """

    def __init__(self,graph,alpha=0.99,max_iter=30):
        super(LGC, self).__init__(graph,max_iter=30)
        self.alpha=alpha

    def _build_propagation_matrix(self):
        """ LGC computes the normalized Laplacian as its propagation matrix"""
        D2 = np.sqrt(sparse.diags((1.0/(self.graph.sum(1))).T.tolist()[0],offsets=0))
        S = D2.dot(self.graph).dot(D2)
        return self.alpha*S

    def _build_base_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max()+1
        B = np.zeros((n_samples,n_classes))
        B[self.x_,self.y_] = 1
        return (1-self.alpha)*B

class HMN(Base):
    """Harmonic funcsion (HMN) for GBSSL

    Parameters
    ----------
    max_iter : float
      maximum number of iterations allowed

    Attributes
    ----------
    x_ : array, shape = [n_samples]
        Input array of node IDs.

    Examples
    --------
    <<<

    References
    ----------
    Zhu, X., Ghahramani, Z., & Lafferty, J. (2003, August).
    Semi-supervised learning using gaussian fields and harmonic functions.
    In ICML (Vol. 3, pp. 912-919).
    """

    def _build_propagation_matrix(self):
        D = sparse.diags((1.0/(self.graph.sum(1))).T.tolist()[0],offsets=0)
        P = D.dot(self.graph)
        n_samples = self.graph.shape[0]
        labeled = np.arange(n_samples)[self.x_]
        P[labeled] = 0
        return P

    def _build_base_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max()+1
        B = np.zeros((n_samples,n_classes))
        B[self.x_,self.y_] = 1
        return B

class PARW(Base):
    """Partially Absorbing Random Walk (PARW) for GBSSL

    Parameters
    ----------
    lamb: float (default=0.001)
      Absorbing parameter
    max_iter : float
      maximum number of iterations allowed

    Attributes
    ----------
    x_ : array, shape = [n_samples]
        Input array of node IDs.

    Examples
    --------
    <<<

    References
    ----------
    Wu, X. M., Li, Z., So, A. M., Wright, J., & Chang, S. F. (2012).
    Learning with partially absorbing random walks.
    In Advances in Neural Information Processing Systems (pp. 3077-3085).
    """
    def __init__(self,graph,lamb=1.0,max_iter=30):
        super(PARW, self).__init__(graph,max_iter=30)
        self.lamb=lamb

    def _build_propagation_matrix(self):
        d = np.array(self.graph.sum(1).T)[0]
        Z = sparse.diags(1.0 / (d+self.lamb))
        P = Z.dot(self.graph)
        return P

    def _build_base_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max()+1
        B = np.zeros((n_samples,n_classes))
        B[self.x_,self.y_] = 1
        d = np.array(self.graph.sum(1).T)[0]
        Z = sparse.diags(1.0 / (d+self.lamb))
        Lamb = sparse.diags(self.lamb,shape=(n_samples,n_samples))
        return Z.dot(Lamb).dot(B)
