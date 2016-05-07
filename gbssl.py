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

    def _init_label_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max()+1
        return np.zeros((n_samples,n_classes))

    def _arrange_params(self):
        """Do nothing by default"""
        pass

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

        self._arrange_params()

        self.F_ = self._init_label_matrix()

        self.P_ = self._build_propagation_matrix()
        self.B_ = self._build_base_matrix()

        remaining_iter = self.max_iter
        while remaining_iter > 0:
            self.F_ = self._propagate()
            remaining_iter -= 1

        return self

    def _propagate(self):
        return self.P_.dot(self.F_) + self.B_

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
        P[self.x_] = 0
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

class MAD(Base):
    """Modified Adsorption (MAD) for GBSSL

    Parameters
    ----------
    mu : array, shape = [3] > 0 (default = [1.0, 0.5, 1.0])
      Define importance among inj, cont, and abnd
    beta : float
      Used to determine p_inj_, p_cont_ and p_abnd_
    max_iter : float
      maximum number of iterations allowed

    Attributes
    ----------
    x_ : array, shape = [n_samples]
        Input array of node IDs.
    p_inj_ : array, shape = [n_samples]
      Probability to inject
    p_cont_ : array, shape = [n_samples]
      Probability to continue random walk
    p_abnd_ : array, shape = [n_samples]
        defined as 1 - p_inj - p_cont

    Examples
    --------
    <<<

    References
    ----------
    Talukdar, P. P., & Crammer, K. (2009).
    New regularized algorithms for transductive learning.
    In Machine Learning and Knowledge Discovery in Databases (pp. 442-457). Springer Berlin Heidelberg.
    """
    def __init__(self,graph,mu=np.array([1.0,0.5,1.0]),beta=2.0,max_iter=30):
        super(MAD, self).__init__(graph,max_iter=30)
        self.mu = mu
        self.beta = beta

    def _init_label_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max()+1
        return np.zeros((n_samples,n_classes+1)) # including dummy label

    def _build_normalization_term(self):
        W = self.graph.T.multiply(sparse.csr_matrix(self.p_cont_)).T
        d = np.array(W.sum(1).T)[0]
        dT = np.array(W.sum(0))[0]
        return sparse.diags(1.0/(self.mu[0]*self.p_inj_ + self.mu[1]*(d+dT) + self.mu[2]))

    def _build_propagation_matrix(self):
        Z = self._build_normalization_term()
        W = self.graph.T.multiply(sparse.csr_matrix(self.p_cont_)).T
        WT = W.T
        return Z.dot(self.mu[1]*(W+WT))

    def _build_base_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max()+1
        B = np.zeros((n_samples,n_classes+1)) # including dummy label
        B[self.x_,self.y_] = 1
        Z = self._build_normalization_term()
        S = sparse.diags(self.p_inj_)
        R = np.zeros((n_samples,n_classes+1))
        R[:,-1] = self.p_abnd_
        return Z.dot(self.mu[0]*S.dot(B)+self.mu[2]*R)

    def _arrange_params(self):
        P = sparse.csr_matrix(self.graph / np.maximum(self.graph.sum(1),1))
        logP = P.copy()
        logP.data = np.log(logP.data)
        H = - np.array(P.multiply(logP).sum(1).T)[0]
        c = np.log(self.beta) / np.log(self.beta+np.exp(H))
        d = np.zeros(self.graph.shape[0])
        d[self.x_] = (1-c[self.x_]) * np.sqrt(H[self.x_])
        z = np.maximum(c+d,1)
        self.p_inj_ = d / z
        self.p_cont_ = c / z
        self.p_abnd_ = 1 - self.p_inj_ - self.p_cont_

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
        return (self.F_[x,:-1].T / np.sum(self.F_[x,:-1], axis=1)).T

class OMNIProp(Base):
    """OMNI-Prop for GBSSL

    Parameters
    ----------
    lamb : float > 0 (default = 1.0)
      Define importance between prior and evidence from neighbors
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
    Yamaguchi, Y., Faloutsos, C., & Kitagawa, H. (2015, February).
    OMNI-Prop: Seamless Node Classification on Arbitrary Label Correlation.
    In Twenty-Ninth AAAI Conference on Artificial Intelligence.
    """

    def __init__(self,graph,lamb=1.0,max_iter=30):
        super(OMNIProp,self).__init__(graph,max_iter)
        self.lamb = lamb

    def _build_propagation_matrix(self):
        d = np.array(self.graph.sum(1).T)[0]
        dT = np.array(self.graph.sum(0))[0]
        Q = (sparse.diags(1.0/(d+self.lamb)).dot(self.graph)).dot(sparse.diags(1.0/(dT+self.lamb)).dot(self.graph.T))
        Q[self.x_] = 0
        return Q

    def _build_base_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max()+1
        unlabeled = np.setdiff1d(np.arange(n_samples),self.x_)

        dU = np.array(self.graph[unlabeled].sum(1).T)[0]
        dT = np.array(self.graph.sum(0))[0]
        n_samples = self.graph.shape[0]
        r = sparse.diags(1.0/(dU+self.lamb)).dot(self.lamb*self.graph[unlabeled].dot(sparse.diags(1.0/(dT+self.lamb))).dot(np.ones(n_samples))+self.lamb)

        b = np.ones(n_classes) / float(n_classes)

        B = np.zeros((n_samples,n_classes))
        B[unlabeled] = np.outer(r,b)
        B[self.x_,self.y_] = 1
        return B

class CAMLP(Base):
    """Confidence-Aware Modulated Label Propagation (CAMLP) for GBSSL

    Parameters
    ----------
    beta : float > 0 (default = 0.1)
      Define importance between prior and evidence from neighbors
    H : array_like, shape = [n_classes, n_classes]
      Define affinities between labels
      if None, identity matrix is set
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
    Yamaguchi, Y., Faloutsos, C., & Kitagawa, H. (2016, May).
    CAMLP: Confidence-Aware Modulated Label Propagation.
    In SIAM International Conference on Data Mining.
    """

    def __init__(self,graph,beta=0.1,H=None,max_iter=30):
        super(CAMLP,self).__init__(graph,max_iter)
        self.beta=beta
        self.H=H

    def _arrange_params(self):
        if self.H == None:
            n_classes = self.y_.max()+1
            self.H = np.identity(n_classes)

    def _propagate(self):
        return self.P_.dot(self.F_).dot(self.H) + self.B_

    def _build_normalization_term(self):
        d = np.array(self.graph.sum(1).T)[0]
        return sparse.diags(1.0/(1.0+d*self.beta))

    def _build_propagation_matrix(self):
        Z = self._build_normalization_term()
        return Z.dot(self.beta*self.graph)

    def _build_base_matrix(self):
        n_samples = self.graph.shape[0]
        n_classes = self.y_.max()+1
        B = np.ones((n_samples,n_classes))/float(n_classes)
        B[self.x_] = 0
        B[self.x_,self.y_] = 1
        Z = self._build_normalization_term()
        return Z.dot(B)
