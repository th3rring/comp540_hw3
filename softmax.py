import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

    J = 0.0
    grad = np.zeros_like(theta)
    m, dim = X.shape
    K = theta.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in J and the gradient in grad. If you are not              #
    # careful here, it is easy to run into numeric instability. Don't forget    #
    # the regularization term!                                                  #
    #############################################################################
    
    # Helper function for calculating probability.
    def calc_prob(i, category):
        prod = np.exp(theta.T@X[i] - np.amax(theta.T@X[i]))
        return prod[category]/np.sum(prod)
    
    # Calculate loss.
    for i in range(m):
        J += np.log(calc_prob(i, y[i]))
        
    J *= -1/m
    J += reg/(2*m) * np.sum(theta ** 2, (0,1))
    
    # Calculate gradient.
    ind = lambda y, k: int(y == k)
    for k in range(K):
        for i in range(m):
            grad[:, k] += X[i] * (ind(y[i], k) - calc_prob(i, k))
        grad[:, k] *= -1/m
        grad[:, k] += reg/m * theta[:, k]
        
            
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
    """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

    J = 0.0
    grad = np.zeros_like(theta)
    m, dim = X.shape

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in J and the gradient in grad. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization term!                                                      #
    #############################################################################
    
    preds_col = (X@theta).T
    exps_adj = np.exp(preds_col - np.amax(preds_col, axis=0))

    ind = np.zeros_like(preds_col)
    np.put_along_axis(ind, np.atleast_2d(y), 1, axis=0)

    J = -1/m * np.sum(np.log(np.take_along_axis(exps_adj, np.atleast_2d(y), axis=0)/np.sum(exps_adj, axis=0)))

    grad = -1/m * ((ind - exps_adj/np.sum(exps_adj, axis=0)) @ X).T
    grad += reg/m * theta

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return J, grad
