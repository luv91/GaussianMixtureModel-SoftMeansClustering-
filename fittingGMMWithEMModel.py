
# see equations in one-Note

import numpy as np                                             # dense matrices
import matplotlib.pyplot as plt                                # plotting
from scipy.stats import multivariate_normal                    # multivariate Gaussian distribution
import copy                                                    # deep copies

# image handling library
from PIL import Image
from io import BytesIO

#%% Step 1: Log-Likelihood calculation

# mathematical trick to avoid over and underflow: 
# https://math.stackexchange.com/questions/4480736/computing-the-log-of-a-sum-of-exponentials

def log_sum_exp(Z):
    """ Compute log(\sum_i exp(Z_i)) for some array Z."""
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))


""" 
Log likelihood formula for the gaussian mixture model, page 7 of pdf:
    https://www.cs.toronto.edu/~jlucas/teaching/csc411/lectures/lec15_16_handout.pdf
    
    The log likelihood quantifies the probability of observing a given set of 
    data under a particular setting of the parameters in our model.
    
    
    """
    
def loglikelihood(data, weights, means, covs):
    """ Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters. """
    
    #following page 7 for the overall equation: https://www.cs.toronto.edu/~jlucas/teaching/csc411/lectures/lec15_16_handout.pdf
    
    num_clusters = len(means)
    num_dim = len(data[0])
    
    ll = 0 
    
    # outer loop (summation over datapoints in log-likelihood)
    for d in data:
        
        # inner loop will calculate the probability of each data point belonging to 
        # each Gaussian component (each cluster/distribution that is combined to make the overall 
        # gaussian mixture model)
        
        contributionOfDataPointToEachCluster = np.zeros(num_clusters)
        
        for k in range(num_clusters):
            
            # check pdf page 61, for equations: https://www.cs.toronto.edu/~urtasun/courses/CSC411_Fall16/13_mog.pdf
            #(x-mu); one of the intermediate components
            dataPointMinusMeans = np.array(d) - means[k]
            
            # np.linalg.inv(covs[k]) ==> inverse of covaraince
            exponentInNormalDistributionOfDataPoint = np.dot(dataPointMinusMeans.T, 
                                                             np.dot(np.linalg.inv(covs[k]), dataPointMinusMeans))
            
            
            # # Compute loglikelihood contribution for this data point and this cluster
            
            contributionOfDataPointToEachCluster[k] += np.log(weights[k])
            
            contributionOfDataPointToEachCluster[k] -= 1/2* (num_dim * np.log(2*np.pi) + 
                                                             np.log(np.linalg.det(covs[k])) + 
                                                             exponentInNormalDistributionOfDataPoint)
            
        # increment loglikelihood contribution of this data point across all clusters
        ll += log_sum_exp(contributionOfDataPointToEachCluster)
        
    return ll
            
            
            
#%% Step 2: E-step: assign cluster responsibilities, given current parameters
        
"""
E-Step: 
The first step in the EM algorithm is to compute cluster responsibilities. Let rik denote the
 responsibility of cluster k for data point i.. 
 
 SciPy provides a convenient function multivariate_normal.pdf that computes 
 the likelihood of seeing a data point in a multivariate Gaussian distribution. The usage is :
     
     multivariate_normal.pdf([data point], mean=[mean vector], cov=[covariance matrix])
 
"""

def compute_responsibilities(data, weights, means, covariances):
    
    '''E-step: compute responsibilities, given the current parameters'''
    
    num_data = len(data)
    
    num_clusters = len(means)
    
    # responsibility is a matrix because it is defined as a responsibility of each data point i
    # towards cluster k
    resp = np.zeros((num_data, num_clusters))
    
    # Update resp matrix so that resp[i,k] is the responsibility of cluster k for data point i.
    # To compute likelihood of seeing data point i given cluster k, use multivariate_normal.pdf.

    for i in range(num_data):
        
        for k in range(num_clusters):
            
            resp[i,k] = weights[k] * multivariate_normal.pdf(data[i], mean=means[k], cov=covariances[k])
    
    
    # Add up responsibilities over each data point and normalize
    # in normalization addition happens over the clusters k, so axis 1
    row_sums = resp.sum(axis = 1)[: , np.newaxis]
    
    resp = resp/row_sums
    
    return resp

""" Checkpoint: testing compute_responsibilities function for error"""
"""
covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])
say I have two distributions distribution 1 and distributuion2
in each distribution i have two colors say red and green

so distribution 1 covariance will be: [sigmasquared(red),  sigma(red,green)
                                       sigma(green,red), sigmasquared(green)]
similar will be for distribution 2 and they are clubbed together

since at the end we are looking for a overall gaussian.. (combination of all distributions)

"""
resp = compute_responsibilities(data=np.array([[1.,2.],[-1.,-2.]]), weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])

if resp.shape==(2,2) and np.allclose(resp, np.array([[0.10512733, 0.89487267], [0.46468164, 0.53531836]])):
    print('Checkpoint passed!')
else:
    print("Error in Implementation")
    
#%% Step 3: M-step: Update parameters, given current cluster responsibilities; page 24 for equations: https://www.cs.toronto.edu/~jlucas/teaching/csc411/lectures/lec15_16_handout.pdf


"""
Once the cluster responsibilities are computed, we update the parameters 
(weights, means, and covariances) associated with the clusters.

part a: compute soft counts
part b: update weights
partc: update mean
part d: update co-variance


"""

"""
Computing soft counts. Before updating the parameters, we first compute what is
 known as "soft counts". The soft count of a cluster is the sum of all cluster responsibilities for that cluster
"""
def compute_soft_counts(resp):
    
    # this will be summed up over each data point. 
    # see the equation in One_note, part 3 or 
    #page-24, (https://www.cs.toronto.edu/~jlucas/teaching/csc411/lectures/lec15_16_handout.pdf)
    
    counts = np.sum(resp, axis = 0)   # dimension 1 by k
    
    return counts

"""
Commpute Weights: The cluster weights show us how much each cluster is represented over all data points.

"""

def compute_weights(counts):
    
    num_clusters = len(counts)
    weights = [0.]* num_clusters
    
    for k in range(num_clusters): # this loop is just to update the kth cluster and not summing over it
        
        # Update the weight for cluster k using the M-step update rule for the cluster weight
        
        weights[k] = counts[k]/np.sum(counts)   # np.sum(counts) (summed up over all cluster axis)
        
        #print("np.sum(counts)",np.sum(counts))
    return weights

""" Checkpoint: testing weights function for error"""

resp = compute_responsibilities(data=np.array([[1.,2.],[-1.,-2.],[0,0]]), weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])
counts = compute_soft_counts(resp)
weights = compute_weights(counts)



if np.allclose(weights, [0.27904865942515705, 0.720951340574843]):
    print('Checkpoint passed!')
else:
    print("Error in Implementation")

    

"""

Commpute Means: The mean of each cluster is set to the weighted average of
                all data points, weighted by the cluster responsibilities..
                
"""

def compute_means(data, resp, counts):
    
    num_clusters = len(counts)
    num_data = len(data)
    means = [np.zeros(len(data[0]))] * num_clusters
    
    for k in range(num_clusters):  # this loop is just to update the kth cluster and not summing over it
        
        # Update means for cluster k using the M-step update rule for the mean variables.
        
        weighted_sum = 0
        
        # check out the equation. https://www.cs.toronto.edu/~jlucas/teaching/csc411/lectures/lec15_16_handout.pdf
        # page 24
        for i in range(num_data):
            
            weighted_sum += data[i]* resp[i][k]
            
            
        means[k] = weighted_sum/counts[k]
        
    return means
    
""" Checkpoint: testing Means function for error"""

data_tmp = np.array([[1.,2.],[-1.,-2.]])
resp = compute_responsibilities(data=data_tmp, weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])
counts = compute_soft_counts(resp)
means = compute_means(data_tmp, resp, counts)

if np.allclose(means, np.array([[-0.6310085, -1.262017], [0.25140299, 0.50280599]])):
    print('Checkpoint passed!')
else:
    print("Error in Implementation")

"""

Commpute Co-variance: The covariance of each cluster is set to the weighted average of 
all outer products, weighted by the cluster responsibilities..
Look in page (24): https://www.cs.toronto.edu/~jlucas/teaching/csc411/lectures/lec15_16_handout.pdf
                
"""
def compute_covariances(data, resp, counts, means):
    num_clusters = len(counts)
    num_dim = len(data[0])
    num_data = len(data)
    covariances = [np.zeros((num_dim,num_dim))] * num_clusters
    
    for k in range(num_clusters):
        # Update covariances for cluster k using the M-step update rule for covariance variables.
        # This will assign the variable covariances[k] to be the estimate for \hat{\Sigma}_k.
        weighted_sum = np.zeros((num_dim, num_dim))
        for i in range(num_data):
            
            weighted_sum += resp[i][k]*np.outer(data[i] - means[k], data[i] - means[k])
        
        covariances[k] = weighted_sum / counts[k]

    return covariances
""" Checkpoint: testing Co-varaince function for error"""
data_tmp = np.array([[1.,2.],[-1.,-2.]])
resp = compute_responsibilities(data=data_tmp, weights=np.array([0.3, 0.7]),
                                means=[np.array([0.,0.]), np.array([1.,1.])],
                                covariances=[np.array([[1.5, 0.],[0.,2.5]]), np.array([[1.,1.],[1.,2.]])])
counts = compute_soft_counts(resp)
means = compute_means(data_tmp, resp, counts)
covariances = compute_covariances(data_tmp, resp, counts, means)

if np.allclose(covariances[0], np.array([[0.60182827, 1.20365655], [1.20365655, 2.4073131]])) and \
    np.allclose(covariances[1], np.array([[ 0.93679654, 1.87359307], [1.87359307, 3.74718614]])):
    print('Checkpoint passed!')
else:
    print("Error in Implementation")
    
#%% EM Algorithm

"""
1. initialize parameters mean, covariance and weight
2. initialize log-likelihood (over mean, co-variance and wiehgt from step 1)

3. Give some iteration numbers (max_iter), say equal to 10 or 20, for which E and M-step will run. 

    3a. E-Step: compute responsibility (using initial weight, mean, covariance and datapoints)
    3b. M-Step: compute weights, means, covariances
    
    3c. compute log-likelihood.  (using  weight, mean, covariance calcualted in M-step and datapoints)
    3d. check for convergence (between old log-likelihood and nnew log-likelihood from step 3c).
        If converged break
    
    3e. If not broken, update old log likelihood with the new log likelihood value
    
"""
def EM(data, init_means, init_covariances, init_weights, maxiter=1000, thresh=1e-4):
    
    # Make copies of initial parameters, which we will update during each iteration
    means = init_means[:]
    covariances = init_covariances[:]
    weights = init_weights[:]
    
    # Infer dimensions of dataset and the number of clusters
    num_data = len(data)
    num_dim = len(data[0])
    num_clusters = len(means)
    
    # Initialize some useful variables
    resp = np.zeros((num_data, num_clusters))
    ll = loglikelihood(data, weights, means, covariances)
    ll_trace = [ll]
    
    for it in range(maxiter):
        if it % 5 == 0:
            print("Iteration %s" % it)
        
        # E-step: compute responsibilities
        resp = compute_responsibilities(data, weights, means, covariances)

        # M-step
        # Compute the total responsibility assigned to each cluster, which will be useful when 
        # implementing M-steps below. In the lectures this is called N^{soft}
        counts = compute_soft_counts(resp)
        
        # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
        
        weights = compute_weights(counts)
        
        # Update means for cluster k using the M-step update rule for the mean variables.
        # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
        
        means = compute_means(data, resp, counts)
        
        # Update covariances for cluster k using the M-step update rule for covariance variables.
        # This will assign the variable covariances[k] to be the estimate for \hat{\Sigma}_k.
        
        covariances = compute_covariances(data, resp, counts, means)
        
        # Compute the loglikelihood at this iteration
       
        ll_latest = loglikelihood(data, weights, means, covariances)
        ll_trace.append(ll_latest)
        
        # Check for convergence in log-likelihood and store
        if (ll_latest - ll) < thresh and ll_latest > -np.inf:
            break
        ll = ll_latest
    
    if it % 5 != 0:
        print("Iteration %s" % it)
    
    out = {'weights': weights, 'means': means, 'covs': covariances, 'loglik': ll_trace, 'resp': resp}

    return out

""" Running EM Algorithm """
# =============================================================================
# np.random.seed(4)
# 
# # Initialization of parameters
# chosen = np.random.choice(len(data), 3, replace=False)
# initial_means = [data[x] for x in chosen]
# initial_covs = [np.cov(data, rowvar=0)] * 3
# initial_weights = [1/3.] * 3
# 
# # Run EM 
# results = EM(data, initial_means, initial_covs, initial_weights)
# =============================================================================
