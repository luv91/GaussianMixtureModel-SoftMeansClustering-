EM Algorithm is given as:

These steps are explained in detail in python notebook: fittingGMMWithEMModel.py or fittingGMMWithEMModel with plots.ipynb
1. initialize parameters mean, covariance and weight

2. initialize log-likelihood (over mean, co-variance and wiehgt from step 1)

3. Give some iteration numbers (max_iter), say equal to 10 or 20, for which E and M-step will run. 

    3a. E-Step: compute responsibility (using initial weight, mean, covariance and datapoints)
    3b. M-Step: compute weights, means, covariances
    
    3c. compute log-likelihood.  (using  weight, mean, covariance calcualted in M-step and datapoints)
    3d. check for convergence (between old log-likelihood and nnew log-likelihood from step 3c).
        If converged break
    
    3e. If not broken, update old log likelihood with the new log likelihood value
    
# Question: how can we Test it
### Testing Phase:

1. assumption 1: we are taking three distributions/clusters
2. assumption2 : Each cluster is two dimensional data. 
3. So initially, each cluster will have initial mean, initial cov-arainces and initial weights
For example:

# Model parameters
1. ### init_means = 
	a. [   # 3 by 2, with 3 being 3 distributions/cluster, and 2 being data is 2-D. 
    b. [5, 0], # mean of cluster 1  # 2 dimension because each cluster is having 2 dimension
    c. [1, 1], # mean of cluster 2
    d. [0, 5]  # mean of cluster 3
]
2. ###init_covariances = 
	[ # 3 by 2 by 2 
    [[.5, 0.], [0, .5]], # covariance of cluster 1  # 2 by 2 because each cluster is 2-d, so two axis will interact with each other. 
    [[.92, .38], [.38, .91]], # covariance of cluster 2
    [[.5, 0.], [0, .5]]  # covariance of cluster 3
]
3. ### init_weights = 
	[1/4., 1/2., 1/4.]  # weights of each cluster

# Model Parameter
1. ### init_means
   | Cluster | Mean Vector |
   | ------- | ----------- |
   | 1       | `[5, 0]`    |
   | 2       | `[1, 1]`    |
   | 3       | `[0, 5]`    |

2. ### init_covariances
   | Cluster | Covariance Matrix |
   | ------- | ----------------- |
   | 1       | `[[.5, 0.],`      |
   |         | ` [0, .5]]`       |
   | 2       | `[[.92, .38],`    |
   |         | ` [.38, .91]]`    |
   | 3       | `[[.5, 0.],`      |
   |         | ` [0, .5]]`       |

3. ### init_weights
   | Weights      |
   | ------------ |
   | `1/4`        |
   | `1/2`        |
   | `1/4`        |

# Generating data. 

### Say we have to generate 100 datapoints.  How to generate?

1. Generate cluster number: we have 3 clusters, so randomly generate 3 numers between 0,1 and 2 (because 3 clusters) according to assigned weight. 
	means if weight of cluster 0 is 60%, so 60% of time number 0 will be generated. 
	this happens using: k = np.random.choice(len(weights), 1, p=weights)[0]
	
2. Once we have a cluster number, (out of 0,1 and 2 for three clusters), generate a datapoint x. How ?
	we want to generate a data point from the multivariate normal distribution. (multivariante because data is 2d, two axis (x1, x2); (red, green)
	
	-->  take the cluster center number, use the corersponding mean and covraiances of that cluster center number (provided initial data above)
	### Initial cluster:
	![Alt text](D:\computer science\Coding One Stop\NortheasternSubjects\GaussianMixtureModel-SoftMeansClustering-\Initial_Cluster.png "Optional title")
	--> run multivariate normal distribution to sample from it. x = np.random.multivariate_normal(means[k], covariances[k])
	
3. keep on appending the data point in list. 

### Now we have say 100 data points. generated.. they will form 3 separate clusters because they are geenreated from 3 separate means and co-variances. 

# Now we have to run the EM algorithm
  
  1. We have 100 datapoints information (say, we have generated 100 points)
  2. Choose 3 points out of 100 at random, and generate means, covariances and weightsfor those 3 points. 
  3. These three means, cov-raiances and weights represent that of a whole dataset. it is a simple iinitialization, so clusters can overlap initially. (for example if 2 out of 3 cluster have same/nearby mean)
  4. Now run EM algorithm on this. After some iterations, you will see that EM algorithm converged. (final cluster plot)






