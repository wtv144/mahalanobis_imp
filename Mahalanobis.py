import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import erf 
from math import sqrt, exp

#x is the first coordinate, y is the second coordinate, n_clusters is the numebr of cluster
#ccov is an array containing all the cluster covariances
#cmeans is an array that contains all the cluster means
#cluster_p is the probability associated with each cluster
def mahalanobis_d(x,y,n_clusters, ccov, cmeans,cluster_p):
    v = np.array(x-y)
    Gnum = 0
    Gden = 0
    for i in range(0,n_clusters):
        ck = np.linalg.inv(ccov[i])
        u = np.array(cmeans[i] - y)
        val = ck*cluster_p[i]
        b2 =  1/(v.T @ ck @v)
        a = b2 * v.T @ ck @ u
        Z = u.T @ ck @ u - b2 * (v.T @ ck @ u)**2
        pxk = sqrt(np.pi*b2/2)*exp(-Z/2) * (erf((1-a)/sqrt(2*b2)) - erf(-a/sqrt(2*b2)))
        Gnum+= val*pxk 
        Gden += cluster_p[i] * pxk

    G = Gnum/Gden
    mdist = sqrt(v.T @ G @ v)
    print( "Mahalanobis distance between " + str(x) + " and "+str(y) + " is "+ str(mdist) )
    return mdist 


def main():
    #create the data,
    #cluster 1
    means1 = [1,1]
    cov1 = [[10,0],[0,2]]
    samples1 = np.random.multivariate_normal(means1,cov1,200)
    means2 = [8,4]
    cov2 = [[6,0],[0,4]]
    samples2 = np.random.multivariate_normal(means2,cov2,200)
    means3 = [5,15]
    cov3 = [[7,0],[0,3]]
    samples3 = np.random.multivariate_normal(means3,cov3,200)
    n_clusters = 3
    clust_cov  = np.array([cov1, cov2, cov3]).astype(float)
    clust_means = np.array([means1, means2, means3]).astype(float)
    clustp = [1/2,1/4,1/4]
    data = np.concatenate((samples1, samples2, samples3) ,axis =0).astype(float)
    np.random.shuffle(data)
    x1 = np.array([3 ,2])
    x2 = np.array([-1 , 4])

    x = mahalanobis_d(x1,x2,n_clusters,clust_cov,clust_means,clustp)

    #Test 2
    means = np.array([[0,0]]).astype(float)
    covs =  np.array([[[10,4],[2,8]]]).astype(float)
    xi = np.array([10,0]).astype(float)
    xj = np.array([-4,2]).astype(float)
    mahalanobis_d(xi,xj,1,covs,means,[1])

if __name__ == '__main__':
    main()