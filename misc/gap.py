import numpy as np
from numpy import zeros
import random
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering as ag
import scipy.spatial.distance
dst = scipy.spatial.distance.euclidean
import scipy.cluster.vq
import misc.utils as utils

def agclust(data,k):
    a = ag(n_clusters=k)
    a.fit(data)
    lbs = a.labels_
    cids = []
    kmc = np.zeros([k,data.shape[1]])
    for i in range(k):
        lst = np.where(lbs==i)[0]
        kmc[i,:] = np.mean(data[lst,:],axis=0)
        cids.append(lst)    
    return kmc,lbs
    
    
def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 
def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)
    
def Wk(mu, clusters):
    K = len(mu)
    return sum([np.linalg.norm(mu[i]-c)**2/(2*len(c)) \
               for i in range(K) for c in clusters[i]])
def dist(data,K):
    kmc,kml = agclust(data,K)
#    kmc,kml = scipy.cluster.vq.kmeans2(data, K)
    disp = sum([dst(data[m,:],kmc[kml[m],:]) for m in range(data.shape[0])])
    return disp

def bounding_box(X):
#    xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
#    ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    return (xmin,xmax)
 
def gap_statistic(X,B=10,kMin=1,kMax=50):
    (xmin,xmax) = bounding_box(X)
    # Dispersion for real distribution
#    ks = range(1,10)
    Wks = []
    Wkbs = []
    gaps = []
    sks = []
    flag = False
    indk = 0
    k = kMin
    ks = []
    while not flag and k<=kMax:
        print 'boostraping for k=',str(k),
        Wk = np.log(dist(X,k))
        BWkbs = zeros(B)
        for i in range(B):
            print '+',
            Xb = np.random.uniform(low=xmin,high=xmax,size=X.shape)            
            BWkbs[i] = np.log(dist(Xb,k))
        print 'done'
        Wkb = sum(BWkbs)/B
        sk = np.sqrt(sum((BWkbs-Wkb)**2)/B)
        sk = sk*np.sqrt(1+1/B)
        gap = Wkb - Wk
        Wkbs.append(Wkb)
        Wks.append(Wk)
        sks.append(sk)
        gaps.append(gap)
        ks.append(k)
        if indk>0:
            gapinc = gaps[indk-1] - (gaps[indk]-sks[indk])
            if (gapinc>=0):
                print 'GapInc=%.4f, k=%d is a good cluster' % (gapinc,k-1)
                flag = True
            else:
                print 'GapInc=%.4f, try k=%d' % (gapinc,k)
        indk = indk + 1
        k = k + 1
    if not flag:
        print 'kMax=%d still too small for a good cluster. Try larger one' % (kMax)
    return(ks, Wks, Wkbs, sks)

def init_board_gauss(N, k):
    n = float(N)/k
    X = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(0.05,0.05)
        x = []
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
        X.extend(x)
    X = np.array(X)[:N]
    return X



def analyze_gap_result(B,PLOT=False):
    ks,logWks,logWkbs,sk = utils.loadobj('temp/gapstats_B'+str(B)+'.pkl')    
    gaps = np.asarray(logWkbs) - np.asarray(logWks)
    gapinc = gaps[:-1]-gaps[1:]+sk[1:]
    lst = np.where(gapinc>=0)[0]
    best_cluster_number = ks[lst[0]]
    print 'best cluster number is:', ks[lst[0]]
    if PLOT:
        plt.plot(ks[:-1],gapinc,'-o')
        plt.plot([min(ks),max(ks)],[0,0],'k--')
        plt.xlabel('Cluster number')
        plt.ylabel('Decision bound (>=0)')
        plt.title('BS='+str(B)+', Best Cluster Number = ' + str(ks[lst[0]]))
        plt.grid()    
    return best_cluster_number

def fit_gap_stats(z,bootstraps=1000,kMin=1,kMax=50):
    print 'Clustering for matrix observation x feature: ', z.shape
    B = bootstraps
    gap_result = gap_statistic(z,B,kMin,kMax)
    utils.saveobj('temp/gapstats_B'+str(B)+'.pkl',gap_result)
    best_cluster_number = analyze_gap_result(B)
    return best_cluster_number
    