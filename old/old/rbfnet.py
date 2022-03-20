# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %%
def rbfunc(x,c,sigma):
    dist = cal_dist(x,c)
    # print(dist)
    return np.exp(-1 / (2 * sigma**2) * (dist)**2)

def cal_dist(x,center):
    return (np.sum(x-center))**0.5

def kmeans(X, k):
    #randomly select k center
    tmp = np.random.choice(range(len(X)), size=k)
    clusters = []
    for i in tmp:
        clusters.append(X[i])
    clusters = np.array(clusters) 
    prevClusters = clusters.copy()

    stds = np.zeros(k)
    conv = False
    while not conv:
        dist = []
        closestCluster = []
        for x in X:
            tmp = []
            for c in clusters:
                d = cal_dist(x,c)
                tmp.append(d)
            idx = np.argmin(tmp)
            closestCluster.append(idx) # find the closest center
            dist.append(tmp) # distance to k center

        # update clusters by taking the mean of all of the points assigned to that cluster
        for i in range(k):
            pointsForCluster = []
            for j in range(len(closestCluster)):
                if closestCluster[j]==i:
                    pointsForCluster.append(X[j])
            # print(np.array(pointsForCluster).shape)
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)
            # print(clusters)
        # converge if clusters haven't moved
        conv = np.linalg.norm(clusters - prevClusters) < 1e-6
        prevClusters = clusters.copy()

    clustersNoP = []
    for i in range(k):
        pointsForCluster = []
        for j in range(len(closestCluster)):
            if closestCluster[j]==i:
                pointsForCluster.append(X[j])
        if len(pointsForCluster) < 2:
            # keep track of clusters with no points or 1 point
            clustersNoP.append(i)
            continue
        else:
            stds[i] = np.std(pointsForCluster)

    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersNoP) > 0:
        print('there are',len(clustersNoP), 'cluster with 0 or 1 point')
        pointsToAverage = []
        for i in range(k):
            if i not in clustersNoP:
                pointsForCluster = []
                for j in range(len(closestCluster)):
                    if closestCluster[j]==i:
                        pointsForCluster.append(X[j])
                pointsToAverage.append(pointsForCluster)
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersNoP] = np.mean(np.std(pointsToAverage))
    
    return clusters, stds


# %%
class RBFN(object):
    def __init__(self,k=3,lr=0.01,epochs=100,rbf=rbfunc,infer_std=True):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        # self.inferStds = infer_std
        self.w = np.random.randn(k)
        self.b = np.random.randn(1)
    def fit(self, X, y):
        self.centers, self.stds = kmeans(X, self.k)
        # print(self.centers, self.stds)
        
        for epoch in range(self.epochs):
            # break
            for i in range(len(X)):
                # forward
                tmp = []
                pre_t = 0
                for c, s, in zip(self.centers, self.stds):
                    t = self.rbf(X[i], c, s)
                    if np.isnan(t):
                        tmp.append(pre_t)
                    else:
                        tmp.append(t)
                        pre_t = t
                a = np.array(tmp)
                # print(a)
                cal_y = a.T.dot(self.w) + self.b
                e = ((y[i] - cal_y).flatten() ** 2) * 0.5
                # print(y[i],cal_y)
                print('error: {0:.5f}'.format(e[0]))
                
                # backward
                error = -(y[i] - cal_y).flatten()
                # print(error)
                
                # update
                self.w = self.w - self.lr * a * error
                self.b = self.b - self.lr * error
                # print(self.w,self.b)

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            tmp = []
            pre_t = 0
            for c, s, in zip(self.centers, self.stds):
                t = self.rbf(X[i], c, s)
                if np.isnan(t):
                    tmp.append(pre_t)
                else:
                    tmp.append(t)
                    pre_t = t
            a = np.array(tmp)
            cal_y = a.T.dot(self.w) + self.b
            y_pred.append(cal_y)
        return np.array(y_pred)

# %%

filename = 'bank-additional-full.csv'
data = pd.read_csv(filename,sep=';')
correct_y = []
for i in range(len(data)):
    if data[data.columns[-1]][i]=='yes':
        correct_y.append(1)
    else:
        correct_y.append(0)
data_dum = pd.get_dummies(data[data.columns[:-1]])
dataset = data_dum.values.tolist()
print(dataset[0])

rbfn = RBFN(k=3,lr=0.01,epochs=100)
rbfn.fit(dataset,correct_y)
y_pred = rbfn.predict(dataset)
# %%
