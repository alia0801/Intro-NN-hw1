# %%
from os import error
import numpy as np
import random
import math
from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
# %%
def get_dataset():
    filename = 'bank-additional-full.csv'
    data = pd.read_csv(filename,sep=';')
    correct_y = []
    for i in range(len(data)):
        if data[data.columns[-1]][i]=='yes':
            correct_y.append(1)
        else:
            correct_y.append(0)
    data_dum = pd.get_dummies(data[data.columns[:-1]])
    # dataset = data_dum.values.tolist()
    data_normal = pd.DataFrame(z_score_normalization(data_dum, data_dum.keys()))
    dataset = data_normal.values.tolist()
    # print(dataset[0])
    # for i in range(len(correct_y)):
    #     dataset[i].append(correct_y[i])
    return dataset,correct_y

def z_score_normalization(df, cols):

    train_set_normalized = df.copy()
    for col in cols:
        all_col_data = train_set_normalized[col].copy()
        # print(all_col_data)
        mu = all_col_data.mean()
        std = all_col_data.std()
        
        z_score_normalized = (all_col_data - mu) / std
        train_set_normalized[col] = z_score_normalized
    return train_set_normalized

def y_one_hot(y):
    cats = list(set(y))
    new_y = []
    for yyy in y:
        idx = cats.index(yyy)
        tmp = [0]*len(cats)
        tmp[idx] = 1
        # print(yyy,idx,tmp)
        new_y.append(tmp)
    return new_y

def splitData(data,y, trainPct):
    numTrainRows = int(len(data) * trainPct)
    numTestRows = len(data) - numTrainRows
    trainData = data[:numTrainRows]
    testData = data[numTrainRows:]
    trainY = y[:numTrainRows]
    testY = y[numTrainRows:]
    return trainData, testData,trainY,testY

def get_balance_data(dataset,correct_y,onehot_y):
    # dataset,correct_y = get_dataset()
    # print(len(dataset))
    max_count = min(correct_y.count(1),correct_y.count(0))
    count = 0
    count1 = 0
    count0 = 0
    balance_data0 = []
    balance_data1 = []
    balance_y0 = []
    balance_y1 = []
    while(count1<max_count or count0<max_count):
        if count%10000==0:
            print(count)
        y = correct_y[count]
        tmp = dataset[count]
        count+=1
        if y==0:
            count0+=1
            if count0>=max_count:
                continue
            else:
                balance_data0.append(tmp)
                balance_y0.append(onehot_y[count])
        elif y==1:
            count1+=1
            if count1>=max_count:
                continue
            else:
                balance_data1.append(tmp)
                balance_y1.append(onehot_y[count])
    balance_data = []
    balance_y = []
    for i in range(len(balance_data1)):
        balance_data.append(balance_data0[i])
        balance_data.append(balance_data1[i])
        balance_y.append(balance_y0[i])
        balance_y.append(balance_y1[i])
    # print(len(balance_data))
    return balance_data,balance_y

# %%

# activation function
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    # print(np.maximum(0,Z)[0])
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def softmax(x):
    if np.max(x)>500:
        x = x-np.max(x)
    return np.exp(x) / np.exp(x).sum()

def softmax_backward(dA, Z):
    sig = softmax(Z)
    return dA * sig * (1 - sig)

def cal_dist(x,center):
    return (np.sum(x-center))**0.5

def rbfunc(x,c,sigma):
    dist = cal_dist(x,c)
    # return np.exp(-1 / (2 * sigma**2) * (dist)**2)
    return np.sqrt(1+ ((dist)**2) / (2 * sigma**2) )

def get_loss(pred_y,real_y):
    # print(pred_y,real_y)
    loss = (((real_y-pred_y)**2).mean())
    # print(loss)
    return loss

def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()
    # Y_hat = np.transpose(Y_hat)
    # Y = np.transpose(Y)
    # count = 0
    # for i in range(len(Y)):
    #     y=Y[i].tolist()
    #     y_h=Y_hat[i].tolist()
    #     if y.index(max(y))==y_h.index(max(y_h)):
    #         count+=1
    # acc = count/len(Y)
    # return acc
# %%

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

def init_layer_parm(inputsz,kernalsz,outputsz,x,seed=100):
    np.random.seed(seed)
    w = np.random.rand(outputsz,kernalsz)
    # c = np.random.rand(kernalsz,inputsz)
    # sigma = np.random.rand(kernalsz)
    c,sigma = kmeans(x,kernalsz)
    params_values =  {}
    params_values['w']=w
    params_values['c']=c
    params_values['sigma']=sigma
    return params_values

def full_forward(X,params_values):
    w = params_values['w']
    # print('w',w.shape)
    all_kernal = []
    for x in X:
        kernal = []
        pre_k=0
        for i in range(len(params_values['c'])):
            c = params_values['c'][i]
            s = params_values['sigma'][i]
            # print('c',c.shape,'s',s.shape)
            k = rbfunc(x,c,s)
            if np.isnan(k):
                kernal.append(pre_k)
            else:
                kernal.append(k)
        all_kernal.append(kernal)
    all_kernal = np.array(all_kernal)#(kernalsz,batchsz)
    # print(all_kernal)
    # print(all_kernal.shape)
    v = np.dot(all_kernal,w.T)
    # output = []
    # for vvv in v:
    #     o = softmax(vvv)
    #     output.append(o)
    # output = np.array(output)
    # print(v)
    return v,params_values,all_kernal

def full_backward(Y_hat, Y, param,all_kernal):
    w = param['w'] #outputsz*kernalsz
    c = param['c'] #kernal size*input size
    s = param['sigma'] #kernal size*input size
    grads = {}
    Y = Y.reshape(Y_hat.shape) #outputsz*batchsz
    da_pre = -(Y- Y_hat)
    dw = []#output size*kernal size
    for i in range(w.shape[0]):#output size
        tmp=[]
        for j in range(w.shape[1]):#number of kernal
            sum = 0
            for k in range(Y.shape[1]):#batchsz
                error = Y[i][k]-Y_hat[i][k]
                kernal = all_kernal[j][k]
                sum+=error*kernal
            tmp.append(sum)
        dw.append(tmp)
    grads['dw'] = np.array(dw)

    return grads
    # dc=[]
    # for i in range(c.shape[1]):#input size
    #     tmp=[]
    #     for j in range(c.shape[0]):#number of kernal
    #         sum = 0
    #         for k in range(Y.shape[1]):#batchsz
    #             error = Y[k]-Y_hat[k]
    #             kernal = all_kernal[j][k]
    #             sig = s[j]
    #             weights = w[0][j]
    #             sum+=error*kernal*(1/sig/sig)*cal_dist()*weights
    #         tmp.append(sum)
    #     dw.append(tmp)

def update(param,grads,lr):
    # print(param["w"].shape,grads["dw"].shape)
    param["w"] += lr * grads["dw"]
    return param  

# %%
def train(X, Y, epochs, lr, batchsz,seed=100, verbose=False, callback=None):
    # initiation of neural net parameters
    params_values = init_layer_parm(input_sz,hidden_sz,output_sz,X,seed=seed)
    # print(params_values)
    # initiation of lists storing the history 
    # of metrics calculated during the learning process 
    loss_history = []
    valid_loss_history = []
    accuracy_history = []
    params_history = []
    val_accuracy_history = []

    X = X.tolist()
    Y = Y.tolist()

    cut = round(len(X)*0.8)
    validX = np.array(X[cut:])
    validY = np.array(Y[cut:])
    X = X[:cut]
    Y = Y[:cut]

    batch_x = []
    batch_y = []
    count = 0
    while True:
        if (count+1)*batchsz<len(X):
            tmp_x = np.array(X[count*batchsz:(count+1)*batchsz])
            tmp_y = np.array(Y[count*batchsz:(count+1)*batchsz])
            # print(count,tmp_x.shape,tmp_y.shape)
            batch_x.append(tmp_x)
            batch_y.append(tmp_y)
            count+=1
        else:
            tmp_x = np.array(X[count*batchsz:])
            tmp_y = np.array(Y[count*batchsz:])
            batch_x.append(tmp_x)
            batch_y.append(tmp_y)
            break
    print(len(batch_x),batch_x[0].shape)
    # performing calculations for subsequent iterations
    for i in range(epochs):
        # print('epoch',epochs)

        for j in range(len(batch_x)):
            # x = np.transpose(batch_x[j])
            # y = np.transpose(batch_y[j])
            x = batch_x[j]
            y = batch_y[j]
            # step forward
            Y_hat, params_values,all_kernal = full_forward(x, params_values)
            # step backward - calculating gradient
            grads_values = full_backward(Y_hat, y, params_values, all_kernal)
            # updating model state
            params_values = update(params_values, grads_values, lr)
        p = deepcopy(params_values)
        params_history.append(p)
        # print(params_history)
        # print(grads_values['dw'])
        Y_hat_all, _,_ = full_forward(X, params_values)
        Y_hat_all+=0.001
        loss = get_loss(Y_hat_all, np.transpose(Y))
        loss_history.append(loss)
        accuracy = get_accuracy_value(Y_hat_all, np.transpose(Y))
        accuracy_history.append(accuracy)

        validY_hat, _,_ = full_forward(validX, params_values)
        validY_hat+=0.001
        val_loss = get_loss(validY_hat, validY)
        valid_loss_history.append(val_loss)
        val_accuracy = get_accuracy_value(validY_hat, validY)
        val_accuracy_history.append(val_accuracy)


        if(i % 10 == 0):
            if(verbose):
                print("Iteration: {:05} - loss: {:.5f} - accuracy: {:.5f} - val-loss: {:.5f} - val-accuracy: {:.5f}".format(i, loss, accuracy,val_loss,val_accuracy))
                # print("Iteration: {:05} - loss: {:.5f} ".format(i, loss))
            if(callback is not None):
                callback(i, params_values)
        # print('epoch',i)
        # for p in params_history:
        #     print(p['w'])
        
    history = [loss_history,valid_loss_history,params_history,accuracy_history,val_accuracy_history]
    # print(len(params_history))
    return params_values,history
# %%
if __name__ == '__main__':
    dataset,correct_y = get_dataset()
    balance_data,balance_y = get_balance_data(dataset,correct_y,correct_y)
    trainPct = 0.9
    trainData, testData,trainY,testY = splitData(balance_data,balance_y,trainPct )
    trainData = np.array(trainData)
    trainY = np.array(trainY)
    testData = np.array(testData)
    testY = np.array(testY)
    print(trainData.shape,testData.shape,trainY.shape,testY.shape)
    # print(testY[0])

    input_sz = len(dataset[0])
    hidden_sz = 10
    output_sz = 1
    print(input_sz,hidden_sz,output_sz)

    lr=0.0003
    epoch=100
    batchsz=100
    seed = 50

    # Training
    params_values,history = train(trainData, trainY, epochs= epoch, lr= lr,batchsz=batchsz,seed=seed,verbose=True)
    # print(params_values)

    # Prediction
    Y_test_hat, _,_ = full_forward(testData, params_values)
    # print(np.transpose(Y_test_hat))
    # print(testY)

    acc = get_accuracy_value(Y_test_hat,  np.transpose(testY))
    print(acc)

    # plot
    loss_history = history[0]
    valid_loss_history = history[1]
    params_history = history[2]
    acc_history = history[3]
    valid_acc_history = history[4]
    plt.plot(loss_history, label='train loss - lr='+str(lr))
    plt.plot(valid_loss_history, label='validation loss - lr='+str(lr))
    plt.legend()
    plt.savefig('loss - lr='+str(lr)+'rbfn2.png')
    plt.clf()

    # plot acc
    plt.plot(acc_history, label='train accuracy - lr='+str(lr))
    plt.plot(valid_acc_history, label='validation accuracy - lr='+str(lr))
    plt.legend()
    plt.savefig('acc-lr='+str(lr)+'rbfn2.png')
    plt.clf()

    total_w = []
    for param in params_history:
        now_all_w = param['w'].reshape(-1)
        # print(now_all_w.shape)
        # print(now_all_w)
        total_w.append(now_all_w.tolist())
    total_w = np.transpose(np.array(total_w))
    # print(total_w.shape)
    # print(total_w[0])
    for w in total_w:
        plt.plot(w)
    plt.savefig('weights_rbfn2-lr='+str(lr)+'.png')
    plt.clf()