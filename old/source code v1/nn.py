# %%
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

def init_layer_parm(nn_archit,seed=100):
    np.random.seed(seed)
    n_lyr = len(nn_archit)
    parm = {}

    for idx,lyr in enumerate(nn_archit):
        lyr_idx = idx+1
        lyr_input_sz = lyr['input_dim']
        lyr_output_sz = lyr['output_dim']

        parm['w'+str(lyr_idx)] = np.random.rand(lyr_output_sz,lyr_input_sz)
        parm['b'+str(lyr_idx)] = np.random.rand(lyr_output_sz,1)

    return parm

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
        x = x-np.max(x)+300
    return np.exp(x) / np.exp(x).sum()

def softmax_backward(dA, Z):
    sig = softmax(Z)
    return dA * sig * (1 - sig)

def single_lyr_forward(a_pre,w_now,b_now,activation = 'relu'):
    # print('w_n',w_now.shape,',a_p',a_pre.shape,',b_n',b_now.shape)
    z_now = np.dot(w_now,a_pre)+b_now
    # print(z_now[0])

    if activation=='relu':
        activation_func = relu
    elif activation=='sigmoid':
        activation_func = sigmoid
    elif activation=='softmax':
        activation_func = softmax
    else:
        raise Exception('Non-supported activation function')

    return activation_func(z_now),z_now

def full_forward(x,parm,nn_archit):
    memory = {}
    a_now = x
    # print('forward')
    for idx,lyr in enumerate(nn_archit):
        lyr_idx = idx+1
        a_pre = a_now
        # print('layer',lyr_idx)

        activ = lyr['activation']
        w_now = parm['w'+str(lyr_idx)]
        b_now = parm['b'+str(lyr_idx)]

        a_now,z_now = single_lyr_forward(a_now,w_now,b_now,activ)

        memory['a'+str(idx)] = a_pre
        memory['z'+str(lyr_idx)] = z_now
        # print(a_now[0])
        # print('a_n',a_now.shape)
        # print(memory.keys())

    return a_now,memory

def get_loss(pred_y,real_y): # cross entropy
    
    m = real_y.shape[1]
    # loss = -1/m *( np.dot( pred_y,np.log(real_y).T ) + np.dot( 1-pred_y,np.log(1-real_y).T ) )
    # print(np.squeeze(loss))
    # return np.squeeze(loss)
    # loss = 0
    # for i in range(len(real_y)):
    #     p = pred_y[i]
    #     y = real_y[i]
    #     # print(p)
    #     # log_likelihood = -np.log(p[range(m),y])
    #     log_likelihood = [-np.log(p[j]) for j in range (len(y))]
    #     # print(log_likelihood)
    #     l = np.sum(log_likelihood) / m
    #     # print(l)
    #     loss += l
    # loss /= len(real_y)
    loss = (((real_y-pred_y)**2).mean())
    # print(loss)
    return loss


def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_accuracy_value(Y_hat, Y):
    # Y_hat_ = convert_prob_into_class(Y_hat)
    # return (Y_hat_ == Y).all(axis=0).mean()
    Y_hat = np.transpose(Y_hat)
    Y = np.transpose(Y)
    count = 0
    for i in range(len(Y)):
        y=Y[i].tolist()
        y_h=Y_hat[i].tolist()
        if y.index(max(y))==y_h.index(max(y_h)):
            count+=1
    acc = count/len(Y)
    return acc

def single_lyr_backward(da_now,w_now,b_now,z_now,a_pre,activation='relu'):
    # a_pre: 向後傳播的error
    m = a_pre.shape[1]# number of examples 這層有幾個neuron

    if activation=='relu':
        backward_activation_func = relu_backward
    elif activation=='sigmoid':
        backward_activation_func = sigmoid_backward
    elif activation=='softmax':
        backward_activation_func = softmax_backward
    else:
        raise Exception('Non-supported activation function')

    dz_now = backward_activation_func(da_now,z_now)
    # print('w_n',w_now.shape,'z_n',z_now.shape,',da_n',da_now.shape,',dz_n',dz_now.shape,',a_p',a_pre.shape)
    dw_now = np.dot(dz_now,a_pre.T)/m
    db_now = np.sum(dz_now,axis=1,keepdims=True)/m
    da_pre = np.dot(w_now.T,dz_now)
    # print('dz_now',dz_now)
    # print('da_now',da_now)
    # print('z_now',z_now)
    # print('w_now',w_now)
    # print('da_pre',da_pre)
    # print('da_p',da_pre.shape,',dw_n',dw_now.shape,',db_n',db_now.shape)

    return da_pre,dw_now,db_now

# %%
def full_backward(Y_hat, Y, memory, param, nn_archi):
    grads = {}
    # print('backward')
    # print(Y.shape)
    m = Y.shape[1]# number of examples
    # a hack ensuring the same shape of the prediction vector and labels vector
    Y = Y.reshape(Y_hat.shape)
    
    # initiation of gradient descent algorithm
    # da_pre = - (np.divide(Y, Y_hat+0.001) - np.divide(1 - Y, 1 - Y_hat)) #loss'(y)
    da_pre = -(Y- Y_hat) #loss'(y)
    # print(da_pre.shape)
    
    for lyr_idx_pre, lyr in reversed(list(enumerate(nn_archi))):
        # we number network layers from 1
        lyr_idx_now = lyr_idx_pre + 1
        # print('layer',lyr_idx_now)
        # extraction of the activation function for the current layer
        activ_function_now = lyr["activation"]
        
        da_now = da_pre
        # print(memory.keys())
        a_pre = memory["a" + str(lyr_idx_pre)]#a_pre給成a_now
        z_now = memory["z" + str(lyr_idx_now)]
        
        w_now = param["w" + str(lyr_idx_now)]
        b_now = param["b" + str(lyr_idx_now)]
        
        da_pre, dw_now, db_now = single_lyr_backward( da_now,w_now,b_now,z_now,a_pre,activ_function_now)
        # print('da_p',da_pre)
        # print('dw_n',dw_now)
        # print('db_n',db_now)
        grads["dw" + str(lyr_idx_now)] = dw_now
        grads["db" + str(lyr_idx_now)] = db_now
    # print(grads)
    return grads

# %%
def update(param, grads, nn_archi, lr):
    # print(param)
    # print(grads)

    # iteration over network layers
    for lyr_idx, lyr in enumerate(nn_archi, 1):
        # print('layer', lyr_idx)
        # print(param["w" + str(lyr_idx)].shape,grads["dw" + str(lyr_idx)].shape )
        param["w" + str(lyr_idx)] -= lr * grads["dw" + str(lyr_idx)]        
        param["b" + str(lyr_idx)] -= lr * grads["db" + str(lyr_idx)]

    return param

# %%
def train(X, Y, nn_archi, epochs, lr, batchsz,seed=100, verbose=False, callback=None):
    # initiation of neural net parameters
    params_values = init_layer_parm(nn_archi, seed=seed)
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
    validX = np.transpose(np.array(X[cut:]))
    validY = np.transpose(np.array(Y[cut:]))
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
            x = np.transpose(batch_x[j])
            y = np.transpose(batch_y[j])
            # step forward
            Y_hat, cashe = full_forward(x, params_values, nn_archi)
            # step backward - calculating gradient
            grads_values = full_backward(Y_hat, y, cashe, params_values, nn_archi)
            # updating model state
            params_values = update(params_values, grads_values, nn_archi, lr)
        
            # loss = get_loss(Y_hat, y)
            # loss_history.append(loss)
            # accuracy = get_accuracy_value(Y_hat, y)
            # accuracy_history.append(accuracy)

            # validY_hat, _ = full_forward(validX, params_values, nn_archi)
            # # calculating metrics and saving them in history
            # val_loss = get_loss(validY_hat, validY)
            # valid_loss_history.append(val_loss)
            # val_accuracy = get_accuracy_value(validY_hat, validY)
            # val_accuracy_history.append(val_accuracy)
            # break
        
        # params_history.append(params_values)
        p = deepcopy(params_values)
        params_history.append(p)

        Y_hat_all, _ = full_forward(np.transpose(X), params_values, nn_archi)
        Y_hat_all+=0.001
        loss = get_loss(Y_hat_all, np.transpose(Y))
        loss_history.append(loss)
        accuracy = get_accuracy_value(Y_hat_all, np.transpose(Y))
        accuracy_history.append(accuracy)

        validY_hat, _ = full_forward(validX, params_values, nn_archi)
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
        
        history = [loss_history,valid_loss_history,params_history,accuracy_history,val_accuracy_history]
            
    return params_values,history
# %%
if __name__ == '__main__':

    dataset,correct_y = get_dataset()
    onehot_y = y_one_hot(correct_y)
    balance_data,balance_y = get_balance_data(dataset,correct_y,onehot_y)
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
    output_sz = len(set(correct_y))
    print(input_sz,hidden_sz,output_sz)

    lr=0.01
    epoch=100
    batchsz=100
    seed = 50
    acti = 'sigmoid'

    NN_ARCHITECTURE = [
        {"input_dim": input_sz, "output_dim": hidden_sz, "activation": acti},
        {"input_dim": hidden_sz, "output_dim": output_sz, "activation": "softmax"}
    ]

    # Training
    params_values,history = train(trainData, trainY, NN_ARCHITECTURE, epochs= epoch, lr= lr,batchsz=batchsz,seed=seed,verbose=True)
    # print(params_values)
    # Prediction
    Y_test_hat, _ = full_forward(np.transpose(testData), params_values, NN_ARCHITECTURE)
    # print(np.transpose(Y_test_hat))
    print(Y_test_hat)
    print(testY)

    # acc = get_accuracy_value(Y_test_hat, np.transpose(testY.reshape((testY.shape[0], 1))))
    acc = get_accuracy_value(Y_test_hat,  np.transpose(testY))
    print(acc)


    # plot
    loss_history = history[0]
    valid_loss_history = history[1]
    params_history = history[2]
    acc_history = history[3]
    valid_acc_history = history[4]

    # plot loss
    plt.plot(loss_history, label='train loss - lr='+str(lr))
    plt.plot(valid_loss_history, label='validation loss - lr='+str(lr))
    plt.legend()
    plt.savefig('loss-lr='+str(lr)+acti+'.png')
    plt.clf()

    # plot acc
    plt.plot(acc_history, label='train accuracy - lr='+str(lr))
    plt.plot(valid_acc_history, label='validation accuracy - lr='+str(lr))
    plt.legend()
    plt.savefig('acc-lr='+str(lr)+acti+'.png')
    plt.clf()

    #plot weights
    total_w = [] #20個
    for i in range(len(params_history)):
        now_all_w = (params_history[i]['b2']).reshape(-1)
        # print(now_all_w.shape)
        total_w.append(now_all_w.tolist())
    total_w = np.transpose(np.array(total_w))
    # print(total_w.shape)
    for w in total_w:
        plt.plot(w)
    plt.savefig('bias2-lr='+str(lr)+acti+'.png')
    plt.clf()

    total_w = []
    for i in range(len(params_history)):
        now_all_w = (params_history[i]['b1']).reshape(-1)
        # print(now_all_w.shape)
        total_w.append(now_all_w.tolist())
    total_w = np.transpose(np.array(total_w))
    # print(total_w.shape)
    for w in total_w:
        plt.plot(w)
    plt.savefig('bias1-lr='+str(lr)+acti+'.png')
    plt.clf()

    total_w = []
    for i in range(len(params_history)):
        now_all_w = (params_history[i]['w1']).reshape(-1)
        # print(now_all_w.shape)
        total_w.append(now_all_w.tolist())
    total_w = np.transpose(np.array(total_w))
    # print(total_w.shape)
    for w in total_w:
        plt.plot(w)
    plt.savefig('weights1-lr='+str(lr)+acti+'.png')
    plt.clf()

    total_w = []
    for i in range(len(params_history)):
        now_all_w = (params_history[i]['w2']).reshape(-1)
        # print(now_all_w.shape)
        total_w.append(now_all_w.tolist())
    total_w = np.transpose(np.array(total_w))
    # print(total_w.shape)
    for w in total_w:
        plt.plot(w)
    plt.savefig('weights2-lr='+str(lr)+acti+'.png')
    plt.clf()
    # print(w)