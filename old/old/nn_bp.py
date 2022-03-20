# %%
from os import error
import random
import numpy as np
import math
import pandas as pd
from pandas.core.frame import DataFrame
# %%
filename = 'bank-additional-full.csv'
data = pd.read_csv(filename,sep=';')
data_dum = pd.get_dummies(data)
# %%
def init_network(input_sz,hidden_sz,output_sz):
    network = []
    hidden_lyr = []
    for i in range(hidden_sz):
        tmp = {'w':[]}
        for j in range(input_sz+1):
            tmp['w'].append(random.random())
        hidden_lyr.append(tmp)
    network.append(hidden_lyr)

    output_lyr = []
    for i in range(output_sz):
        tmp = {'w':[]}
        for j in range(hidden_sz+1):
            tmp['w'].append(random.random())
        output_lyr.append(tmp)    
    network.append(output_lyr)

    return network

# network = init_network(2,1,2)
# for lyr in network:
#     print(lyr)

# %%
def cal_neuron(weights,inputs):
    ans = weights[-1] #bias
    for i in range(len(weights)-1):
        ans += weights[i]*inputs[i]
    return ans

def sigmoid_func(x):
    return 1/(1+math.exp(x*-1))

def forward(network, datas):
    inputs = datas
    for lyr in network:
        new_input = []
        for neuron in lyr:
            v = cal_neuron(neuron['w'],inputs)
            neuron['out'] = sigmoid_func(v)
            new_input.append(neuron['out'])
        inputs = new_input
    return inputs

# network = init_network(2,1,2)
# row = [1, 0, None]
# output = forward(network, row)
# print(output)

# %%
def sigmoid_derivative(x):
    return x*(1-x)

def backward(network,correct_y):
    for i in reversed(range(len(network))):
        lyr = network[i]
        error = []
        if i == len(network)-1: #output layer
            for j in range(len(lyr)):
                neuron = lyr[j]
                error.append(neuron['out']-correct_y[j])
        else:
            for j in range(len(lyr)):
                e = 0
                for neuron in network[i+1]:
                    e += neuron['w'][j]*neuron['delta']
                error.append(e)
        for j in range(len(lyr)):
            neuron = lyr[j]
            neuron['delta'] = error[j]*sigmoid_derivative(neuron['out'])

# network = init_network(2,1,2)
# row = [1, 0, None]
# output = forward(network, row)
# expected = [0, 1]
# backward(network, expected)
# for layer in network:
# 	print(layer)       


# %%
def update_w(network,datas,lr):
    for i in range(len(network)):
        inputs = datas[:-1] # output layer
        if i !=0:# hidden layer
            inputs = []
            for neuron in network[i-1]:
                inputs.append(neuron['out'])
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['w'][j] += -1*lr*neuron['delta']*inputs[j]
            neuron['w'][-1] += -1*lr*neuron['delta']

def train(network,train_data,lr,n_epoch,n_out):
    for epoch in range(n_epoch):
        total_e = 0
        for data in train_data:
            out = forward(network,data)
            correct_y = []
            for i in range(n_out):
                correct_y.append(0)
            correct_y[data[-1]]=1
            e = []
            for i in range(len(correct_y)):
                tmp = 0.5*((correct_y[i]-out[i])**2)
                e.append(tmp)
            total_e += sum(e)
            backward(network,correct_y)
            update_w(network,data,lr)
        print('epoch=',epoch,',lr=',lr,'error=',total_e)

# %%
def predict(network, row):
	outputs = forward(network, row)
	return outputs.index(max(outputs))

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
for i in range(len(correct_y)):
    dataset[i].append(correct_y[i])

print(dataset[0])

# %%
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = init_network(n_inputs, 10, n_outputs)
train(network, dataset, 0.5, 10, n_outputs)
# for layer in network:
# 	print(layer)
# %%
y_pred = []
for data in dataset:
    prediction = predict(network, data)
    y_pred.append(prediction)
# %%
