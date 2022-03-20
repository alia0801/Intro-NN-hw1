import math
import numpy as np
import random
def sigmoid(x):
    return 1/(1+math.exp(x*-1))

def hypertan(x):
    if x < -20.0:
        return -1.0
    elif x > 20.0:
        return 1.0
    else:
        return math.tanh(x)

 
def softmax(oSums):
    result = np.zeros(shape=[len(oSums)], dtype=np.float32)
    m = max(oSums)
    divisor = 0.0
    for k in range(len(oSums)):
        divisor += math.exp(oSums[k] - m)
    for k in range(len(result)):
        result[k] =  math.exp(oSums[k] - m) / divisor
    return result
	

def totalWeights(nInput, nHidden, nOutput):
   tw = (nInput * nHidden) + (nHidden * nOutput) + nHidden + nOutput
   return tw


class NeuralNetwork:

    def __init__(self, numInput, numHidden, numOutput, seed):
        self.ni = numInput
        self.nh = numHidden
        self.no = numOutput
    
        self.iNodes = np.zeros(shape=[self.ni], dtype=np.float32)
        self.hNodes = np.zeros(shape=[self.nh], dtype=np.float32)
        self.oNodes = np.zeros(shape=[self.no], dtype=np.float32)
    
        self.ihWeights = np.zeros(shape=[self.ni,self.nh], dtype=np.float32)
        self.hoWeights = np.zeros(shape=[self.nh,self.no], dtype=np.float32)
    
        self.hBiases = np.zeros(shape=[self.nh], dtype=np.float32)
        self.oBiases = np.zeros(shape=[self.no], dtype=np.float32)
    
        self.rnd = random.Random(seed) # allows multiple instances
        self.initializeWeights()

    def setWeights(self, weights):
        if len(weights) != totalWeights(self.ni, self.nh, self.no):
            print("Warning: len(weights) error in setWeights()")	

        idx = 0
        for i in range(self.ni):
            for j in range(self.nh):
                self.ihWeights[i,j] = weights[idx]
                idx += 1
    
        for j in range(self.nh):
            self.hBiases[j] = weights[idx]
            idx += 1

        for j in range(self.nh):
            for k in range(self.no):
                self.hoWeights[j,k] = weights[idx]
                idx += 1
    
        for k in range(self.no):
            self.oBiases[k] = weights[idx]
            idx += 1
	  
    def getWeights(self):
        tw = totalWeights(self.ni, self.nh, self.no)
        result = np.zeros(shape=[tw], dtype=np.float32)
        idx = 0  # points into result
    
        for i in range(self.ni):
            for j in range(self.nh):
                result[idx] = self.ihWeights[i,j]
                idx += 1
    
        for j in range(self.nh):
            result[idx] = self.hBiases[j]
            idx += 1

        for j in range(self.nh):
            for k in range(self.no):
                result[idx] = self.hoWeights[j,k]
                idx += 1
    
        for k in range(self.no):
            result[idx] = self.oBiases[k]
            idx += 1
    
        return result

    def initializeWeights(self):
        numWts = totalWeights(self.ni, self.nh, self.no)
        wts = np.zeros(shape=[numWts], dtype=np.float32)
        lo = -0.01 
        hi = 0.01
        for idx in range(len(wts)):
            wts[idx] = (hi - lo) * self.rnd.random() + lo
        self.setWeights(wts)

    def computeOutputs(self, xValues):
        hSums = np.zeros(shape=[self.nh], dtype=np.float32)
        oSums = np.zeros(shape=[self.no], dtype=np.float32)

        for i in range(self.ni):
            self.iNodes[i] = xValues[i]

        for j in range(self.nh):
            for i in range(self.ni):
                hSums[j] += self.iNodes[i] * self.ihWeights[i,j]

        for j in range(self.nh):
            hSums[j] += self.hBiases[j]
    
        for j in range(self.nh):
            # self.hNodes[j] = hypertan(hSums[j])
            self.hNodes[j] = sigmoid(hSums[j])

        for k in range(self.no):
            for j in range(self.nh):
                oSums[k] += self.hNodes[j] * self.hoWeights[j,k]

        for k in range(self.no):
            oSums[k] += self.oBiases[k]
    
        softOut = softmax(oSums)
        for k in range(self.no):
            self.oNodes[k] = softOut[k]
    
        result = np.zeros(shape=self.no, dtype=np.float32)
        for k in range(self.no):
            result[k] = self.oNodes[k]
    
        return result    

    def trainBatch(self, trainData, maxEpochs, learnRate):
        # full batch with tanh + softmax & ms error
        # this version accumulates gradients instead of deltas
    
        hoGrads = np.zeros(shape=[self.nh, self.no], dtype=np.float32)  # hidden-to-output weights gradients
        obGrads = np.zeros(shape=[self.no], dtype=np.float32)  # output node biases gradients
        ihGrads = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)  # input-to-hidden weights gradients
        hbGrads = np.zeros(shape=[self.nh], dtype=np.float32)  # hidden biases gradients
    
        oSignals = np.zeros(shape=[self.no], dtype=np.float32)  # output signals: gradients w/o assoc. input terms
        hSignals = np.zeros(shape=[self.nh], dtype=np.float32)  # hidden signals: gradients w/o assoc. input terms
    
        epoch = 0
        x_values = np.zeros(shape=[self.ni], dtype=np.float32)
        t_values = np.zeros(shape=[self.no], dtype=np.float32)
        numTrainItems = len(trainData)
        indices = np.arange(numTrainItems)  # [0, 1, 2, . . n-1]  # rnd.shuffle(v)

        for epoch in range(maxEpochs):
            # self.rnd.shuffle(indices)  # scramble order of training items -- not necessary for full-batch
            # zero-out batch training accumulated weight and bias gradients
            ihWtsAccGrads = np.zeros(shape=[self.ni, self.nh], dtype=np.float32)  # input-to-hidden weights accumulated grads
            hBiasesAccGrads = np.zeros(shape=[self.nh], dtype=np.float32)  # hidden biases accumulated grads
            hoWtsAccGrads = np.zeros(shape=[self.nh, self.no], dtype=np.float32)  # hidden-to-output
            oBiasesAccGrads = np.zeros(shape=[self.no], dtype=np.float32)  # output node biases 

            # calculate accumulated gradients
            for ii in range(numTrainItems):  # visit each item, accumulate weight grads but don't update
                idx = indices[ii]  # note these are unscrambled

                for j in range(self.ni):
                    x_values[j] = trainData[idx, j]  # get the input values	
                for j in range(self.no):
                    t_values[j] = trainData[idx, j+self.ni]  # get the target values
                self.computeOutputs(x_values)  # results stored internally

                ############# calculate output&hidden weight ##############
                # 1. compute output node signals
                for k in range(self.no):
                    derivative = (1 - self.oNodes[k]) * self.oNodes[k]  # softmax 微分
                    oSignals[k] = derivative * (self.oNodes[k] - t_values[k])  # E=(t-o)^2 do E'=(o-t)

                # 2. compute hidden-to-output weight grads and accumulate
                for j in range(self.nh):
                    for k in range(self.no):
                        hoGrads[j, k] = oSignals[k] * self.hNodes[j]
                        hoWtsAccGrads[j,k] += hoGrads[j, k]

                # 3. compute and accumulate output node bias gradients 
                for k in range(self.no):
                    obGrads[k] = oSignals[k]
                    oBiasesAccGrads[k] += obGrads[k]

                ############# calculate hidden&input weight ##############
                # 4. compute hidden node signals
                for j in range(self.nh):
                    sum = 0.0
                    for k in range(self.no):
                        sum += oSignals[k] * self.hoWeights[j,k]
                    # derivative = (1 - self.hNodes[j]) * (1 + self.hNodes[j])  # tanh activation
                    derivative = (1 - self.hNodes[j]) * self.hNodes[j]  # sigmoid 微分
                    hSignals[j] = derivative * sum

                # 5 compute and accumulate input-to-hidden weight gradients using hidden signals
                for i in range(self.ni):
                    for j in range(self.nh):
                        ihGrads[i, j] = hSignals[j] * self.iNodes[i]
                        ihWtsAccGrads[i,j] += ihGrads[i, j]

                # 6. compute and accumulate hidden node bias gradients using hidden signals
                for j in range(self.nh):
                    hbGrads[j] = hSignals[j] * 1.0  # 1.0 dummy input can be dropped
                    hBiasesAccGrads[j] += hbGrads[j]
            
            # update weights - calculate deltas
            ############# calculate output&hidden weight ##############
            # 1. update input-to-hidden weights
            for i in range(self.ni):
                for j in range(self.nh):
                    delta = -1.0 * learnRate * ihWtsAccGrads[i,j] 
                    self.ihWeights[i,j] += delta

            # 2. update hidden node biases
            for j in range(self.nh):
                delta = -1.0 * learnRate * hBiasesAccGrads[j] 
                self.hBiases[j]  += delta      

            ############# calculate hidden&input weight ##############
            # 3. update hidden-to-output weights
            for j in range(self.nh):
                for k in range(self.no):
                    delta = -1.0 * learnRate * hoWtsAccGrads[j,k]
                    self.hoWeights[j,k] += delta

            # 4. update output node biases
            for k in range(self.no):
                delta = -1.0 * learnRate * oBiasesAccGrads[k]
                self.oBiases[k] += delta

            if epoch % 10 == 0:
                mse = self.meanSquaredError(trainData)
                print("epoch = " + str(epoch) + " mse = %0.5f " % mse)

        result = self.getWeights()
        return result



