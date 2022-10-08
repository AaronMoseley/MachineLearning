from cmath import exp, isnan
from operator import index
from unicodedata import numeric
import pandas
import numpy
import math

def predict(weights, row):
    yhat = weights[-1]

    for i in range(len(row) - 1):
        if isnan(row[i]):
            row[i] = 0

        yhat += weights[i] * row[i]
    return (1.0 / (1.0 + exp(-yhat))).real
    
def getWeights(data, lRate, epochs):
    weights = [0.0 for i in range(len(data[0]))]

    for i in range(epochs):
        sumError = 0.0
        for row in data:
            yhat = predict(weights, row)

            error = row[-1] - yhat

            sumError += error * error

            weights[-1] = weights[-1] + lRate * error * yhat * (1.0 - yhat)
            for j in range(len(row) - 1):
                weights[j] = weights[j] + lRate * error * row[j] * yhat * (1.0 - yhat)
        
        print(">epoch=%d, lrate=%.3f, error=%.3f" % (i, lRate, sumError))
    
    return weights

def normalizeData(data):
    minmax = []
    
    for i in range(len(data[0]) - 1):
        colVals = [row[i] for row in data]
        minimum = min(colVals)
        maximum = max(colVals)

        minmax.append([minimum, maximum])

    for i in range(len(data)):
        for j in range(len(data[0]) - 1):
            data[i][j] = (data[i][j] - minmax[j][0]) / (minmax[j][1] - minmax[j][0])

    return data

lRate = 0.0001
epochs = 50
validationSize = 100

trainData = pandas.read_csv("D:\School_Files\Fall_2022\CS460G\LogisticRegression\\train.csv")
testData = pandas.read_csv("D:\School_Files\Fall_2022\CS460G\LogisticRegression\\test.csv")
validation = trainData.head(validationSize)

for i in range(validationSize):
    trainData.drop(0)

trainData = trainData.sort_values(by='SalePrice')

nonNumericTrainData = trainData.select_dtypes(exclude='number').apply(lambda x: pandas.factorize(x)[0])
nonNumericTestData = testData.select_dtypes(exclude='number').apply(lambda x: pandas.factorize(x)[0])
nonNumericValidation = validation.select_dtypes(exclude='number').apply(lambda x: pandas.factorize(x)[0])

for columnName, columnData in nonNumericTrainData.iteritems():
    trainData[columnName] = nonNumericTrainData[columnName]

for columnName, columnData in nonNumericTestData.iteritems():
    testData[columnName] = nonNumericTestData[columnName]

for columnName, columnData in nonNumericValidation.iteritems():
    validation[columnName] = nonNumericValidation[columnName]

trainData = numpy.nan_to_num(trainData.to_numpy(), False)
testData = numpy.nan_to_num(testData.to_numpy(), False)
validation = numpy.nan_to_num(validation.to_numpy(), False)

for i in range(len(trainData)):
    if(trainData[i][-1] > 180000):
        trainData[i][-1] = 1
    else:
        trainData[i][-1] = 0

for i in range(len(validation)):
    if(validation[i][-1] > 180000):
        validation[i][-1] = 1
    else:
        validation[i][-1] = 0

usableTrainData = normalizeData(numpy.delete(trainData, 0, 1))
usableTestData = normalizeData(numpy.delete(testData, 0, 1)).tolist()
usableValidation = normalizeData(numpy.delete(validation, 0, 1))

maximum = max([row[-1] for row in trainData])
minimum = min([row[-1] for row in trainData])

weights = getWeights(usableTrainData, lRate, epochs)
print(weights)

predictedPrices = []
for i in range(len(usableTestData)):
    usableTestData[i].append(1)
    prediction = predict(weights, usableTestData[i])

    realPrediction = 0
    if prediction > 0.5:
        realPrediction = 1

    predictedPrices.append(realPrediction)

print(predictedPrices)

correct = 0
for i in range(len(usableValidation)):
    prediction = round(predict(weights, usableValidation[i]))

    if prediction == usableValidation[i][-1]:
        correct += 1

print("Accuracy: " + str(float(correct) / len(validation)))