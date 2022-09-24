from operator import truediv
import numpy
import pandas
import sys
import sklearn.metrics
import random

def evaluate(weights, point, printSum):
    sum = 0
    for i in range(len(weights)):
        sum += weights[i] * point[i]

    return numpy.sign(sum)

def getWeights(data, maxError):
    numFeatures = len(data[0]) - 1

    weights = []
    for i in range(numFeatures + 1):
        weights.append((random.random() - 0.5) * 350)

    errors = sys.maxsize
    while errors > maxError:
        errors = 0

        for i in range(len(data)):
            realPoint = []
            realPoint.append(1)
            for j in range(len(data[i]) - 1):
                realPoint.append(data[i][j])

            result = evaluate(weights, realPoint, False)

            if result != data[i][numFeatures]:
                for j in range(len(weights)):
                    weights[j] += data[i][numFeatures] * realPoint[j]
                
                errors += 1

    return weights

def confusionMatrix(classifier1, classifier2, validationData, classes):
    real = []
    predicted = []

    for i in range(len(validationData)):
        classInd = classes.index(validationData[i][4])

        real.append(classInd)

        realPoint = []
        realPoint.append(1)
        for j in range(len(validationData[i]) - 1):
            realPoint.append(validationData[i][j])
        
        result1 = evaluate(classifier1, realPoint, True)
        result2 = evaluate(classifier2, realPoint, True)

        predictedVal = -1
        if result1 == -1:
            predictedVal = 0
        elif result2 == -1:
            predictedVal = 1
        else:
            predictedVal = 2

        predicted.append(predictedVal)

    print("Confusion Matrix:")
    print(sklearn.metrics.confusion_matrix(real, predicted))

    print(sklearn.metrics.classification_report(real, predicted, zero_division=0))


rawTrainingData = pandas.read_csv("D:\School_Files\Fall_2022\CS460G\Perceptron\Training.data").to_numpy()
rawValidationData = pandas.read_csv("D:\School_Files\Fall_2022\CS460G\Perceptron\Validation.data").to_numpy()
rawTestingData = pandas.read_csv("D:\School_Files\Fall_2022\CS460G\Perceptron\Testing.data").to_numpy()

class1 = "Iris-setosa"
class2 = "Iris-versicolor"
class3 = "Iris-virginica"

maxErrors = 2

trainingData1 = []
trainingData2 = []
for i in range(len(rawTrainingData)):
    trainingData1.append(rawTrainingData[i])
    
    if rawTrainingData[i][4] == class1:
        trainingData1[i][4] = -1
    elif rawTrainingData[i][4] == class2:
        trainingData1[i][4] = 1

        trainingData2.append(rawTrainingData[i])
        trainingData2[-1][4] = -1
    elif rawTrainingData[i][4] == class3:
        trainingData1[i][4] = 1

        trainingData2.append(rawTrainingData[i])
        trainingData2[-1][4] = 1

classifier1 = getWeights(trainingData1, maxErrors)
classifier2 = getWeights(trainingData2, maxErrors)

confusionMatrix(classifier1, classifier2, rawValidationData, [class1, class2, class3])