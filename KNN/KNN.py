from cgi import test
from msilib.schema import Class


def Distance(point1, point2, power):
    sum = 0

    for i in range(len(point1)):
        sum += abs(float(point2[0][i]) - float(point1[0][i])) ** power

    return sum ** (1 / power)

def Classify(dataset, point, k):
    distances = []
    for i in range(len(dataset)):
        newDist = Distance(dataset[i], point, 2)
        
        if len(distances) == 0:
            distances.append([newDist, dataset[i][1]])
        else:
            insertInd = 0
            while insertInd < len(distances) and distances[insertInd][0] < newDist:
                insertInd += 1
            distances.insert(insertInd, [newDist, dataset[i][1]])

    neighbors = []
    for i in range(min(k, len(distances))):
        neighbors.append(distances[i][1])

    return max(neighbors, key=neighbors.count)

def ConfusionMatrix(dataset, testingSet, k, possibleClasses):
    actualVsPredictedMatrix = []

    for i in range(possibleClasses):
        tempArr = []
        for j in range(possibleClasses):
            tempArr.append(0)
        actualVsPredictedMatrix.append(tempArr)

    classes = []
    datasetInd = 0
    while datasetInd < len(dataset) and len(classes) < possibleClasses:
        try:
            classes.index(dataset[datasetInd][1])
        except:
            classes.append(dataset[datasetInd][1])
        datasetInd += 1

    for i in range(len(testingSet)):
        actualVsPredictedMatrix[classes.index(Classify(dataset, testingSet[i], k))][classes.index(testingSet[i][1])] += 1

    for i in range(len(actualVsPredictedMatrix)):
        for j in range(len(actualVsPredictedMatrix[i])):
            print(actualVsPredictedMatrix[i][j], end=" ")
        print("\n", end="")

    print("\t\t\t\tprecision\trecall\t\tf1-score\tsupport")

    correct = 0

    real = []
    predicted = []

    precision = []
    recall = []
    f1 = []

    for i in range(len(classes)):
        real.append(SumColumnRow(actualVsPredictedMatrix, i, -1))
        predicted.append(SumColumnRow(actualVsPredictedMatrix, -1, i))

        precision.append(float(actualVsPredictedMatrix[i][i]) / real[i])
        recall.append(float(actualVsPredictedMatrix[i][i]) / predicted[i])
        f1.append(2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))

        correct += actualVsPredictedMatrix[i][i]

        print(str(classes[i]) + "\t\t\t" + str(round(precision[i], 2)) + "\t\t" + str(round(recall[i], 2)) + "\t\t" + str(round(f1[i], 2)) + "\t\t" + str(predicted[i]))

    print("\naccuracy\t\t\t\t\t\t\t" + str(round(float(correct / sum(real)), 2)) + "\t\t" + str(sum(real)))
    print("macro avg\t\t\t" + str(round(sum(precision) / len(classes), 2)) + "\t\t" + str(round(sum(recall) / len(classes), 2)) + "\t\t" + str(round(sum(f1) / len(classes), 2)) + "\t\t" + str(sum(real)))

    weightedPrecision = 0
    for i in range(len(classes)):
        weightedPrecision += (predicted[i] / sum(predicted)) * precision[i]

    weightedRecall = 0
    for i in range(len(classes)):
        weightedRecall += (predicted[i] / sum(predicted)) * recall[i]

    weightedF1 = 0
    for i in range(len(classes)):
        weightedF1 += (predicted[i] / sum(predicted)) * f1[i]

    print("weighted avg\t\t\t" + str(round(weightedPrecision, 2)) + "\t\t" + str(round(weightedRecall, 2)) + "\t\t" + str(round(weightedF1, 2)) + "\t\t" + str(sum(real)))

def SumColumnRow(data, column, row):
    sum = 0

    if column != -1:
        for i in range(len(data)):
            sum += data[i][column]
    else:
        for i in range(len(data[0])):
            sum += data[row][i]

    return sum

def ReadFile(fileName):
    data = []
    
    with open(fileName) as f:
        nextLine = f.readline()
        while len(nextLine) > 0:
            newList = nextLine.strip().split(',')
            classifier = newList.pop()

            data.append([newList, classifier])

            nextLine = f.readline()

        f.close()

    return data

first = [[1, 1, 1], "abc"]
second = [[2, 2, 2], "xyz"]
third = [[0.5, 0.5, 0.75], "abc"]
fourth = [[2.75, 2.2, 3], "xyz"]

trainingData = ReadFile("D:\School Files\Fall 2022\CS460G\KNN\TrainingData.txt")
validationData = ReadFile("D:\School Files\Fall 2022\CS460G\KNN\ValidationData.txt")
testData = ReadFile("D:\School Files\Fall 2022\CS460G\KNN\TestData.txt")

print(validationData[14])

trainingData.pop()

ConfusionMatrix(trainingData, validationData, 2, 3)