import numpy
import pandas
import sklearn.metrics
import string

def DataProcessing(data):
    for i in range(len(data)):
        data[i][1] = str(pandas.Series(data[i][1]).str.encode('ascii', 'ignore').str.decode('ascii'))
        
        data[i][1] = data[i][1].replace('.', ' ')
        data[i][1] = data[i][1].replace('\"', ' ')
        data[i][1] = data[i][1].replace('?', ' ')
        data[i][1] = data[i][1].replace('!', ' ')
        data[i][1] = data[i][1].replace(':', ' ')
        data[i][1] = data[i][1].replace(';', ' ')
        data[i][1] = data[i][1].replace('&', ' ')
        data[i][1] = data[i][1].replace('*', ' ')

        data[i][1] = " ".join(data[i][1].split())
        data[i][1] = data[i][1].lower()

    usableData = []

    for i in range(len(data)):
        currEntry = []
        if data[i][0] == "ham":
            currEntry.append(0)
        else:
            currEntry.append(1)

        currEntry.append(data[i][1].split())
        del currEntry[1][0]

        usableData.append(currEntry)

    return usableData

def NumOccurences(word, data):
    occurences = [[0, 0], [0, 0]]

    for i in range(len(data)):
        occurences[data[i][0]][0] += 1
        
        try:
            occurences[data[i][0]][1] += numpy.sign(data[i][1].index(word))
        except:
            pass

    return occurences

rawTrainingData = pandas.read_csv(r"D:\School_Files\Fall_2022\CS460G\Naive_Bayes\Training", sep="\t").to_numpy()

rawValidationData = pandas.read_csv(r"D:\School_Files\Fall_2022\CS460G\Naive_Bayes\Validation", sep="\t").to_numpy()
rawTestingData = pandas.read_csv(r"D:\School_Files\Fall_2022\CS460G\Naive_Bayes\Testing", sep="\t").to_numpy()

trainingData = DataProcessing(rawTrainingData)
validationData = DataProcessing(rawValidationData)
testingData = DataProcessing(rawTestingData)

totalHam = NumOccurences("", trainingData)[0][0]
totalSpam = NumOccurences("", trainingData)[1][0]

real = []
predicted = []

for i in range(len(validationData)):
    real.append(validationData[i][0])

    hamVSpam = [0, 0]

    for j in range(len(validationData[i][1])):
        probHam = (NumOccurences(validationData[i][1][j], trainingData)[0][1] / totalHam) * (totalHam / (totalHam + totalSpam))
        probSpam = (NumOccurences(validationData[i][1][j], trainingData)[1][1] / totalSpam) * (totalSpam / (totalHam + totalSpam))

        if probHam > probSpam:
            hamVSpam[0] += 1
        else:
            hamVSpam[1] += 1

    if hamVSpam[0] > hamVSpam[1]:
        predicted.append(1)
    else:
        predicted.append(0)

print(sklearn.metrics.classification_report(real, predicted))