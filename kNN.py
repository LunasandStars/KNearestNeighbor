# Read MNIST data
import numpy
import pandas

#These two lines below reads the csv files
data = pandas.read_csv('MNIST_test.csv')
dataTraining = pandas.read_csv('MNIST_training.csv')


# Variables for testing and training data
trainingLabels =  dataTraining.iloc[:, 0].as_matrix()
trainingData = dataTraining.drop('label', axis = 1).as_matrix()   #dropping the label in the data set and setting it to a matrix
testLabels = data.iloc[:, 0].as_matrix()
testData = data.drop('label', axis = 1).as_matrix()




import math
#getDistance calculates the Euclidean distance with a Matrix, vector and length
#The function will return a distance
def getDistance(x1, y1):
    distance = 0.0
    for x in range(0, len(x1)):
        distance += (x1[x] - y1[x])**2     #difference of distance
    return math.sqrt(distance)   #Take the squareroot of the summation of the difference between the distance




#This function takes three parameters and calculates the distance from the training data to the test data
#then stores it into an array for distance and one for nearest neighbor depending on k
#this function evaluates a single training case
def getNearestNeighbor(dataTraining, trainingLabels, testCase, k):    #Takes the training data, test data and a k value
    distanceArray = []     #Array is is empty
    neighborArray = []
    # length = len(data)  length  #Get the length of data to store
    for index in range(len(dataTraining)):     # for each x in the maximum will stop at the length of the training data
        distanceFromTest = getDistance(testCase, dataTraining[index])    #Distance of Training to Testing data
        distanceArray.append([trainingLabels[index], distanceFromTest])     #This is putting the list together
    #Sort distanceArray (Gets nearest neighbor for each training case)
    sortDistanceArray = sorted(distanceArray, key = lambda tup: tup[1])
    #The neighborArray is the k nearest neighbor, meaning that it is the k smallest elements from the distance arrray
    for index in range(k):
        neighborArray.append(sortDistanceArray[index][0])
    # return neighborArray's most common element
    return mostOccurence(neighborArray)




# This function sets an array of 10 from 0 to 9 and counts the number of each index depending on how
#many times it occurs in the labels column
def mostOccurence(neighborArray):
    counterRank = numpy.zeros(10, dtype = int)
    for index in neighborArray:
        num = index
        counterRank[num] += 1
    return numpy.argmax(counterRank)


# This function takes the following parameters below, calls getNearestNeighbor
# if the data is equal to the testing labels then count the number of times it is true with truthTracker
# Take the number of times true occurs and divide by the length of the testData
def evaluateTestData(trainingData, trainingLabels, testData, testLabels,k):
    average = 0
    truthTracker = 0
    for index in range(len(testData)):
        neighborhood = getNearestNeighbor(trainingData, trainingLabels, testData[index], k)
        if neighborhood == testLabels[index]:
            truthTracker += 1
    average = truthTracker / float(len(testData))
    #print(truthTracker)
    #print(len(testData))
    #print(average)
    print("The following accuracy is when k =" , k)
    print("Number of Correctly Classified: " , truthTracker)
    print("Total Number of Test Data" , len(testData))
    print("Accuracy = " , average)
    print("Accuracy as Percent: " , (average * 100), "%")

#Calls the function evaluateTestData (Use only odd numbers)
#evaluateTestData(trainingData, trainingLabels, testData, testLabels, 3)
evaluateTestData(trainingData, trainingLabels, testData, testLabels, 5)
evaluateTestData(trainingData, trainingLabels, testData, testLabels, 7)
evaluateTestData(trainingData, trainingLabels, testData, testLabels, 9)
# evaluateTestData(trainingData, trainingLabels, testData, testLabels, 11)
# evaluateTestData(trainingData, trainingLabels, testData, testLabels, 13)

