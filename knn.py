from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNN:

    dataset = ""

    def __init__( self, dataset):
        self.dataset = dataset    
        if dataset == "digits":
            self.knn = KNeighborsClassifier(n_neighbors=1)
        elif dataset == "faces":
            self.knn = KNeighborsClassifier(n_neighbors=103)

    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        reshapedTrainingData = []
        for i in range(len(trainingData)):
            reshapedArray = []
            for val in trainingData[i].values():
                reshapedArray.append(val)
            reshapedTrainingData.append(reshapedArray)

        self.knn.fit(reshapedTrainingData, trainingLabels)
    
    def classify(self, data ):
        guesses = []
        for i in range(len(data)):
            reshapedArray = []
            for val in data[i].values():
                reshapedArray.append(val)
            finalArray = [reshapedArray]
            guesses.append(self.knn.predict(finalArray))
        
        return guesses