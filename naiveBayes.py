# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math
import collections

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    self.count = None
    self.labelCount = None
    self.labelsDict = None
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in list(datum.keys()) ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    #Generating the counts of each label in trainingLabels: eg- 0: 48, 1: 52
    labelCount = {}
    for t in trainingLabels:
      if t not in labelCount.keys():
        labelCount[t] = 1
      else:
        labelCount[t] += 1

    # Generating probabilities of the label counts:        eg- 0: 0.48, 1: 0.52
    for key in labelCount.keys():
        labelCount[key] /= float(len(trainingLabels))
    
    labelsDict = {}

    for key in labelCount.keys():

      #Creating a default dictionary of type list for each label:
      labelsDict[key] = collections.defaultdict(list)
  
      # Storing the training data present at the corresponding indices of its training label:
      # eg- labelTrainData = [{(0,0): 1, ....}, {(0,0): 0, ....}, ...]
      labelTrainData = []
      for i in range(len(trainingLabels)):     
        if trainingLabels[i] == key:                               
          labelTrainData.append(trainingData[i])

      # Storing training data in its corresponding label's dictionary in labelDict:
      # eg- labelsDict = {0: {{(0,0): [1, 0, 0, ...], ....}, {(0,1): [0, 1, 1, ...], ....}, ...}, 
      #                   1: {{(0,0): [0, 1, 0, ...], ....}, {(0,1): [1, 1, 0, ...], ....}, ...}} 
      for i in range(len(labelTrainData)):
        for k in labelTrainData[i].keys():
          labelsDict[key][k].append(labelTrainData[i][k])

    for key in labelCount.keys():     
      for k in labelTrainData[key].keys():
        
        # Calculating the probabilities of each pixel key in labelsDict:
        # eg- labelsDict = {0: {{(0,0): 0.33, ....}, {(0,1): 0.23, ....}, ...}, 
        #                   1: {{(0,0): 0.54, ....}, {(0,1): 0.45, ....}, ...}} 
        keyProb = {}
        for i in range(len(labelsDict[key][k])):
          if labelsDict[key][k][i] not in keyProb:
            keyProb[labelsDict[key][k][i]] = 1
          else:
            keyProb[labelsDict[key][k][i]] += 1
        
        for x in keyProb.keys():
          keyProb[x] /= float(len(labelsDict[key][k]))

        labelsDict[key][k] = keyProb

    # Store count, labelCount and labelsDict values so that they can be later used in calculateLogJointProbabilities() function:
    self.count = labelCount.keys()
    self.labelCount = labelCount
    self.labelsDict = labelsDict

        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"   
    # For each label, we calculate it's probability and select the one that is the highest:
    for i in self.count:    
      labelProb = self.labelCount[i]  

      # We have stored probabilities for each pixel in labelsDict, we fetch it, take its log and add it to the label's probability:
      for key in datum.keys():
        if self.labelsDict[i][key].get(datum[key]) is not None:
          labelProb += math.log(self.labelsDict[i][key].get(datum[key]))
        else:
          # If we do not get a value for a key, we take the value as 0.0001 instead of 0 / None:
          labelProb += math.log(0.0001)
      
      logJoint[i] = labelProb
    
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds