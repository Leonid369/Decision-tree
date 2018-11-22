from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
import math
import random
import copy
import sys
import warnings



warnings.simplefilter("ignore")

if (len(sys.argv) != 7):
    sys.exit("Please give the required amount of arguments - <L> <K> <trainPath> <testPath> <validationPath> <toPrint>")
else:
    trainPath = sys.argv[3]
    testPath = sys.argv[4]
    validationPath = sys.argv[5]
    L = int(sys.argv[1])
    K = int(sys.argv[2])
    toPrint = str(sys.argv[6])

dtraining = pd.read_csv(trainPath)
dtesting = pd.read_csv(testPath)
dvalid = pd.read_csv(validationPath)

'''
dtraining = pd.read_csv("/Users/Sachuleonid/Downloads/data_sets2/training_set.csv")
dtesting = pd.read_csv("/Users/Sachuleonid/Downloads/data_sets2/test_set.csv")
dvalid = pd.read_csv("/Users/Sachuleonid/Downloads/data_sets2/validation_set.csv")
'''

# for removing the empty rows
dtraining = dtraining.dropna()
dtesting = dtesting.dropna()
dvalid = dvalid.dropna()

nodeCount = 0

'''
L = int(input("Enter value for L "))
K = int(input("Enter value for K "))
'''
print("Please wait to complete!")

def calcForEntropy(labels):
    total = labels.shape[0]
    onesCount = labels.sum().sum()
    zerosCount = total - onesCount
    if (total == onesCount) or (total == zerosCount):
        return 0
    else:
        entropy = - ((onesCount/total) * math.log(onesCount/float(total), 2)) - ((zerosCount/total) * math.log(zerosCount/float(total), 2))
        return entropy


def varianceImpurity(labels):
    total = labels.shape[0]
    onesCount = labels.sum().sum()
    zerosCount = total - onesCount
    if (total == onesCount) or (total == zerosCount):
        return 0
    varianceImp = ((zerosCount/total) * (onesCount/total))
    return varianceImp


s = 0
while (s < 3):
    s = s + 1
    if (s == 1):
        print(" ")
        print("*************************************************")
        print("INFORMATION GAIN HEURISTIC")
        print("*************************************************")
        print(" ")


        def infoGainValue(featurelabels):
            total = featurelabels.shape[0]
            onesCount = featurelabels[featurelabels[featurelabels.columns[0]] == 1].shape[0]
            zerosCount = featurelabels[featurelabels[featurelabels.columns[0]] == 0].shape[0]
            parentEntropy = calcForEntropy(featurelabels[['Class']])
            entropyChildWithOne = calcForEntropy(featurelabels[featurelabels[featurelabels.columns[0]] == 1][['Class']])
            entropyChildWithZero = calcForEntropy(featurelabels[featurelabels[featurelabels.columns[0]] == 0][['Class']])
            infoGain = parentEntropy - (onesCount / total) * entropyChildWithOne - (zerosCount / total) * entropyChildWithZero
            return infoGain


        def findTheBestAttri(data):
            maxInfoGain = -1.0
            for x in data.columns:
                if x == 'Class':
                    continue
                currentInfoGain = infoGainValue(data[[x, 'Class']])

                if maxInfoGain < currentInfoGain:
                    maxInfoGain = currentInfoGain
                    bestAttribute = x
            return bestAttribute


        class Node():
            def __init__(self):
                self.left = None
                self.right = None
                self.attribute = None
                self.nodeType = None  # L/R/I corresponds to leaf/Root/Intermidiate
                self.value = None
                self.countOfPositive = None
                self.countOfNegative = None
                self.label = None
                self.nodeId = None

            def setNodeValue(self, attribute, nodeType, value=None, countOfPositive=None, countOfNegative=None):
                self.attribute = attribute
                self.nodeType = nodeType
                self.value = value
                self.countOfPositive = countOfPositive
                self.countOfNegative = countOfNegative


        class Tree():
            def __init__(self):
                self.root = Node()
                self.root.setNodeValue('$@$', 'R')

            def creatingDecisionTree(self, data, tree):
                global nodeCount
                total = data.shape[0]
                onesCount = data['Class'].sum()
                zerosCount = total - onesCount
                if data.shape[1] == 1 or total == onesCount or total == zerosCount:
                    tree.nodeType = 'L'
                    if zerosCount >= onesCount:
                        tree.label = 0
                    else:
                        tree.label = 1
                    return
                else:
                    bestAttribute = findTheBestAttri(data)
                    tree.left = Node()
                    tree.right = Node()

                    tree.left.nodeId = nodeCount
                    nodeCount = nodeCount + 1
                    tree.right.nodeId = nodeCount
                    nodeCount = nodeCount + 1

                    tree.left.setNodeValue(bestAttribute, 'I', 0,
                                           data[(data[bestAttribute] == 0) & (dtraining['Class'] == 1)].shape[0],
                                           data[(data[bestAttribute] == 0) & (dtraining['Class'] == 0)].shape[0])
                    tree.right.setNodeValue(bestAttribute, 'I', 1,
                                            data[(data[bestAttribute] == 1) & (dtraining['Class'] == 1)].shape[0],
                                            data[(data[bestAttribute] == 1) & (dtraining['Class'] == 0)].shape[0])
                    self.creatingDecisionTree(data[data[bestAttribute] == 0].drop([bestAttribute], axis=1), tree.left)
                    self.creatingDecisionTree(data[data[bestAttribute] == 1].drop([bestAttribute], axis=1), tree.right)

            def printingTreeLevels(self, node, level):
                if (node.left is None and node.right is not None):
                    for i in range(0, level):
                        print("| ", end="")
                    level = level + 1
                    print("{} = {}  : {}".format(node.attribute, node.value,
                                                 (node.label if node.label is not None else "")))
                    self.printingTreeLevels(node.right, level)
                elif (node.right is None and node.left is not None):
                    for i in range(0, level):
                        print("| ", end="")
                    level = level + 1
                    print("{} = {}  : {}".format(node.attribute, node.value,
                                                 (node.label if node.label is not None else "")))
                    self.printingTreeLevels(node.left, level)
                elif (node.right is None and node.left is None):
                    for i in range(0, level):
                        print("| ", end="")
                    level = level + 1
                    print("{} = {}  : {}".format(node.attribute, node.value,
                                                 (node.label if node.label is not None else "")))
                else:
                    for i in range(0, level):
                        print("| ", end="")
                    level = level + 1
                    print("{} = {}  : {}".format(node.attribute, node.value,
                                                 (node.label if node.label is not None else "")))
                    self.printingTreeLevels(node.left, level)
                    self.printingTreeLevels(node.right, level)

            def printingTree(self, node):
                self.printingTreeLevels(node.left, 0)
                self.printingTreeLevels(node.right, 0)

            def predictingLabel(self, data, root):
                if root.label is not None:
                    return root.label
                elif data[root.left.attribute][data.index.tolist()[0]] == 1:
                    return self.predictingLabel(data, root.right)
                else:
                    return self.predictingLabel(data, root.left)

            def countOfNodes(self, node):
                if (node.left is not None and node.right is not None):
                    return 2 + self.countOfNodes(node.left) + self.countOfNodes(node.right)
                return 0

            def countOfLeaf(self, node):
                if (node.left is None and node.right is None):
                    return 1
                return self.countOfLeaf(node.left) + self.countOfLeaf(node.right)


        def searchingNode(tree, x):
            tmp = None
            res = None
            if (tree.nodeType != "L"):
                if (tree.nodeId == x):
                    return tree
                else:
                    res = searchingNode(tree.left, x)
                    if (res is None):
                        res = searchingNode(tree.right, x)
                    return res
            else:
                return tmp


        def afterPruning(pNumber, newTree):

            for i in range(pNumber):
                x = random.randint(2, pruningTree.countOfNodes(pruningTree.root) - 1)
                tempNode = Node()
                tempNode = searchingNode(newTree, x)

                if (tempNode is not None):
                    tempNode.left = None
                    tempNode.right = None
                    tempNode.nodeType = "L"
                    if (tempNode.countOfNegative >= tempNode.countOfPositive):
                        tempNode.label = 0
                    else:
                        tempNode.label = 1


        def calcOfAccuracy(data, tree):
            correctCount = 0
            for i in data.index:
                val = tree.predictingLabel(data.iloc[i:i + 1, :].drop(['Class'], axis=1), tree.root)
                if val == data['Class'][i]:
                    correctCount = correctCount + 1
            return (correctCount / data.shape[0]) * 100


        dtree = Tree()
        dtree.creatingDecisionTree(dtraining, dtree.root)

        maximumAccuracy = calcOfAccuracy(dvalid, dtree)
        DbestTree = copy.deepcopy(dtree)
        countOfNodes = DbestTree.countOfNodes(DbestTree.root)
        c = 0

        while c < L:
            c += 1
            pruneNumber = K

            pruningTree = Tree()
            pruningTree = copy.deepcopy(DbestTree)
            afterPruning(pruneNumber, pruningTree.root)
            temp = calcOfAccuracy(dvalid, pruningTree)
            if temp > maximumAccuracy:
                maximumAccuracy = temp
                DbestTree = copy.deepcopy(pruningTree)
                countOfNodes = DbestTree.countOfNodes(DbestTree.root)

        print("-------------------------------------")

        print("Accuracy of the Decision tree before pruning = " + str(calcOfAccuracy(dtesting, dtree)) + "%")
        print("")
        if (toPrint == "yes"):
            print("Pre-Pruned Tree")
            print("-------------------------------------")

            dtree.printingTree(dtree.root)

        print("-------------------------------------")

        print("Accuracy of the Decision tree after pruning = " + str(calcOfAccuracy(dvalid, DbestTree)) + "%")
        print("")
        if (toPrint == "yes"):
            print("Post-Pruned Tree")
            print("-------------------------------------")

            DbestTree.printingTree(DbestTree.root)

    if (s == 2):
        print(" ")
        print("*************************************************")
        print("VARIANCE IMPURITY HEURISTIC")
        print("*************************************************")
        print(" ")


        def infoGainValue(featurelabels):
            total = featurelabels.shape[0]
            onesCount = featurelabels[featurelabels[featurelabels.columns[0]] == 1].shape[0]
            zerosCount = featurelabels[featurelabels[featurelabels.columns[0]] == 0].shape[0]
            parentVarianceImp = varianceImpurity(featurelabels[['Class']])
            varianceImpChildWithOne = varianceImpurity(featurelabels[featurelabels[featurelabels.columns[0]] == 1][['Class']])
            varianceImpChildWithZero = varianceImpurity(featurelabels[featurelabels[featurelabels.columns[0]] == 0][['Class']])

            infoGain = parentVarianceImp - ((onesCount / total) * varianceImpChildWithOne) - ((zerosCount / total) * varianceImpChildWithZero)
            return infoGain


        def findTheBestAttri(data):
            maxInfoGain = -1.0
            for x in data.columns:
                if x == 'Class':
                    continue
                currentInfoGain = infoGainValue(data[[x, 'Class']])

                if maxInfoGain < currentInfoGain:
                    maxInfoGain = currentInfoGain
                    bestAttribute = x
            return bestAttribute


        class Node():
            def __init__(self):
                self.left = None
                self.right = None
                self.attribute = None
                self.nodeType = None  # leaf(L)/Root(R)/Intermidiate(I)
                self.value = None
                self.countOfPositive = None
                self.countOfNegative = None
                self.label = None
                self.nodeId = None

            def setNodeValue(self, attribute, nodeType, value=None, countOfPositive=None, countOfNegative=None):
                self.attribute = attribute
                self.nodeType = nodeType
                self.value = value
                self.countOfPositive = countOfPositive
                self.countOfNegative = countOfNegative


        class Tree():
            def __init__(self):
                self.root = Node()
                self.root.setNodeValue('$@$', 'R')

            def creatingDecisionTree(self, data, tree):
                global nodeCount
                total = data.shape[0]
                onesCount = data['Class'].sum()
                zerosCount = total - onesCount
                if data.shape[1] == 1 or total == onesCount or total == zerosCount:
                    tree.nodeType = 'L'
                    if zerosCount >= onesCount:
                        tree.label = 0
                    else:
                        tree.label = 1
                    return
                else:
                    bestAttribute = findTheBestAttri(data)
                    tree.left = Node()
                    tree.right = Node()

                    tree.left.nodeId = nodeCount
                    nodeCount = nodeCount + 1
                    tree.right.nodeId = nodeCount
                    nodeCount = nodeCount + 1

                    tree.left.setNodeValue(bestAttribute, 'I', 0,
                                           data[(data[bestAttribute] == 0) & (dtraining['Class'] == 1)].shape[0],
                                           data[(data[bestAttribute] == 0) & (dtraining['Class'] == 0)].shape[0])
                    tree.right.setNodeValue(bestAttribute, 'I', 1,
                                            data[(data[bestAttribute] == 1) & (dtraining['Class'] == 1)].shape[0],
                                            data[(data[bestAttribute] == 1) & (dtraining['Class'] == 0)].shape[0])
                    self.creatingDecisionTree(data[data[bestAttribute] == 0].drop([bestAttribute], axis=1), tree.left)
                    self.creatingDecisionTree(data[data[bestAttribute] == 1].drop([bestAttribute], axis=1), tree.right)

            def printingTreeLevels(self, node, level):
                if (node.left is None and node.right is not None):
                    for i in range(0, level):
                        print("| ", end="")
                    level = level + 1
                    print("{} = {}  : {}".format(node.attribute, node.value,
                                                 (node.label if node.label is not None else "")))
                    self.printingTreeLevels(node.right, level)
                elif (node.right is None and node.left is not None):
                    for i in range(0, level):
                        print("| ", end="")
                    level = level + 1
                    print("{} = {}  : {}".format(node.attribute, node.value,
                                                 (node.label if node.label is not None else "")))
                    self.printingTreeLevels(node.left, level)
                elif (node.right is None and node.left is None):
                    for i in range(0, level):
                        print("| ", end="")
                    level = level + 1
                    print("{} = {}  : {}".format(node.attribute, node.value,
                                                 (node.label if node.label is not None else "")))
                else:
                    for i in range(0, level):
                        print("| ", end="")
                    level = level + 1
                    print("{} = {}  : {}".format(node.attribute, node.value,
                                                 (node.label if node.label is not None else "")))
                    self.printingTreeLevels(node.left, level)
                    self.printingTreeLevels(node.right, level)

            def printingTree(self, node):
                self.printingTreeLevels(node.left, 0)
                self.printingTreeLevels(node.right, 0)

            def predictingLabel(self, data, root):
                if root.label is not None:
                    return root.label
                elif data[root.left.attribute][data.index.tolist()[0]] == 1:
                    return self.predictingLabel(data, root.right)
                else:
                    return self.predictingLabel(data, root.left)

            def countOfNodes(self, node):
                if (node.left is not None and node.right is not None):
                    return 2 + self.countOfNodes(node.left) + self.countOfNodes(node.right)
                return 0

            def countOfLeaf(self, node):
                if (node.left is None and node.right is None):
                    return 1
                return self.countOfLeaf(node.left) + self.countOfLeaf(node.right)


        def searchingNode(tree, x):
            tmp = None
            res = None
            if (tree.nodeType != "L"):
                if (tree.nodeId == x):
                    return tree
                else:
                    res = searchingNode(tree.left, x)
                    if (res is None):
                        res = searchingNode(tree.right, x)
                    return res
            else:
                return tmp


        def afterPruning(pNumber, newTree):

            for i in range(pNumber):
                x = random.randint(2, pruningTree.countOfNodes(pruningTree.root) - 1)
                tempNode = Node()
                tempNode = searchingNode(newTree, x)

                if (tempNode is not None):
                    tempNode.left = None
                    tempNode.right = None
                    tempNode.nodeType = "L"
                    if (tempNode.countOfNegative >= tempNode.countOfPositive):
                        tempNode.label = 0
                    else:
                        tempNode.label = 1


        def calcOfAccuracy(data, tree):
            correctCount = 0
            for i in data.index:
                val = tree.predictingLabel(data.iloc[i:i + 1, :].drop(['Class'], axis=1), tree.root)
                if val == data['Class'][i]:
                    correctCount = correctCount + 1
            return (correctCount / data.shape[0]) * 100


        dtree = Tree()
        dtree.creatingDecisionTree(dtraining, dtree.root)

        maximumAccuracy = calcOfAccuracy(dvalid, dtree)
        DbestTree = copy.deepcopy(dtree)
        countOfNodes = DbestTree.countOfNodes(DbestTree.root)
        c = 0

        while c < L:
            c += 1
            pruneNumber = K

            pruningTree = Tree()
            pruningTree = copy.deepcopy(DbestTree)
            afterPruning(pruneNumber, pruningTree.root)
            temp = calcOfAccuracy(dvalid, pruningTree)
            if temp > maximumAccuracy:
                maximumAccuracy = temp
                DbestTree = copy.deepcopy(pruningTree)
                countOfNodes = DbestTree.countOfNodes(DbestTree.root)

        print("-------------------------------------")

        print("Accuracy of the Decision tree before pruning = " + str(calcOfAccuracy(dtesting, dtree)) + "%")

        if (toPrint == "yes"):
            print("Pre-Pruned Tree")
            print("-------------------------------------")

            dtree.printingTree(dtree.root)

        print("-------------------------------------")

        print("Accuracy of the Decision tree after pruning = " + str(calcOfAccuracy(dvalid, DbestTree)) + "%")
        print("")
        if (toPrint == "yes"):
            print("Post-Pruned Tree")
            print("-------------------------------------")

            DbestTree.printingTree(DbestTree.root)


