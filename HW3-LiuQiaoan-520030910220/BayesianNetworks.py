# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from functools import reduce

## Function to create a conditional probability table
## Conditional probability is of the form p(x1 | x2, ..., xk)
## varnames: vector of variable names (strings) first variable listed 
##           will be x_i, remainder will be parents of x_i, p1, ..., pk
## probs: vector of probabilities for the flattened probability table
## outcomesList: a list containing a vector of outcomes for each variable
## factorTable is in the type of pandas dataframe
## See the test file for examples of how this function works
def readFactorTable(varnames, probs, outcomesList):
    factorTable = pd.DataFrame({'probs': probs})

    totalfactorTableLength = len(probs)
    numVars = len(varnames)

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(totalfactorTableLength / (k * numLevs))
        k = k * numLevs

    return factorTable

## Build a factorTable from a data frame using frequencies
## from a data frame of data to generate the probabilities.
## data: data frame read using pandas read_csv
## varnames: specify what variables you want to read from the table
## factorTable is in the type of pandas dataframe
def readFactorTablefromData(data, varnames):
    numVars = len(varnames)
    outcomesList = []

    for i in range(0, numVars):
        name = varnames[i]
        outcomesList = outcomesList + [list(set(data[name]))]

    lengths = list(map(lambda x: len(x), outcomesList))
    m = reduce(lambda x, y: x * y, lengths)
   
    factorTable = pd.DataFrame({'probs': np.zeros(m)})

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    numLevels = len(outcomesList[0])

    # creates the vector called fact to index probabilities 
    # using matrix multiplication with the data frame
    fact = np.zeros(data.shape[1])
    lastfact = 1
    for i in range(len(varnames) - 1, -1, -1):
        fact = np.where(np.isin(list(data), varnames[i]), lastfact, fact)
        lastfact = lastfact * len(outcomesList[i])

    # Compute unnormalized counts of subjects that satisfy all conditions
    a = (data - 1).dot(fact) + 1
    for i in range(0, m):
        factorTable.at[i,'probs'] = sum(a == (i+1))

    # normalize the conditional probabilities
    skip = int(m / numLevels)
    for i in range(0, skip):
        normalizeZ = 0
        for j in range(i, m, skip):
            normalizeZ = normalizeZ + factorTable['probs'][j]
        for j in range(i, m, skip):
            if normalizeZ != 0:
                factorTable.at[j,'probs'] = factorTable['probs'][j] / normalizeZ

    return factorTable


## Join of two factors
## Factor1, Factor2: two factor tables
##
## Should return a factor table that is the join of factor 1 and 2.
## You can assume that the join of two factors is a valid operation.
## Hint: You can look up pd.merge for mergin two factors
def joinFactors(Factor1, Factor2):
    # your code 
    intersect = set(Factor1).intersection(set(Factor2))
    intersect.remove("probs")
    if len(intersect) == 0:
        Factor =  pd.merge(Factor1, Factor2, how = "cross")
    else:
        Factor = pd.merge(Factor1, Factor2, on = list(intersect), how = "left")

    Factor["probs_x"] = Factor["probs_x"] * Factor["probs_y"]
    Factor.rename(columns={"probs_x":"probs"}, inplace=True)
    Factor.drop(columns=["probs_y"], inplace = True)

    return Factor

## Marginalize a variable from a factor
## table: a factor table in dataframe
## hiddenVar: a string of the hidden variable name to be marginalized
##
## Should return a factor table that marginalizes margVar out of it.
## Assume that hiddenVar is on the left side of the conditional.
## Hint: you can look can pd.groupby
def marginalizeFactor(factorTable, hiddenVar):
    # your code 
    labels = list(factorTable)
    if hiddenVar in labels:
        labels.remove(hiddenVar)
        labels.remove("probs")
        if labels:
            return factorTable.groupby(labels)["probs"].sum().to_frame().reset_index()
        else:
            return None
    else:
        return factorTable

## Marginalize a list of variables 
## bayesnet: a list of factor tables and each table iin dataframe type
## hiddenVar: a string of the variable name to be marginalized 
## NOTE: hiddenVar should be a list of string of the variable name to be marginalized according to PDF.
##
## Should return a Bayesian network containing a list of factor tables that results
## when the list of variables in hiddenVar is marginalized out of bayesnet.
def marginalizeNetworkVariables(bayesNet, hiddenVar):
    # your code 
    for h in hiddenVar:
        bayesNet = [marginalizeFactor(factorTable, h) for factorTable in bayesNet]
        bayesNet = [factor for factor in bayesNet if factor is not None] #list(filter(None, bayesNet))
    return bayesNet

## Update BayesNet for a set of evidence variables
## bayesnet: a list of factor and factor tables in dataframe format
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## Set the values of the evidence variables. Other values for the variables
## should be removed from the tables. You do not need to normalize the factors
def evidenceUpdateNet(bayesnet, evidenceVars, evidenceVals):
    # your code 
    return [evidenceUpdateFactor(factorTable, evidenceVars, evidenceVals) for factorTable in bayesnet]


## Run inference on a Bayesian network
## bayesnet: a list of factor tables and each table iin dataframe type
## hiddenVar: a string of the variable name to be marginalized
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## This function should run variable elimination algorithm by using 
## join and marginalization of the sets of variables. 
## The order of the elimiation can follow hiddenVar ordering
## It should return a single joint probability table. The
## variables that are hidden should not appear in the table. The variables
## that are evidence variable should appear in the table, but only with the single
## evidence value. The variables that are not marginalized or evidence should
## appear in the table with all of their possible values. The probabilities
## should be normalized to sum to one.
def inference(bayesnet, hiddenVar, evidenceVars, evidenceVals):
    # your code 
    bayesnet = reduce(joinFactors, bayesnet)
    bayesnet = marginalizeNetworkVariables([bayesnet], hiddenVar)
    bayesnet = evidenceUpdateNet(bayesnet, evidenceVars, evidenceVals)[0]
    bayesnet["probs"] = bayesnet["probs"] / bayesnet["probs"].sum()
    return bayesnet


## you can add other functions as you wish.

def evidenceUpdateFactor(factorTable, evidenceVars, evidenceVals):
    # use `query` method.
    if isinstance(evidenceVars, list):
        if evidenceVars and set(evidenceVars).issubset(set(factorTable)):
            query_condition = [f"({var} == {val})" for var, val in zip(evidenceVars, evidenceVals)]
            return factorTable.query("&".join(query_condition)).reset_index().drop(columns = ["index"])
        else:
            return factorTable
    else:
        if evidenceVars and evidenceVars in set(factorTable):
            return factorTable[factorTable[evidenceVars] == int(evidenceVals)].reset_index().drop(columns = ["index"]) 
        else:
            return factorTable

def my_function():
    return
