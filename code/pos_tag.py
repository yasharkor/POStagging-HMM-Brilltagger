import argparse
import nltk
from sklearn.model_selection import KFold
# different estimator functions for hmm
from nltk.probability import (
    LaplaceProbDist,
    LidstoneProbDist,
    MLEProbDist,
    ELEProbDist,
    WittenBellProbDist,
    SimpleGoodTuringProbDist,
    KneserNeyProbDist,
)
from nltk.tbl.template import Template
# different templates for brill
from nltk.tag.brill import Pos, Word, fntbl37, nltkdemo18, brill24

# create estimator functions with a variety of gammas
def lidstone1(fdist, bins):
    return LidstoneProbDist(fdist, 0.1, bins)
def lidstone2(fdist, bins):
    return LidstoneProbDist(fdist, 0.08, bins)
def lidstone3(fdist, bins):
    return LidstoneProbDist(fdist, 0.06, bins)
def lidstone4(fdist, bins):
    return LidstoneProbDist(fdist, 0.04, bins)
def lidstone5(fdist, bins):
    return LidstoneProbDist(fdist, 0.02, bins)
def lidstone6(fdist, bins):
    return LidstoneProbDist(fdist, 0.01, bins)

from util import getHmmTagger, getBrillTagger, getTaggedSentences, getSentences, getAccuracy

# This function is used to tag some test data
# it is given a tagger, a set of unlabeled sentences, and optionally
# an output file path. It then uses that tagger to assign POS tags
# to each word in the sentence and returns tagged sentences. If the
# output file path is provided it also writes to the outputfile as required 
# in the assignment description
def tagTest(tagger, testSentences, outputFilePath=None):
    outputSentences = []
    outputTags = []
    # iterate through test sentences
    for s in testSentences:
        o = ''
        # get the predicted tags
        tags = tagger.tag(s)
        # save the tags
        outputTags.append(tags)
        # Create a sentence as in output format
        for i in range(len(s)):
            o += s[i] + ' ' + tags[i][1] + '\n'
        o += '\n'
        # save the sentence
        outputSentences.append(o)
    # write to file
    if outputFilePath != None:
        with open(outputFilePath, 'w', encoding='utf8') as out:
            out.writelines(outputSentences)
    
    return outputTags

# this is the main parameter tuning function
# it was used during development to test out different combinations of parameters
# for different taggers and find their accuracy based on some splits of the train data
def train_test(trainFilePath, mode, estimator=None, template=None, templateName='', ruleCount=10):
    # using sklearn we get several splits of the train data for training and validation purposes
    nSplits = 2
    kf = KFold(n_splits=nSplits)
    # get the tagged sentences of the training data
    X = getTaggedSentences(trainFilePath)
    AverageAccuracy = 0
    # This gives the indices of each item in the training and validation splits
    for train_index, test_index in kf.split(X):
        # split the tagged sentences into training and test data using the indices
        # the prediction will be the same as test except without the real tags
        X_train, X_prediction, X_test = [], [], []
        for i in train_index:
            X_train.append(X[i])
        for i in test_index:
            X_prediction.append([pair[0] for pair in X[i]])
            X_test.append(X[i])
        
        # for each split we get the tagger with the parameters we are testing
        if mode == 'brill':
            tagger = getBrillTagger(X_train, template, ruleCount)
        else:
            tagger = getHmmTagger(X_train, estimator)

        # Using the tagger and the unlabeled validation data get an accuracy
        output = tagTest(tagger, X_prediction)
        # add to the average accuracy across all splits
        AverageAccuracy += getAccuracy(output, X_test) / nSplits
    # store and output the information of parameters and average accuracy across splits
    if mode == 'hmm':
        line = estimator.__name__ + ' ' + str(AverageAccuracy)
        print(line)
    else:
        line = templateName + ' ' + str(ruleCount) + ' ' + str(AverageAccuracy)
        print(line)
    return line

# The main loop for running this program
def main():
    # Get the arguments from the command line
    parser = argparse.ArgumentParser(description='POS tagger using either hmm or brill tagging')
    parser.add_argument('--tagger', required=True, help='The tagger to use', choices=['hmm', 'brill', 'trainhmm', 'trainbrill'])
    parser.add_argument('--test', required=True, help='The path to the test file')
    parser.add_argument('--output', required=True, help='The path to the output file')
    parser.add_argument('--train', required=True, help='The path to the training file')
    args = parser.parse_args()

    testFilePath = args.test
    outputFilePath = args.output
    taggerMode = args.tagger
    trainFilePath = args.train

    # hmm tagger
    if taggerMode == 'hmm':
        # get and process the tagged sentences
        taggedSentences = getTaggedSentences(trainFilePath)
        # train and get a tagger using those tagged sentences with the best estimator
        tagger = getHmmTagger(taggedSentences, nltk.probability.WittenBellProbDist)
        # get and process the test sentences
        testSentences = getSentences(testFilePath)
        # tag and output the test sentences using the tagger
        pred = tagTest(tagger, testSentences, outputFilePath)
        print(getAccuracy(pred, getTaggedSentences(testFilePath)))

    # brill tagger
    elif taggerMode == 'brill': 
        # get and process the tagged sentences
        taggedSentences = getTaggedSentences(trainFilePath)
        # train and get a tagger using those tagged sentences with the best template and rule count
        tagger = getBrillTagger(taggedSentences, fntbl37(), 100)
        # get and process the test sentences
        testSentences = getSentences(testFilePath)
        # tag and output the test sentences using the tagger
        pred = tagTest(tagger, testSentences, outputFilePath)
        print(getAccuracy(pred, getTaggedSentences(testFilePath)))


    # grid search hmm taggers
    elif taggerMode == 'trainhmm':
        # estimators to search through
        estimators = [
            LaplaceProbDist,
            lidstone1,
            lidstone2,
            lidstone3,
            lidstone4,
            lidstone5,
            lidstone6,
            MLEProbDist,
            ELEProbDist,
            WittenBellProbDist,
            SimpleGoodTuringProbDist,
        ]
        lines = []
        for e in estimators:
            # get a training test result and store it
            lines.append(train_test(trainFilePath, 'hmm', estimator=e))
        # write results
        print(lines)

    # grid search brill taggers
    elif taggerMode == 'trainbrill':
        # the templates to search through
        templates = {
            'base': [Template(Pos([-1])), Template(Pos([-1]), Word([0]))],
            'fntbl37': fntbl37(),
            'nltkdemo18': nltkdemo18(),
            'brill24': brill24()
        }
        # the max number of rules to search through
        ruleCounts = [10, 50, 100]
        lines = []
        for tName, t in templates.items():
            for r in ruleCounts:    
                # get a training test result and store it
                lines.append(train_test(trainFilePath, 'brill', template=t, templateName=tName, ruleCount=r))
        # write results
        print(lines)

if __name__ == "__main__":
    main()