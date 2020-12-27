import sys
import nltk
from nltk.tbl.template import Template
from nltk.tag.brill import Pos, Word, fntbl37
from nltk.tag import untag, RegexpTagger, BrillTaggerTrainer
from nltk.corpus import treebank
from nltk.probability import (
    FreqDist,
    ConditionalFreqDist,
    ConditionalProbDist,
    DictionaryProbDist,
    DictionaryConditionalProbDist,
    LidstoneProbDist,
    MutableProbDist,
    MLEProbDist,
    RandomProbDist,
)

# This function reads tagged sentences from an input file
# and returns them as a list of sentences where each sentence
# is a list of tuples in the form of (word, tag)
def getTaggedSentences(filePath):
    taggedSentences = []

    with open(filePath, 'r', encoding='utf8') as f:
        s = []
        for line in f:
            # this is a linebreak indicating end of a sentence
            if line == '\n':
                # save the sentence
                taggedSentences.append(s)
                # reset to a new blank sentence
                s = []
            else:
                # get the word and the tag and add them to the sentence
                s.append((line.split()[0].strip(), line.split()[1].strip()))
    return taggedSentences

# This function reads tagged sentences from an input file
# and returns them as a list of sentences where each sentence
# is a list of words
def getSentences(filePath):
    sentences = []

    with open(filePath, 'r', encoding='utf8') as f:
        s = []
        for line in f:
            # this is a linebreak indicating end of a sentence
            if line == '\n':
                # save the sentence
                sentences.append(s)
                # reset to a new blank sentence
                s = []
            else:
                # get the word and add it to the sentence
                s.append(line.split()[0].strip())
    return sentences

# This function takes a list of tagged sentences in the same format
# as the return type of getTaggedSentences, as well as an estimator
# function, and returns an HMM based tagger that was trained off of 
# those sentences using that estimator function
def getHmmTagger(taggedSentences, estimator):
    # Create the trainer
    trainer = nltk.tag.hmm.HiddenMarkovModelTrainer()
    # train the tagger
    tagger = trainer.train_supervised(taggedSentences, estimator=estimator)
    return tagger

# This function takes a list of tagged sentences in the same format
# as the return type of getTaggedSentences, as well as a set of rule
# templates, and a maximum number of rules, and returns a Brill transformation
# based tagger that was trained off of those sentences using the 
# templates and the maximum number of rules
def getBrillTagger(taggedSentences, template, maxRules):
    # Initialize a basic set of rules (same as that of the nltk
    # documentation for BrillTaggerTrainer)
    baseline = RegexpTagger([
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),   # cardinal numbers
    (r'(The|the|A|a|An|an)$', 'AT'),   # articles
    (r'.*able$', 'JJ'),                # adjectives
    (r'.*ness$', 'NN'),                # nouns formed from adjectives
    (r'.*ly$', 'RB'),                  # adverbs
    (r'.*s$', 'NNS'),                  # plural nouns
    (r'.*ing$', 'VBG'),                # gerunds
    (r'.*ed$', 'VBD'),                 # past tense verbs
    (r'.*', 'NN')                      # nouns (default)
    ])
    # reset the templates from previous runs
    Template._cleartemplates() 
    # create the trainer
    tt = BrillTaggerTrainer(baseline, template, trace=3)
    # train the tagger
    tagger1 = tt.train(taggedSentences, max_rules=maxRules)
    return tagger1

# This function takes a set of predicted POS tags for sentences
# and compares them against the actual pos tags for those sentences
# returning the percentage correct
def getAccuracy(prediction, actual):
    total = 0
    correct = 0
    # go through each sentence
    for i in range(len(actual)):
        actualSentence = actual[i]
        predictedSentence = prediction[i]
        # go through each word, tag pair in that sentence
        for j in range(len(actualSentence)):
            predictedTag = predictedSentence[j][1]
            actualTag = actualSentence[j][1]
            if actualTag == predictedTag:
                correct += 1
            total += 1
    return correct / total