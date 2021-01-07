# Part of speech tagging HMM and Brill tagger



## Installation and Execution Instructions
- Clone the repository
- In your python environment ensure that you have installed the nltk and sklearn packages, these can be installed with the following commands:
 ``` python3 -m pip install nltk ``` and ```python3 -m pip install scikit-learn==0.20 ``` (version 0.20 only needed on lab machines with python 3.5)
- Navigate to the code folder of the repository (```yasharkor/POStagging-HMM-Brilltagger/code``` on github)
- from the terminal execute the following command: ``` python3 .\pos_tag.py --tagger [tagger mode] --test [path to test file] --output [path to output file] --train [path to training file] ```
- The arguments can be entered in any order
- The **[path to test file]** argument is the relative path to the file you wish to POS tag
- The **[path to output file]** argument is the relative path to the file you wish to write the tagged output sentences to
- The **[path to training file]** argument is the relative path to the file you will use to train your tagger
- The **[tagger mode]** argument is the mode you wish to run the program in. You can select from the following modes:

  - **hmm** - tag the POS using an hmm tagger
  - **brill** - tag the POS using a brill tagger
- Additionally there are two training modes

  - **trainhmm** - grid search through several estimator functions to find the best one
  - **trainbrill** - grid search through several templates and rule counts to find the best ones
- An example usage of this program would be: ``` python3 ./pos_tag.py --tagger brill --test ../data/test.txt --output ../output/in_domain_brill.txt --train ../data/train.txt ```
## Data

The training data can be found in [data/train.txt](data/train.txt), the in-domain test data can be found in [data/test.txt](data/test.txt), and the out-of-domain test data can be found in [data/test_ood.txt](data/test_ood.txt).

## Resources Consulted
- https://stackoverflow.com/questions/9233027/unicodedecodeerror-charmap-codec-cant-decode-byte-x-in-position-y-character
- https://stackoverflow.com/questions/28540800/how-to-pass-in-an-estimator-to-nltks-ngrammodel
- https://www.geeksforgeeks.org/nlp-brill-tagger/
- https://streamhacker.com/2008/12/03/part-of-speech-tagging-with-nltk-part-3/
- https://stackoverflow.com/questions/32106090/nltk-brill-tagger-splitting-words


## Introduction

As you know, part-of-speech (POS) tagging is one of the most traditional and important natural language processing techniques. Not only is it a task in itself but it is also applied as a preliminary step to other NLP tasks. This technique allows the extraction of texts’ structures and aids the interpretation of a word by defining its category in the given context. Having a lot of errors in such an early processing step can have catastrophic consequences in later stages of the process, as errors tend to propagate throughout the system. This is why the accuracy of a POS tagger is a critical metric to pay attention to, and why the choice of tagger should be carefully considered when using part-of-speech tags as features.

## Task
In this project, we are going to work with part-of-speech taggers. More specifically, we are going to tune, compare, and contrast two part-of-speech taggers’ performance on in-domain and out-of-domain text samples. The two taggers are the Hidden Markov Model (HMM) tagger and the Brill tagger. Both of them are available in the [nltk package](https://www.nltk.org/). The [HMM tagger](https://www.nltk.org/api/nltk.tag.html?#module-nltk.tag.hmm) is a generative tagger defined by a set of states and state-transition probabilities. This tagger assigns probabilities for every possible word and tag sequence and outputs the most likely tag sequence. The [Brill tagger](https://www.nltk.org/api/nltk.tag.html?#module-nltk.tag.brill) is a transformational tagger. It starts the training process with a baseline tagger and learns more parsing rules from word and tag sequence samples.

To perform this task we will need to train these two taggers using data from a specific domain and test their accuracy in predicting tag sequences from data belonging to the same domain and data from a different domain. The training process require to tune the tagger, that means we will have to examine different parameter sets and pick the one which yields the best tagging accuracy. Some examples of tunable parameters are the estimator function for the HMM tagger and the baseline rules template for the Brill tagger. 

## Input: POS tagged sentences
This task’s input data are part-of-speech tagged sentences from [The Georgetown University Multilayer Corpus (GUM)](http://corpling.uis.georgetown.edu/gum/). This corpus contains texts from different web sources and is annotated with several linguistic features, including POS tags. We are going to work with three types of text from this dataset. The training and in-domain testing sentences are composed of travel and how-to guides, these two text types are grouped together because they have similar words and sentence structure. The out-of-domain testing data contains fiction texts.

The train texts and the two sets of testing texts follow the same structure, a token and its corresponding [Penn Treebank](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) POS tag per line. Sentences are separated with an extra line between them.

## Acknowledgement

This is a solution to Assignment 4 for CMPUT 497 - Intro to NLP at the University of Alberta, created during the Fall 2020 semester. Yashar Kor: yashar@ualberta.ca, Thomas Maurer: tmaurer@ualberta.ca
