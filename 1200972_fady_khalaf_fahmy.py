'''
Created on Apr 8, 2017

@author: fady
'''
import numpy as np
import operator
from nltk.corpus import stopwords

trainData = open("data/train")
testData = open("data/test")


def wordCondProp(word, target):
    if target == "spam":
        if SpamText.get(word) == None:
            return 1.0/(spamWordsCount+len(distinctWords))
        else:
            return float(SpamText[word]+1)/(spamWordsCount+len(distinctWords))
    elif target == "ham" :
        if hamText.get(word) == None:
            return 1.0/(hamWordsCount+len(distinctWords))
        else:
            return float(hamText[word]+1)/(hamWordsCount+len(distinctWords))


#my global variables used throughout program..
spamCounter = 0   # number of spam mails in training data.
hamCounter = 0    # number of non-spam mails in training data.
SpamText = dict() # dict of spam words with their frequencies.
hamText = dict()  # dict of non-spam with their freuqncies.
spamWordsCount = 0   # spam words appeard in spam mails, including multiple occurances
hamWordsCount = 0    # non-spam words appeard in spam mails, including multiple occurances
distinctWords = set() # distinct words in all mails.
stops = set(stopwords.words('english'))  

#starting code
for line in trainData:
    words = line.split()
    vocabs = words[2::2]
    occurances = words[3::2]
    if words[1] == "spam":
        spamCounter += 1
        for index,word in enumerate(vocabs):
            if word not in stops:
                if SpamText.get(word) == None : 
                    SpamText[word] = int(occurances[index])
                else:
                    SpamText[word] += int(occurances[index])
                spamWordsCount += int(occurances[index])
                distinctWords.add(word)
    else:
        hamCounter += 1
        for index,word in enumerate(vocabs):
            if word not in stops:
                if hamText.get(word) == None : 
                    hamText[word] = int(occurances[index])
                else:
                    hamText[word] += int(occurances[index])
                hamWordsCount += int(occurances[index])
                distinctWords.add(word)
            
# printing top 5 words given the document is spam
sortedList = sorted(SpamText.items(), key = operator.itemgetter(1))
sortedList = sortedList[-1::-1]
sortedList = sortedList[:5]
print "the top most 5 words given spam document"
print sortedList

# printing top 5 words given the document is spam
sortedList = sorted(hamText.items(), key = operator.itemgetter(1))
sortedList = sortedList[-1::-1]
sortedList = sortedList[:5]
print "the top most 5 words given non-spam document"
print sortedList             

# computing priors
posPrior = float(hamCounter)/(hamCounter+spamCounter)
negPrior = float(spamCounter)/(hamCounter+spamCounter)

    
# computing accuracy of classification on test Data
sents = []
original_classification = []
my_classification = []
posMail = 0
negMail = 0
for line in testData:
    words = line.split()
    original_classification.append(words[1])
    words = words[2:]
    sents.append(words)
for sent in sents:
    posMail = np.log(posPrior)
    negMail = np.log(negPrior)
    words = sent[::2]
    cnts = sent[1::2]
    for w,c in zip(words,cnts):
        posMail += np.log(int(c)) + np.log(wordCondProp(w, "ham"))
        negMail += np.log(int(c)) + np.log(wordCondProp(w, "spam"))
    my_classification.append("ham") if posMail > negMail else my_classification.append("spam")
similar = 0
disimilar = 0
for i,j in zip(my_classification, original_classification):
    if i == j:
        similar +=1
    else:
        disimilar += 1
print float(similar)/(similar+disimilar)

trainData.close()
testData.close()




