###################################
# CS B551 Fall 2015, Assignment #5
#
# Your names and user ids:
# Dipika Bandekar: dipiband
# Shwethambari Surendran : ssurendr
#
####
# Report:

# 1. Posterior function:
# -- Use Bayes law P(B/A) = P(A/B) * P(B) / P(A)
# -- We have passed the entire sentence and label tag associated with the sentence
# -- Consider each word and calculate P(W/S) * P(S2/S1)
# -- Take product of all and calculate the log value

# 1. train function:
# -- self.types : Create the list of all parts of speech (12 parts of speech)
# -- transidict : Transitional Probabilities for combination of tag given prior pos tag
# -- posdict : Part of Speech dictionary calculates the probibility of particular pos given all part of speech in training set
# -- pos1 : The probability of first word being a particular pos given the total occurrence of that particular pos
# -- worddict : The probability of word given its pos tag in the training data set

# 2. Naive Bayes:
# -- Calculate the product of pos1 * likelihood of word given that pos1 for all parts of speech
# -- Select the product with maximum probability
# -- If word is not present in the dictionary than likelihood would be taken as (1/len(worddict))

# 3. MCMC:
# -- Created the first particle by assigning random values to each word of first sentence
# -- Initial loop for all samples given by sample count
# -- Second loop for each word in given sentence
# -- While loop to check for each type ie pos
# -- a) first word in the sentence then we take product of (P(S1) * P(S2/S1) * P(W1/S1))
# -- b) Last word in the sentence then we take product of (P(W4/S4) * P(S4/S3)) when W4 is last word and S4 its pos
# -- c) Any other word say W3 product of (P(W3/S3) * P(S3/S2) * P(S4/S3))
# -- Then multipy all probabilities of a word given all parts of speech and store it in posprob dictionary
# -- posprob dictionary contains the probability of all parts of speech for a particular word
# -- Normalize all probabilities such that their values are between 0 - 1
# -- Then multiply the probability by 10 and pick up a random value from the list generated (for instance if 2 pos noun : 0.8 and adv : 0.2
#    then create a list (noun, noun, noun, noun, noun, noun, noun, noun, adv, adv ) and pick up randomly from the list
# -- Return 5 particles as output
# --Note: We have taken 100 samples intially and keeping count of pos for each word position in a dictionary and passing the dictionary to mcmc function to print part of speech with maximun count

# 4. Max Marginal:
# -- Using the dictionary of the previous function we are checking the count of each part of speech for a given word
# -- Return the pos which has occurred maximum number of times for a given word

# 5. Best Algorithm:
# -- We have taken the outputs of our viterbi algorithm as the best output as its accuracy is good. We tried taking the best of each algorithm but it was giving less accuracy as compared to viterbi

# 6. Viterbi Algorithm:
# -- Calculate the initial state, emission and transitional probabilities
# -- Initial state is P(S) * P(W/S) for all part of speech and select maximum, also keep track of path of traversal
# -- For all the other states P(S) * P(W/S) * P(S2/S1) for all part of speech and select the maximum, also keep track of path of traversal
# -- Backtrack using the path list to insure the correctness of path chosen

# Assumptions Problems faced:
# 1) Posterior function is taking excess time for execution. When tried to run without posterior function the test data(big) is running in approx 5 min for 100 samples
# 2) E.g. for 100 samples big data (approx 5 min without posterior function) and (approx 1 hour with posterior function in pycharm)
# 3) For an unknown word we are assuming the most frequent part of speech in all algorithms
# 4) 1000 samples is running fine for tiny data but it is taking time for big test data file


# Output for tiny data: (best output on running a couple of times for 1000 samples, complete output attached in tiny_output text file)

# ==> So far scored 3 sentences with 42 words.
#                   Words correct:     Sentences correct:
#   0. Ground truth:      100.00%              100.00%
#          1. Naive:       97.62%               66.67%
#        2. Sampler:       92.86%               33.33%
#   3. Max marginal:       97.62%               66.67%
#            4. MAP:       97.62%               66.67%
#           5. Best:       97.62%               66.67%
# ----

# Output for test data: (Output on running for 100 samples, complete output attached in test_output text file)


import random
import math
from collections import defaultdict
import time


class Solver:

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        #start = time.time()
        ct = 0
        lenwrddict = len(self.worddict)
        product = 1
        for val in [(((self.worddict[sentence[i]][label[i]]*self.pos1[label[i]]))*self.transidict[label[i]][label[i-1]]) if sentence[i] in self.worddict.keys() and label[i] in self.worddict[sentence[i]].keys() else (((1/(float)(lenwrddict))*self.pos1[label[i]])*self.transidict[label[i]][label[i-1]]) for i in range(0, len(sentence))]:
            product *= val
        logval = math.log(product,10) if product != 0 else 0
        return logval

    # Do the training!
    #
    def train(self, data):
        self.types = ['adj', 'adv', 'adp','conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
        self.count=defaultdict(float)
        for i in self.types:
            self.count[i]=0
        for value in data:
            for type in self.types:
                    self.count[type]=self.count[type]+int(str(value[1]).count(type))
        total = 0.0
        for i in self.count:
            total += self.count[i]

        # finding the part of speech probability pos
        self.posdict = defaultdict(float)
        for i in self.count.iterkeys():
            self.posdict[i]=round((self.count[i]/total), 4)

        # Finding transition probability
        self.transidict = defaultdict(float)
        for i in self.types:
         self.transidict[i] = {}
        for i in self.types:
            self.transidict[i] = {}

        for sentence in data:
             for types in self.types:
                 for k in range(0, len(sentence[1]) - 1):
                     if sentence[1][k] == types:
                         if sentence[1][k] in self.transidict and sentence[1][k+1] in self.transidict[sentence[1][k]]:
                             self.transidict[sentence[1][k]][sentence[1][k+1]] += 1
                         else:
                             self.transidict[sentence[1][k]][sentence[1][k+1]]=1

        for i in self.types:
           for j in self.types:
                            if j in self.transidict[i].keys():
                               self.transidict[i][j] = float(self.transidict[i][j])/float(self.count[i])
                            else:
                               self.transidict[i][j] = 0.0

        self.pos1 = defaultdict(float)
        for m in data:
            strn = m[1][0]
            if self.pos1[strn]:
                self.pos1[strn] += 1
            else:
                self.pos1[strn] = 1
        for ct in self.pos1.iterkeys():
            self.pos1[ct] = float(self.pos1[ct])/float(len(data))

        self.worddict = defaultdict(float)
        for m in data:
            for word in m[0]:
                self.worddict[word] = {}
        for m in data:
          for n, k in zip(m[0], m[1]):
                    if k in self.worddict[n].keys():
                       self.worddict[n][k] += 1
                    else:
                       self.worddict[n][k] = 1
        for i in self.worddict.iterkeys():
            for j in self.worddict[i].iterkeys():
                self.worddict[i][j]= float(self.worddict[i][j])/float(self.count[j])
    pass

    # Functions for each algorithm
    def naive(self, sentence):
        self.sequence = []
        for word in sentence:
            product = {}
            for i in self.types:
               if word in self.worddict and i in self.worddict[word]:
                    product[i] = self.pos1[i] * self.worddict[word][i] + 0.5
               else:
                    product[i] = self.pos1[i] * 1/len(self.worddict)
            maximum = max(product.iterkeys(), key=lambda k: product[k])
            self.sequence += [maximum]
        return[[self.sequence], []]

    def mcmc(self, sentence, sample_count):
        particle = []
        result = []
        self.totalcount = sample_count
        #start1 = time.time()
        self.typecount = defaultdict(dict)
        for word in range(0, len(sentence)):
            particle = particle + [random.choice(self.types)]
        for cnt in range(1, sample_count + 1):
            for position in range(0, len(particle)):
                count = 0
                posprob = defaultdict(float)
                while count < len(self.types):
                #assigning one part of speech to particle[x] say noun
                    tag = self.types[count]
                    probMultiply = []
                    if len(particle) == 1:
                        probMultiply.append(self.pos1[tag])
                    elif position == 0:
                        probMultiply.append(self.pos1[tag])
                        probMultiply.append(self.transidict[particle[1]][tag])
                    elif position == len(particle) - 1:
                         probMultiply.append(self.transidict[particle[position]][particle[position - 1]])
                    else:
                         probMultiply.append(self.transidict[particle[position]][particle[position - 1]])
                         probMultiply.append(self.transidict[particle[position + 1]][particle[position]])
                    word1 = sentence[position]
                    if word1 in self.worddict and tag in self.worddict[word1]:
                         probMultiply.append(self.worddict[word1][tag])
                    else:
                        probMultiply.append(0.0)
                    if bool(posprob[tag]) is False:
                      posprob[tag] = 1.0
                    for i in probMultiply:
                         posprob[tag] = posprob[tag] * i
                    count += 1
                #normalize the 12 values and pick up one value
                tot = sum(posprob.values())
                if tot != 0.0:
                     for j in posprob:
                         posprob[j] = round(posprob[j]/tot, 2)

                prob1 = max(posprob, key=posprob.get)
                if posprob[prob1] == 1.0:
                     particle[position] = prob1
                else:
                     for j in posprob:
                         posprob[j] = int(posprob[j] * 10)
                     final = []
                     for m in posprob:
                         if posprob[m] != 0:
                             for n in range(0, int(posprob[m])):
                                 final.append(m)
                         if len(final) == 0:
                             particle[position] = random.choice(self.types)
                         else:
                             particle[position] = random.choice(final)
                if particle[position] in self.typecount[position].keys():
                     self.typecount[position][particle[position]] = self.typecount[position][particle[position]] + 1
                else:
                     self.typecount[position][particle[position]] = 1
            if sample_count != 5 and cnt > sample_count - 6:
                result.append(particle)
            elif sample_count == 5:
                result.append(particle)
        #end1 = time.time()
        #print "for samples 50 time taken", (end1 - start1)
        return [result, []]

    def max_marginal(self, sentence):
        self.pos88, poscount = [], []
        for i in range(0, len(self.typecount)):
            temp11 = max(self.typecount[i], key=self.typecount[i].get)
            self.pos88.append(temp11)
            poscount.append(round((float(self.typecount[i][temp11])/float(self.totalcount)), 2))
        return [ [self.pos88], [poscount]]

    def viterbi(self, sentence):
        Vit=[{}]
        path={}
        self.maximumpos = max(self.posdict.keys(), key=lambda y: self.posdict[y])
        for y in range(0,len(self.types)):
          if sentence[0] in self.worddict and self.types[y] in self.worddict[sentence[0]]:
              Vit[0][self.types[y]] = (self.posdict[self.types[y]] * self.worddict[sentence[0]][self.types[y]])
              path[self.types[y]] = [self.types[y]]
          else:
            Vit[0][self.types[y]] =(self.posdict[self.maximumpos] * 1/len(self.worddict))
            path[self.types[y]] = [self.maximumpos]

         # Run Viterbi for t > 0
        for t in range(1, len(sentence)):
          Vit.append({})
          newpath = {}

          for y in self.types:
             if sentence[t] in self.worddict and y in self.worddict[sentence[t]]:
                (prob, state) = max(((Vit[t-1][y0] * self.transidict[y0][y] * self.worddict[sentence[t]][y]), y0) for y0 in self.types)
                Vit[t][y] = prob
                newpath[y] = path[state] + [y]
             else:
                (prob, state)= max(((Vit[t-1][y0] * 1/len(self.worddict)*1/len(self.worddict)) , y0) for y0 in self.types )
                Vit[t][y] = prob
                newpath[y] = path[state] + [y]

          # Don't need to remember the old paths
          path = newpath

         # Return the most likely sequence over the given time frame
        n = len(sentence) - 1
        (prob, state) = max((Vit[n][y], y) for y in self.types)
        self.viterbi22 = path[state]
        return [[path[state]], []]

    def best(self, sentence):
        '''
        start = time.time()
        mydictionary = defaultdict(dict)
        final = [self.sequence, self.pos88, self.viterbi22]
        for i in range(0, len(self.sequence)):
            for j in range(0, len(final)):
                if final[j][i] in mydictionary[i].keys():
                    mydictionary[i][final[j][i]] = mydictionary[i][final[j][i]] + 1
                else:
                    mydictionary[i][final[j][i]] = 1
        best11 = []
        for i in range(0, len(mydictionary)):
                    temp11 = max(mydictionary[i], key=mydictionary[i].get)
                    best11.append(temp11)
        end = time.time()
        #print "time taken", end - start
        '''
        return [[self.viterbi22], []]


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #    Most algorithms only return a single labeling per sentence, except for the
    #    mcmc sampler which is supposed to return 5.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for max_marginal() and is the marginal probabilities for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Naive":
            return self.naive(sentence)
        elif algo == "Sampler":
            return self.mcmc(sentence, 100)
        elif algo == "Max marginal":
            return self.max_marginal(sentence)
        elif algo == "MAP":
            return self.viterbi(sentence)
        elif algo == "Best":
            return self.best(sentence)
        else:
            print "Unknown algo!"