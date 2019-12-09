###################################
# CS B551 Fall 2016, Assignment #3
#
# Your names and user ids: 
# Kwangwuk Lee, lee2074
# Xueqiang Wang, xw48
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Part 1: Markov Model
# In this part, we consider a sentence as Markov Chains. Each state is corresponded with a word. P(W_i+1=word2|W_i=word1) represents the probability of word2 given 
# the last word is word1.
# 1) based on training data, we calculate P(W1) and P(W_i+1|W_i), in method 'preprocessing';
# 2) using markov model to generate sentences. In this part, we adopt greedy algorithm. Firstly, we choose top initial words. And then, we choose the next word, which 
#    would maximize P(W_i+1|W_i). Generation terminates when an ending word appear (i.e., '.', '!', '?').
# 3) for the grammar checker, we calculate probability of a sentence and other candidates and give our suggestions.
# We made assumptions in implementation: for example, if a probability could not be derived from training data, we assume it is 0.00001 (defined as INFINITY in code.)
#
#
# Part 2
# In this part, we consider the part-of-speech tagging problem as a Bayesian network. When calculating probabilities, we follow instructions given in 'a3.pdf'. In part 2.2,
# we use naive bayes algorithm to calculate P(S_i|W_i) based on P(W|S), P(S) and P(W).Then we consider dependancies between nearby words, specifically, we try to 
# maximize argmax(P(S_i+1|S_i)P(S_i|W_i)).
# In Viterbi algorithm, we firstly construct the probability graph and then choose the most probable end tag. Then do a backward tracing on the graph. 
# During all above implementation, we assume the probability of non-existing party as 0.00001 (defined as INFINITY in code.)
#
# Note the probabilities, including logarithm of the posterior probability, vary with our assumption of probability of unknown <word, tag of speech> pair. 
####

import random
import math
import operator
from heapq import heappush, heappop
from sets import Set
import copy
sorted_w1 = []
wi_wiplus_1 = {}
INFINITY = 11.51 # -log(0.00001)

# read training file and calculate probability
def preprocessing(fname):
    global wi_wiplus_1
    global sorted_w1
    print 'processing training data'
    train_file = open(fname, 'r')
    train_data = []
    for line in train_file:
        data = tuple([ w.lower() for w in line.split() ])
        train_data += [(data[0::2], data[1::2])]

    w1 = [ item[0][0] for item in train_data ]
    sorted_w1 += sorted(((-math.log(float(w1.count(e)) / float(len(w1))), e) for e in set(w1)))
    for item in train_data:
        words = item[0]
        for i in range(0, len(words) - 1):
            if words[i] in wi_wiplus_1:
                if words[i + 1] in wi_wiplus_1[words[i]]:
                    wi_wiplus_1[words[i]][words[i + 1]] += 1
                else:
                    wi_wiplus_1[words[i]][words[i + 1]] = 1
            else:
                freq = {}
                freq[words[i + 1]] = 1
                wi_wiplus_1[words[i]] = freq

    for key in wi_wiplus_1.keys():
        key_sum = sum(wi_wiplus_1[key].values())
        for subkey in wi_wiplus_1[key].keys():
            wi_wiplus_1[key][subkey] = -math.log(float(wi_wiplus_1[key][subkey]) / float(key_sum))

        wi_wiplus_1[key] = sorted(wi_wiplus_1[key].items(), key=operator.itemgetter(1))


def do_part12(argv):
    if len(argv) != 3 or argv[1] != 'part1.2':
        print '    ./label.py part1.2 training_file'
        sys.exit(-1)
    preprocessing(argv[2])
    sentence_terminator = ['.', '?', '!']
    used_pair = []
    for i in range(0, 5):
        first_word = sorted_w1[i][1]
        sentence = [first_word]
        cur_word = first_word
        while True:
            try:
                for j in range(0, len(wi_wiplus_1[cur_word])):
                    next_word = wi_wiplus_1[cur_word][j][0]
                    if (cur_word, next_word) not in used_pair:
                        used_pair.append((cur_word, next_word))
                        break

                sentence.append(next_word)
                if next_word in sentence_terminator:
                    break
                cur_word = next_word
            except Exception as e:
                print e
                break

        print ' '.join(sentence)


def prob(sentence):
    prob = 0.0
    for i in sorted_w1:
        if i[1] == sentence[0]:
            prob += i[0]

    for i in range(0, len(sentence) - 1):
        wi = sentence[i]
        wi_plus1 = sentence[i + 1]
        item_prob = INFINITY
        if wi in wi_wiplus_1.keys():
            for item in wi_wiplus_1[wi]:
                if item[0] == wi_plus1:
                    item_prob = item[1]

        prob += item_prob

    return prob


def do_part13(argv):
    if len(argv) != 4 or argv[1] != 'part1.3':
        print '    ./label.py part1.3 training_file "Test sentence"'
        sys.exit(-1)
    preprocessing(argv[2])
    test_sentence = [ w.lower() for w in argv[3].split() ]
    confused_words = Set(['accept',
     'except',
     'affect',
     'effect',
     'allusion',
     'illusion',
     'allready',
     'already',
     'altogether',
     'alltogether',
     'ascent',
     'assent',
     'breath',
     'breathe',
     'capital',
     'capitol',
     'cite',
     'sight',
     'site',
     'complement',
     'compliment',
     'conscience',
     'conscious',
     'council',
     'counsel',
     'elicit',
     'illicit',
     'eminent',
     'immanent',
     'imminent',
     'its',
     "it's",
     'lead',
     'led',
     'lie',
     'lay',
     'lose',
     'loose',
     'passed',
     'past',
     'precede',
     'proceed',
     'principal',
     'principle',
     'quote',
     'quotation',
     'stationary',
     'stationery',
     'than',
     'then',
     'their',
     'there',
     "they're",
     'through',
     'threw',
     'thorough',
     'though',
     'thru',
     'to',
     'too',
     'two',
     'which',
     'that',
     'who',
     'whom'])
    confused_map = [['accept', 'except'],
     ['affect', 'effect'],
     ['allusion', 'illusion'],
     ['allready', 'already'],
     ['altogether', 'alltogether'],
     ['ascent', 'assent'],
     ['breath', 'breathe'],
     ['capital', 'capitol'],
     ['cite', 'sight', 'site'],
     ['complement', 'compliment'],
     ['conscience', 'conscious'],
     ['council', 'counsel'],
     ['elicit', 'illicit'],
     ['eminent', 'immanent', 'imminent'],
     ['its', "it's"],
     ['lead', 'led'],
     ['lie', 'lay'],
     ['lose', 'loose'],
     ['passed', 'past'],
     ['precede', 'proceed'],
     ['principal', 'principle'],
     ['quote', 'quotation'],
     ['stationary', 'stationery'],
     ['than', 'then'],
     ['their', 'there', "they're"],
     ['through',
      'threw',
      'thorough',
      'though',
      'thru'],
     ['to', 'too', 'two'],
     ['which', 'that'],
     ['who', 'whom']]
    candidates = []
    original_prob = prob(test_sentence)
    print 'Original:', ' '.join(test_sentence), '     Prob:', math.exp(-original_prob)
    heappush(candidates, (original_prob, test_sentence))
    for i in range(0, len(test_sentence)):
        if test_sentence[i] not in confused_words:
            continue
        should_replace = test_sentence[i]
        for confused_item in confused_map:
            if should_replace in confused_item:
                exist_candidates = copy.deepcopy(candidates)
                for exist_item in exist_candidates:
                    for replace_with in confused_item:
                        if should_replace == replace_with:
                            continue
                        sentence_cp = copy.deepcopy(exist_item[1])
                        sentence_cp[i] = replace_with
                        heappush(candidates, (prob(sentence_cp), sentence_cp))

    suggested = heappop(candidates)
    if suggested[1] == test_sentence or suggested[0] >= original_prob:
        print 'should REMAIN original sentence'
    else:
        print 'should REVISE original sentence'
        print 'New:', suggested[1], '     Prob:', math.exp(-suggested[0])

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    __PS1 = {}
    __PSiSi_plus1 = {}
    __PWiSi = {}
    __PSi = {}
    __PWi = {}

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling

    def posterior(self, sentence, label):
        prob = 0.0
        for i in range(0, len(sentence)):
            tag = label[i]
            word = sentence[i]

            for tag in self.__PSi.keys():
                p_si = self.__PSi[tag]

                p_w_si = INFINITY
                if word in self.__PWiSi[tag]:
                    p_w_si = self.__PWiSi[tag][word]

                p_wi = INFINITY
                if word in self.__PWi:
                    p_wi = self.__PWi[word]

                prob -= (p_w_si + p_si - p_wi)

        return prob

    # Do the training!
    #
    def train(self, data):
        s1 = [ item[1][0] for item in data ]
        for e in set(s1):
            self.__PS1[e] = -math.log(float(s1.count(e)) / float(len(s1)))

        each_word_weight = 1.0 / float(sum((len(item[0]) for item in data)))

        for item in data:
            tags = item[1]
            for tag_item in tags:
                if tag_item in self.__PSi:
                    self.__PSi[tag_item] += each_word_weight
                else:
                    self.__PSi[tag_item] = each_word_weight

            for i in range(0, len(tags) - 1):
                if tags[i] in self.__PSiSi_plus1:
                    if tags[i + 1] in self.__PSiSi_plus1[tags[i]]:
                        self.__PSiSi_plus1[tags[i]][tags[i + 1]] += 1
                    else:
                        self.__PSiSi_plus1[tags[i]][tags[i + 1]] = 1
                else:
                    freq = {}
                    freq[tags[i + 1]] = 1
                    self.__PSiSi_plus1[tags[i]] = freq

        for key in self.__PSiSi_plus1.keys():
            key_sum = sum(self.__PSiSi_plus1[key].values())
            for subkey in self.__PSiSi_plus1[key].keys():
                self.__PSiSi_plus1[key][subkey] = -math.log(float(self.__PSiSi_plus1[key][subkey]) / float(key_sum))

        for key in self.__PSi.keys():
            self.__PSi[key] = -math.log(self.__PSi[key])


        for item in data:
            words = item[0]
            tags = item[1]
            for word_item in words:
                if word_item in self.__PWi:
                    self.__PWi[word_item] += each_word_weight
                else:
                    self.__PWi[word_item] = each_word_weight

            for i in range(0, len(words)):
                if tags[i] in self.__PWiSi:
                    if words[i] in self.__PWiSi[tags[i]]:
                        self.__PWiSi[tags[i]][words[i]] += 1
                    else:
                        self.__PWiSi[tags[i]][words[i]] = 1
                else:
                    freq = {}
                    freq[words[i]] = 1
                    self.__PWiSi[tags[i]] = freq

        for key in self.__PWiSi.keys():
            key_sum = sum(self.__PWiSi[key].values())
            for subkey in self.__PWiSi[key].keys():
                self.__PWiSi[key][subkey] = -math.log(float(self.__PWiSi[key][subkey]) / float(key_sum))

        for key in self.__PWi:
            self.__PWi[key] = -math.log(self.__PWi[key])

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        tags = []
        probs = []
        for word in list(sentence):
            candidate_tags = []
            for tag_item in self.__PSi.keys():
                p_si = self.__PSi[tag_item]
                p_w_si = INFINITY
                if word in self.__PWiSi[tag_item]:
                    p_w_si = self.__PWiSi[tag_item][word]
                p_wi = INFINITY
                if word in self.__PWi:
                    p_wi = self.__PWi[word]
                heappush(candidate_tags, (p_w_si + p_si - p_wi, tag_item))

            real_tag = heappop(candidate_tags)
            tags.append(real_tag[1])
            probs.append(real_tag[0])

        return tags

    def hmm_ve(self, sentence):
        sentence_lst = list(sentence)
        tags = []
        for i in range(0, len(sentence_lst)):
            candidate_tags = []
            for tag_item in self.__PSi.keys():
                p_sj_sj_minus_1 = INFINITY
                if i == 0 and tag_item in self.__PS1:
                    p_sj_sj_minus_1 = self.__PS1[tag_item]
                if i != 0 and tag_item in self.__PSiSi_plus1[tags[i - 1]]:
                    p_sj_sj_minus_1 = self.__PSiSi_plus1[tags[i - 1]][tag_item]
                p_wi_sj = INFINITY
                if sentence_lst[i] in self.__PWiSi[tag_item]:
                    p_wi_sj = self.__PWiSi[tag_item][sentence_lst[i]]
                heappush(candidate_tags, (p_sj_sj_minus_1 + p_wi_sj, tag_item))

            tag_pair = heappop(candidate_tags)
            tags.append(tag_pair[1])

        return tags

    def hmm_viterbi(self, sentence):
        sentence_lst = list(sentence)
        tag_graph = []
        for layer in range(0, len(sentence_lst)):
            layer_boundary = len(tag_graph)
            for tag_item in self.__PSi.keys():
                if layer == 0:
                    prev = -1
                    if tag_item in self.__PS1:
                        s = self.__PS1[tag_item]
                    else:
                        s = INFINITY
                    if sentence_lst[layer] in self.__PWiSi[tag_item]:
                        s += self.__PWiSi[tag_item][sentence_lst[layer]]
                    else:
                        s += INFINITY
                    tag_graph.append((layer,
                     tag_item,
                     s,
                     prev))
                else:
                    choose_path = None
                    for i in range(layer_boundary - 1, -1, -1):
                        if tag_graph[i][0] != layer - 1:
                            break
                        last_tag = tag_graph[i][1]
                        last_prop = tag_graph[i][2]
                        prev = i
                        s = last_prop
                        if tag_item in self.__PSiSi_plus1[last_tag]:
                            s += self.__PSiSi_plus1[last_tag][tag_item]
                        else:
                            s += INFINITY
                        if sentence_lst[layer] in self.__PWiSi[tag_item]:
                            s += self.__PWiSi[tag_item][sentence_lst[layer]]
                        else:
                            s += INFINITY

                        if choose_path == None:
                            choose_path = (layer, tag_item, s, prev)
                        else:
                            if choose_path[2] > s:
                                choose_path = (layer, tag_item, s, prev)

                    tag_graph.append(choose_path)

        last_tag = tag_graph[len(tag_graph) - 1]
        for i in range(len(tag_graph) - 1, -1, -1):
            if tag_graph[i][0] < len(sentence_lst) - 1:
                break
            if tag_graph[i][2] < last_tag[2]:
                last_tag = tag_graph[i]

        reverse_tags = []
        reverse_tags.append(last_tag[1])
        cur_tag = last_tag
        while True:
            next_prev = cur_tag[3]
            if next_prev < 0:
                break
            cur_tag = tag_graph[next_prev]
            reverse_tags.append(cur_tag[1])

        print 'LEN', len(reverse_tags)
        return reverse_tags[::-1]

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for simplified() and complex() and is the marginal probability for each word.
    #
    def solve(self, algo, sentence):
        if algo == 'Simplified':
            return self.simplified(sentence)
        if algo == 'HMM VE':
            return self.hmm_ve(sentence)
        if algo == 'HMM MAP':
            return self.hmm_viterbi(sentence)
        print 'Unknown algo!'
