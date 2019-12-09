#!/usr/bin/env python

'''
##############################################
CS B551 Spring 2017, Assignment #4

Created on 2017. 4. 25.

Your names and user ids: 
Kwangwuk Lee, lee2074
Xueqiang Wang, xw48

1. Our goal is to figure out P(topic|words) for each file in test directories
2. Since we just figure out the one that has the highest probability. 
3. In training part, We need to calculate P(topic | words) for each topic and each word
4. To achieve no.3, we read all words in each topic and calculate P(words | topic)
5. Since we just want to know the highest probability, we can ignore common factors to easily calculate.
   1) Each topic has equal probability : 1/20, So we can ignore P(topic)
      P(words|topic) / P(words)
   2) P(words) are also common factor. 
      P(topic|words)
6. We assume that words are in conditional independence so that we can simply get the probability as below
   P(words|topic) = P(w1|topic) * P(w2|topic) * P(w3|topic) * ... * P(wo|topic)

'''
"""
               atheism        autos          baseball       christian      crypto         electronics    forsale        graphics           guns       hockey         mac            medical        mideast        motorcycles    pc             politics       religion       space          windows        xwindows
atheism        191            0              1              12             1              0              0              0              2              1              0              2              3              1              1              2              13             1              0              1
autos          0              358            0              0              1              5              9              2              1              0              1              0              0              15             2              2              0              0              0              0
baseball       0              0              143            0              0              0              1              0              0              6              0              2              0              1              0              0              0              0              0              0
christian      3              0              1              368            0              0              0              1              0              2              0              3              0              1              0              2              15             1              1              0
crypto         1              1              0              1              367            2              4              2              8              1              2              1              0              0              3              0              1              1              0              1
electronics    1              4              0              0              6              126            1              5              0              0              5              6              0              2              10             1              1              4              1              0
forsale        0              13             0              0              0              7              264            4              1              0              12             1              0              3              15             0              0              2              3              0
graphics       1              0              2              1              11             11             6              301            0              0              16             1              0              1              12             0              0              7              6              13
guns           0              0              0              1              5              1              0              0              288            0              1              2              1              1              0              10             8              1              0              0
hockey         0              0              3              0              2              0              2              0              1              388            1              0              0              1              0              1              0              0              0              0
mac            0              2              1              0              2              12             8              9              0              0              328            1              0              1              18             0              0              3              0              0
medical        3              8              0              4              1              7              6              5              2              0              0              334            3              3              3              9              1              6              1              0
mideast        10             0              1              1              0              0              0              1              2              0              0              0              345            1              0              15             0              0              0              0
motorcycles    0              8              0              0              0              0              3              0              0              0              0              0              0              386            1              0              0              0              0              0
pc             0              0              0              0              2              24             15             7              0              0              27             0              0              0              305            0              0              0              11             1
politics       7              1              1              0              4              0              0              1              76             0              0              2              1              0              0              199            10             8              0              0
religion       40             2              0              15             0              0              0              3              16             0              0              1              2              0              0              7              161            4              0              0
space          3              1              0              2              2              3              1              3              3              0              1              4              0              0              2              4              0              361            0              4
windows        1              2              3              2              9              3              12             58             2              0              24             3              0              2              70             12             3              5              164            19
xwindows       0              1              1              0              2              3              4              56             0              0              6              0              0              2              10             0              0              5              5              300
Accuracy : 0.826226167952
"""


       



import os
import sys
import string
import operator
import json
import math
import operator



global except_list
except_list = ['is', 'was', 'are', 'were', 'be', 'being',
              'who', 'when', 'where', 'what', 'why', 'whose', 'whatever', 'whom', 'which',
              'the', 'as', 'of', 'to', 'over', 'off', 'up', 'down', 'above', 'below', 'into', 'these', 'those', 'in', 'on', 'by',
              'i', 'my', 'me', 'mine', 'you', 'your', 'yours', 'he', 'his', 'him', 'she', 'her', 'hers', 'it', 'its', 'we', 'our', 'us', 'ours', 'that', 'this',
              'and', 'or', 'however', 'other']


def extract_letters(strLine):
    strList = []
    for OneWord in strLine :
        if( OneWord in string.letters ) :
            strList.append( OneWord )
    return ''.join(strList).lower()

def print_confusion_matrix(real_topic, predicted_topic, topics):
    """if '.DS_Store' in topics:
        topics.remove('.DS_Store')
    accuracy_count = 0
    print '\n\n================ Confusion Maxrix =================='
    title = '\t\t'
    for i in topics:
        if len(i) > 7:
            title += i + '\t' 
        else:
            title += i + '\t\t'
    print title
    
    for i in topics:
        if len(i) > 7:
            line = i + '\t'
        else:
            line = i + '\t\t'
        for j in topics:
            if max([int(result_data[i][b]) for b in topics]) == int(result_data[i][j]):
                value = '*' + str(int(result_data[i][j])) + '*'
                if i == j:
                    accuracy_count += 1
            else:
                value = str(int(result_data[i][j]))
            if len(value) > 7:
                line += value + '\t'
            else:
                line += value + '\t\t'
            
        print line"""
    
    accuracy_count = 0
    result_data = dict()
    for i in topics:
        result_data[i] = dict()
        for j in topics:
            result_data[i][j] = 0
    
    for file in real_topic.keys():
        result_data[real_topic[file]][predicted_topic[file]] += 1
        if real_topic[file] == predicted_topic[file]:
            accuracy_count += 1
    
    topic_str = '\t\t'
    for i in topics:
        #topic_str += i + '\tt'
        if len(i) > 7:
            topic_str += i + '\t'
        else:
            topic_str += i + '\t\t'
    print topic_str
        
    for i in topics:
        result_str = ''
        if len(i) > 7:
            result_str += i + '\t'
        else:
            result_str += i + '\t\t'
        for j in topics:
            result_str += str(result_data[i][j]) + '\t\t'
        print result_str
    
    
    
    accuracy = float(accuracy_count) / float(len(real_topic.keys()))
    
    print 'Accuracy : ' + str(accuracy) 
            
        
        
    
    
    #print 'accuracy : ' + str(float(accuracy_count)/float(len(topics)) * 100) + '%'
        
        
    

def train_dataset(dir_name):
    trained_data = dict()
    result = dict()
    topics = os.listdir(dir_name)
    for topic in topics:
        if topic == '.DS_Store':
            continue
        print 'processing ' + topic + '...'
        trained_data[topic] = []
        for file in os.listdir(os.path.join(dir_name, topic)):
            if file == '.DS_Store':
                continue
            for word in open(os.path.join(dir_name, topic, file), 'r').read().split():
                temp = extract_letters(word)
                if len(temp) > 1:# and temp not in except_list: 
                    trained_data[topic].append(temp)
                    
        count_dict = dict()
        for word in trained_data[topic]:
            if not word in count_dict.keys():
                count_dict[word] = float(trained_data[topic].count(word))/float(len(trained_data[topic]))
        result[topic] = count_dict

    with open('distinctive_words.txt', 'w') as f:
        print "making the data structure as distictive_words.txt"
        json.dump(result, f)
        
def test(dir_name):
    with open(sys.argv[3], 'r') as f:
        try:
            model = json.load(f)
        except ValueError:
            model = {}
            
    real_topic = dict()
    predicted_topic = dict()

    test_data = dict()
    result_data = dict()        
    topics = os.listdir(dir_name)
    if '.DS_Store' in topics:
        topics.remove('.DS_Store')
    
    for topic in topics:
        test_data[topic] = []
        print 'processing ' + topic + '...'
        for file in os.listdir(os.path.join(dir_name, topic)):
            if file == '.DS_Store':
                continue
            result_data = dict()
            for i in topics:
                result_data[i] = 0
            for word in open(os.path.join(dir_name, topic, file), 'r').read().split():
                temp = extract_letters(word)
                if len(temp) > 1 and temp not in except_list:
                    for i in topics:
                        try:
                            #log(a*b) = log(a) + log(b) 
                            result_data[i] += math.log(model[i][temp])
                        except KeyError:
                            #If there does not exist the word in the model the probability will be always 0.
                            #Therefore, to smooth it, I give very small value to the result when the values not existing in the model are shown up.
                            result_data[i] += math.log(0.0001/float(len(model[i])))
            real_topic[file] = topic
            predicted_topic[file] = max(result_data.iteritems(), key=operator.itemgetter(1))[0]
            #print topic + '  ' + predicted_topic[file]
            
            
    print_confusion_matrix(real_topic, predicted_topic, topics)
                
                
            
    
        
    #print_confusion_matrix(result_data, topics)    

def main():
    if len(sys.argv) != 4:
        print "check the parameters : topics.py mode dataset_directory model-file"
        exit()

        
    if sys.argv[1] == 'train':
        train_dataset(sys.argv[2])
    elif sys.argv[1] == 'test':
        test(sys.argv[2])

        
        
        
    

main()