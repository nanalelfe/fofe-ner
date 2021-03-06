from scipy.sparse import csr_matrix
from Queue import Queue
from threading import Thread
from itertools import izip, islice, imap, combinations, chain
from hanziconv import HanziConv
import numpy, re, random, logging, codecs, copy, glob, os
from lxml import etree
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

logger = logging.getLogger()


def OntoNotes(directory):
    """
    Parameters
    ----------
        directory: str
            directory in which the parsed data is located

    Yields
    ------
        sentence  : list of str
            original sentence
        ner_begin : list of int
            start indices of NER, inclusive
        ner_end   : list of int
            end indices of NER, excusive
        ner_label : list of int
            The entity type of sentence[ner_begin[i]:ner_end[i]] is label[i]
    """

    entity2cls = {
        # OntoNotes labels
        'PERSON': 0,
        'FAC': 1,
        'ORG': 2,
        'GPE': 3,
        'LOC': 4,
        'PRODUCT': 5,
        'DATE': 6,
        'TIME': 7,
        'PERCENT': 8,
        'MONEY': 9,
        'QUANTITY': 10,
        'ORDINAL': 11,
        'CARDINAL': 12,
        'EVENT': 13,
        'WORK_OF_ART': 14,
        'LAW': 15,
        'LANGUAGE': 16,
        'NORP': 17
    }

    sentence, ner_begin, ner_end, ner_label = [], [], [], []

    sentence_end = False
    caught = [False, None]

    for filename in glob.glob(os.path.join(directory, "*gold*")):
        with codecs.open( filename, 'rb', 'utf8' ) as textfile:
            for line in textfile:
                tokens = line.strip().split()
                if len(tokens) > 5:
                    ne = tokens[10]
                    word  = tokens[3]
                    sentence.append(word)
                    if ne != '*':
                        if ne[-1] == '*':
                            ne = ne.strip('(').strip('*')
                            caught[0] = True
                            caught[1] = len(sentence) - 1
                            ner_begin.append(len(sentence) - 1)
                            ner_label.append(entity2cls[ne])
                        elif (ne[0] == '(' and ne[-1] == ')'):
                            ne = ne.strip('(').strip(')')
                            ner_begin.append(len(sentence) - 1)
                            ner_end.append(len(sentence))
                            ner_label.append(entity2cls[ne])
                        elif ne == '*)':
                            ner_end.append(len(sentence))
                            caught[0] = False
                            caught[1] = None

                elif len(sentence) > 0:
                    yield sentence, ner_begin, ner_end, ner_label
                    sentence, ner_begin, ner_end, ner_label = [], [], [], []


def PredictionParser( sample_generator, result, ner_max_length, n_label_type = 4 ):
    """
    This function is modified from some legancy code. 'table' was designed for 
    visualization. 

    Parameters
    ----------
        sample_generator : iterable
            Likes of CoNLL2003 and KBP2015

        result : str
            path to a filename where each line is predicted class (in integer) 
            followed by the probabilities of each class

        ner_max_length: int
            maximum length of mention

        reinterpret_threshold: float
            NOT USED ANYMORE

        n_label_type : int
            numer of memtion types

    Yields
    ------
        s : list of str
            words in a sentence

        table : numpy.ndarray
            table[i][j - 1] is a pair of string represnetation of predicted class
            and the corresponding probability of s[i][j]

        estimate : tuple
            (begin,end,class) triples

        actual : tuple
            (begin,end,class) triples
    """
    if n_label_type == 4:
        idx2ner = [ 'PER', 'LOC', 'ORG', 'MISC', 'O' ]

    elif n_label_type == 18:

        idx2ner = ['PERSON', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'DATE', 'TIME', 'PERCENT',
                    'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'EVENT', 'WORK_OF_ART', 'LAW',
                    'LANGUAGE', 'NORP', 'O']
    else:
        # idx2ner = [ 'PER_NAM', 'PER_NOM', 'ORG_NAM', 'GPE_NAM', 'LOC_NAM', 'FAC_NAM', 'TTL_NAM', 'O'  ]
        idx2ner = [ 'PER_NAM', 'ORG_NAM', 'GPE_NAM', 'LOC_NAM', 'FAC_NAM',
                    'PER_NOM', 'ORG_NOM', 'GPE_NOM', 'LOC_NOM', 'FAC_NOM',
                    'O' ]  

    # sg = SampleGenerator( dataset )
    if isinstance(result, str):
        fp = open( result, 'rb' )
    else:
        fp = resultf
    sg = sample_generator

    # @xmb 20160717
    lines, cnt = fp.readlines(), 0

    for i in range(1000000):
        s, boe, eoe, cls = sg.next()
        actual = set( zip(boe, eoe, cls) )

        table = numpy.empty((len(s), len(s)), dtype = object)
        table[:,:] = None #''
        estimate = set()
        actual = set( zip(boe, eoe, cls) )
        subwords = []

        for i in xrange(len(s)):
            for j in xrange(i + 1, len(s) + 1):
                if j - i <= ner_max_length:
                    # @xmb 20160717
                    # line = fp.readline()
                    line = lines[cnt]
                    cnt += 1

                    tokens = line.strip().split()
                    predicted_label = int(tokens[1])
                    actual_label = int(tokens[0])
                    all_prob = numpy.asarray([ numpy.float32(x) for x in tokens[2:] ])

                    if (predicted_label != actual_label):
                        predicted, probability = idx2ner[predicted_label], all_prob
                        subwords.append((s[i:j], idx2ner[actual_label], idx2ner[predicted_label]))

        if len(subwords) > 0:
            yield s, subwords

    if isinstance(result, str):
        fp.close()


def PrettyPrint( sample_generator, result, ner_max_length, n_label_type):
    parser = PredictionParser(sample_generator, result, ner_max_length, n_label_type)

    for sentence, subwords in parser:
        print("==================================================================================")
        text = '\n' + ' '.join(sentence) + '\n'
        print(str(text))
        print("MENTION \t ACTUAL LABEL \t PREDICTED LABEL\n")
        for subword, actual_label, predicted_label in subwords:
            print(str(' '.join(subword)) + '\t' + actual_label + '\t' + predicted_label + '\n')


if __name__ == "__main__":

	test_path = "/eecs/research/asr/quanliu/Datasets/CoNLL2012/data/test/conll"
	testing_file = "./ontonotes-result/ontonotes-test.predicted"

	parser = PrettyPrint(OntoNotes(test_path), testing_file, 7, n_label_type = 18)

	for element in parser:
		print(element)





