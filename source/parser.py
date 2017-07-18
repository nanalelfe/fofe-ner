import cython, numpy
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map as ordered_map
from cython.operator cimport dereference, preincrement

from Queue import Queue
from threading import Thread
from itertools import izip, islice, imap, combinations, chain
from hanziconv import HanziConv
import numpy, re, random, logging, codecs, copy, glob, os
from lxml import etree



def KBP2015( filename ):
    """
    Parameters
    ----------
        filename : str
            path to directory containing NER-annotated Gigaword

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
    cdef int i, cls, cnt
    cdef vector[vector[int]] buffer_stack
    cdef list entity_begin
    cdef list engity_end
    cdef list entity_label
    buffer_stack.resize( 10 )

    logger.info( 'According to Liu, TTL_NAM are all labeled as PER_NOM.' )
    entity2cls = {  # KBP2015 label
                    'PER_NAM' : 0, 
                    'PER_NOM' : 5, 
                    'ORG_NAM' : 1, 
                    'GPE_NAM' : 2, 
                    'LOC_NAM' : 3, 
                    'FAC_NAM' : 4, 
                    'TTL_NAM' : 5,

                    # iflytek label
                    'PER_NAME' : 0,  
                    'ORG_NAME' : 1, 
                    'GPE_NAME' : 2, 
                    'LOC_NAME' : 3, 
                    'FAC_NAME' : 4, 
                    'PER_NOMINAL' : 5,
                    'ORG_NOMINAL' : 6,
                    'GPE_NOMINAL' : 7,
                    'LOC_NOMINAL' : 8,
                    'FAC_NOMINAL' : 9,
                    'PER_NOM' : 5,
                    'ORG_NOM' : 6,
                    'GPE_NOM' : 7,
                    'LOC_NOM' : 8,
                    'FAC_NOM' : 9,
                    'TITLE_NAME' : 5,
                    'TITLE_NOMINAL' : 5
                } 

    with codecs.open( filename ) as text_file:
        for line in text_file:
            line = line.strip()

            # bar3idx = line.rfind( '|||' )
            # sentence = [ tokens.split('#')[1] for tokens in line[:bar3idx].strip().split() ]
            # label = line[bar3idx + 3:].split()
            sentence, label = line.rsplit( '|||', 1 )
            sentence = [ tokens.split('#')[1] for tokens in sentence.strip().split() ]
            label = label.split()

            entity_begin, entity_end, entity_label = [], [], []

            cnt = 0
            for l in label:
                if l == 'X':
                    cnt += 1
                elif l.startswith( '(' ):
                    buffer_stack[ entity2cls[ l[1:] ] ].push_back( cnt )
                elif l.startswith( ')' ):
                    cls = entity2cls[ l[1:] ]
                    entity_begin.append( buffer_stack[cls].back() )
                    entity_end.append( cnt )
                    entity_label.append( cls )
                    buffer_stack[cls].pop_back() 

            if cnt > 0:
                assert cnt == len(sentence)
                for i in range( buffer_stack.size() ):
                    assert buffer_stack[i].size() == 0
                yield sentence, entity_begin, entity_end, entity_label


if __name__ == '__main__':
    generator = KBP2015("/local/scratch/nana/iflytek-clean-eng/checked")
    for item in generator:
        print(item)