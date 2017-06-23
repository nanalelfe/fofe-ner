import glob, os

def OntoNotes(directory):
    """
    Parameters
    ----------
        directory: str
            directory in which the OntoNotes project is located
        files : str
            path to a file containing all of the paths to files containing NER-annotated
            data

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

    ret = []

    for filename in glob.glob(os.path.join(directory, "cnn_0160.v4_gold_conll")):
        textfile = open(filename, "r")
        for line in textfile:
            tokens = line.strip().split
            ret.append(token)
    return ret



if __name__ == '__name__':
    print(OntoNotes("/eecs/research/asr/quanliu/Datasets/CoNLL2012/data/development/conll"))