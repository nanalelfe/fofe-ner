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

    sentence_end = False
    caught = [False, None]

    for filename in glob.glob(os.path.join(directory), '*'):
        textfile = open(filename, "r")
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


if __name__ == '__main__':
    generator = OntoNotes("/eecs/research/asr/quanliu/Datasets/CoNLL2012/data/development/conll")
    for line in generator:
        print(line)