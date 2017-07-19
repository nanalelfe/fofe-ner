from LinkingUtil import LoadED
from itertools import izip, islice, imap, combinations, chain

def KBP(filename):
    generator = imap( lambda x: x[:4], LoadED( filename ) )
    for item in generator:
        yield item


if __name__ == '__main__':
    generator = KBP("/home/chwang/nana/EDL-DATA/KBP-EDL-2015/eng-train-parsed")
    for item in generator:
        print(item)