#!/home/chwang/anaconda2/envs/tensorflow/bin/python

from itertools import izip, islice, imap, combinations, chain
import logging
import thread, threading, os
import Queue
from LinkingUtil import LoadED

logger = logging.getLogger()

def LoadED( rspecifier, language = 'eng' ):
    print("in Load ED")

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

    if os.path.isfile( rspecifier ):
        with codecs.open( rspecifier, 'rb', 'utf8' ) as fp:
            processed, original = fp.read().split( u'=' * 128, 1 )
            original = original.strip()

            # texts, tags, failures = processed.split( u'\n\n\n', 2 )
            texts = processed.split( u'\n\n\n' )[0]
            for text in texts.split( u'\n\n' ):
                parts = text.split( u'\n' )
                # assert len(parts) in [2, 3], 'sentence, offsets, labels(optional)'
                if len( parts ) not in [2, 3]:
                    logger.exception( text )
                    continue

                sent, boe, eoe, target, mids, spelling = parts[0].split(u' '), [], [], [], [], []
                offsets = map( lambda x : (int(x[0]), int(x[1])),
                               [ offsets[1:-1].split(u',') for offsets in parts[1].split() ] )
                assert len(offsets) == len(sent), rspecifier + '\n' + \
                        str( offsets ) + '\n' + str( sent ) + '\n%d vs %d' % (len(offsets), len(sent))

                if len(parts) == 3:
                    for ans in parts[-1].split():
                        try:
                            begin_idx, end_idx, mid, mention1, mention2 = ans[1:-1].split(u',')
                            target.append( entity2cls[str(mention1 + u'_' + mention2)] )
                            boe.append( int(begin_idx) )
                            eoe.append( int(end_idx) )
                            mids.append( mid )
                            spelling.append( original[ offsets[boe[-1]][0] : offsets[eoe[-1] - 1][1] ] )
                        except ValueError as ex1:
                            logger.exception( rspecifier )
                            logger.exception( ans )
                        except KeyError as ex2:
                            logger.exception( rspecifier )
                            logger.exception( ans )

                        try:
                            assert 0 <= boe[-1] < eoe[-1] <= len(sent), \
                                    '%s  %d  ' % (rspecifier.split('/')[-1], len(sent)) + \
                                    '  '.join( str(x) for x in [sent, boe, eoe, target, mids] )
                        except IndexError as ex:
                            logger.exception( rspecifier )
                            logger.exception( str(boe) + '   ' + str(eoe) )
                            continue
                    assert( len(boe) == len(eoe) == len(target) == len(mids) )

                # move this part to processed_sentence
                # if language == 'eng':
                #     for i,w in enumerate( sent ):
                #         sent[i] = u''.join( c if 0 <= ord(c) < 128 else chr(0) for c in list(w) )
                yield sent, boe, eoe, target, mids, spelling


    else:
        for filename in os.listdir( rspecifier ):
            for X in LoadED( os.path.join( rspecifier, filename ), language ):
                yield X

def KBP(path, iflytek=None):
    files = list(os.listdir(path))[:]
    if iflytek is not None:
        files += list(os.listdir(iflytek))[:]
    while len(files) != 0:
        filename = files[0]
        while True:
            queueLock.acquire()
            print("KBP acquired the lock")
            if not kbp_queue.full():
                print("Putting file " + filename + " info the queue.")
                generator = LoadED(filename)
                print("LoadED generated")
                kbp_queue.put(generator)
                print("Put into queue")
                files.remove(filename)
                queueLock.release()
                break
            print("Almost there")

            queueLock.release()
            print("KBP released the lock")
    while True:
        if not kbp_queue.full():
            kbp_queue.put(None)
            break

def consumer():
    while True:
        queueLock.acquire()
        print("consumer acquired the lock")
        if not kbp_queue.empty():
            next_gen = kbp_queue.get()
            if next_gen is None:
                print("Returned None.")
                break
            print("Got the next generator.")
        queueLock.release()
        print("consumer released the lock")


if __name__ == '__main__':
    print("starting now")

    BUF_SIZE = 2
    kbp_queue = Queue.Queue(BUF_SIZE)
    queueLock = threading.Lock()

    thread.start_new_thread(KBP, ("/home/chwang/nana/mock",))
    # try:
    #     threading.start_new_thread(KBP, ("/home/chwang/nana/EDL-DATA/KBP-EDL-2015/eng-train-parsed",))

    # except:
    #     print("Error: unable to start thread")

    consumer()

    # generator = KBP("/home/chwang/nana/EDL-DATA/KBP-EDL-2015/eng-train-parsed")
    # for item in generator:
    #     print(item)
