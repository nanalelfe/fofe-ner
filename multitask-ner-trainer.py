#!/home/chwang/anaconda2/envs/tensorflow/bin/python

#/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy, logging, argparse, time, copy, os, cPickle, sys
from subprocess import Popen, PIPE, call
from Queue import Queue
from threading import Thread
from math import floor
from random import shuffle, random
from itertools import product, chain


logger = logging.getLogger(__name__)

if __name__ == '__main__':

    #################
    ##  ARGUMENTS  ##
    #################
    # - Path to directory where the word embeddings are located
    # - Path to directory where the OntoNotes project is located
    # - Path to file containing all of the paths to the training data
    # - Character embedding dimension
    # - NER embedding dimension
    # - Character set size
    # - Size of fully connected layers after projection
    # - Mini-batch size
    # - Learning rate
    # - Momentum value when MomentumOptimizer is used
    # - Maximum number of iterations
    # - Feature choice (bitmask)
    # - Percentage of overlap examples used during training
    # - Percentage of disjoint example used during training
    # - Use or not use dropout
    # - Character-level forgetting factor
    # - Word-level forgetting factor
    # - Whether or not bow and context share a same word embedding
    # - Decoding algorithm, i.e. {1: highest-score-first, 2: longest-coverage-first, 3: subsumption-removal}
    # - Threshold
    # - Maximum length of NER candidate
    # - When gazetteer is used, True if 4-bit match or False 5-bit match
    # - dimension of z in the HOPE paper; 0 means not used
    # - Number of label types
    # - Kernel height
    # - Kernel depth
    # - Initialize method: uniform or gaussian
    # - Enable distant supervision
    # - Model (hopeless?)
    # - Invoke official evaluator when true
    # - Where to write conll2003-{valid,test}.predicted
    # - Is second pass
    # - GPU fraction

    parser = argparse.ArgumentParser()

    # - Path to directory where the word embeddings are located
    # Used for all datasets
    parser.add_argument('word_embedding', type=str,
                        help='word_embedding.{-case-insensitive, -case-sensitive}.word2vec are assumed')

    parser.add_argument('conll_datapath', type=str,
                        help='path to eng.{train, testa, testb} of CoNLL2003')

    # - Path to directory where the OntoNotes project is located
    parser.add_argument('ontonotes_datapath', type=str,
                        help='path to the preparsed OntoNotes dataset')

    parser.add_argument('kbp_train_datapath', type=str, help='path to the preparsed KBP training dataset')

    parser.add_argument('kbp_valid_datapath', type=str, help='path to the preparsed KBP valid dataset')

    parser.add_argument('kbp_test_datapath', type=str, help='path to the preparsed KBP test dataset')

    parser.add_argument('kbp_gazetteer', type=str, help='path to the kbp gazetteer file')

    parser.add_argument('iflytek_checked_eng', type=str, help='path to the preparsed iFLYTEK checked dataset')

    # - Character embedding dimension
    parser.add_argument('--n_char_embedding', type=int, default=32,
                        help='char embedding dimension')

    # - NER embedding dimension
    parser.add_argument('--n_ner_embedding', type=int, default=32,
                        help='ner embedding dimension')

    # - Character set size
    parser.add_argument('--n_char', type=int, default=128,
                        help='character set size. since ascii is used; 128 is assumed')

    # - Size of fully connected layers after projection
    parser.add_argument('--layer_size', type=str, default='512,512,512',
                        help='size of fully connected layers after projection')

    # - Mini-batch size
    parser.add_argument('--n_batch_size', type=int, default=512,
                        help='mini batch size; the last one may be smaller')

    # - Learning rate
    parser.add_argument('--learning_rate', type=float, default=0.256,
                        help='global initial learning rate')

    # - Momentum value when MomentumOptimizer is used
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value when MomentumOptimizer is used')

    # - Maximum number of iterations
    parser.add_argument('--max_iter', type=int, default=16,
                        help='maximum number of iterations')

    # - Feature choice (bitmask)
    parser.add_argument('--feature_choice', type=int, default=63,
                        help='the features used are picked with a bitmask. They are ' +
                             '1) case-insensitive bfofe with candidate word(s), ' +
                             '2) case-insensitive bfofe without candidate word(s), ' +
                             '3) case-insensitive bag-of-words, ' +
                             '4) case-sensitive bfofe with candidate word(s), ' +
                             '5) case-sensitive bfofe without candidate word(s), ' +
                             '6) case-sensitive bag-of-words, ' +
                             '7) char-level bfofe of candidate word(s), ' +
                             '8) char-level CNN, ' +
                             '9) gazetteer exact match, ' +
                             '10) bigram char-fofe.'
                             'e.g. default choice is 0b000111111, feature 1 to 6 are used')

    # - Percentage of overlap examples used during training
    parser.add_argument('--overlap_rate', type=float, default=0.36,
                        help='what percentage of overlap examples is used during training')

    # - Percentage of disjoint example used during training
    parser.add_argument('--disjoint_rate', type=float, default=0.09,
                        help='what percentage of disjoint example is used during training')

    # - Use or not use dropout
    parser.add_argument('--dropout', action='store_true', default=False,
                        help='whether to use dropout or not')

    # - Character-level forgetting factor
    parser.add_argument('--char_alpha', type=float, default=0.8,
                        help='char-level forgetting factor')

    # - Word-level forgetting factor
    parser.add_argument('--word_alpha', type=float, default=0.5,
                        help='word-level forgetting factor')

    # - Whether or not bow and context share a same word embedding
    parser.add_argument('--share_word_embedding', type=bool, default=True,
                        help='whether or not bow and context share a same word embedding')

    # - Decoding algorithm, i.e. {1: highest-score-first, 2: longest-coverage-first, 3: subsumption-removal}
    parser.add_argument('--algorithm', type=int, default=1,
                        help='decoding algorithm, i.e. {1: highest-score-first, 2: longest-coverage-first, 3: subsumption-removal}')

    # - Threshold
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='if nn output is less than threshold, it is still considered as O')

    # - Maximum length of NER candidate
    parser.add_argument('--n_window', type=int, default=7,
                        help='maximum length of NER candidate')

    # - When gazetteer is used, True if 4-bit match or False 5-bit match
    parser.add_argument('--strictly_one_hot', action='store_true', default=False,
                        help='when gazetteer is used, True if 4-bit match or False 5-bit match')

    # - dimension of z in the HOPE paper; 0 means not used
    parser.add_argument('--hope_out', type=int, default=0,
                        help='dimension of z in the HOPE paper; 0 means not used')

    # - Number of label types
    parser.add_argument('--n_label_type', type=int, default=4,
                        help='By default, PER, LOC, ORG and MISC are assumed')

    # - Kernel height
    parser.add_argument('--kernel_height', type=str, default='2,3,4,5,6,7,8,9')

    # - Kernel depth
    parser.add_argument('--kernel_depth', type=str, default=','.join(['16'] * 8))

    # - Initialize method: uniform or gaussian
    parser.add_argument('--initialize_method', type=str, default='uniform',
                        choices=['uniform', 'gaussian'])

    # - Enable distant supervision
    parser.add_argument('--enable_distant_supervision', action='store_true', default=False)

    # - Model (hopeless?)
    parser.add_argument('--model', type=str, default='hopeless')

    # - Invoke official evaluator when true
    parser.add_argument('--offical_eval', action='store_true', default=False,
                        help='invoke official evaluator when true')

    # - Where to write conll2003-{valid,test}.predicted
    parser.add_argument('--buffer_dir', type=str, default=None,
                        help='where to write conll2003-{valid,test}.predicted')
    # experimental
    # - Is second pass
    parser.add_argument('--is_2nd_pass', action='store_true', default=False,
                        help='run 2nd pass training when true')

    # - GPU fraction
    parser.add_argument('--gpu_fraction', type=float, default=0.96)

    parser.add_argument('--l1', type=float, default=0)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--n_pattern', type=int, default=0)

    # TODO
    # these hyper parameters are from kbp-ed-trainer
    # I add them here to make the interpreter happy at this point
    parser.add_argument('--language', type=str, default='eng', choices=['eng'])
    parser.add_argument('--average', action='store_true', default=False)
    parser.add_argument('--iflytek', action='store_true', default=False)

    # set a logging file at DEBUG level, TODO: windows doesn't allow ":" appear in a file name
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG,
                        filename=('log/multitask ' + time.ctime() + '.log').replace(' ', '-'),
                        filemode='w')

    # direct the INFO-level logging to the screen
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s : %(levelname)s : %(message)s'))
    logging.getLogger().addHandler(console)

    ################################################################################

    args = parser.parse_args()
    logger.info(str(args) + '\n')

    ################################################################################

    if args.is_2nd_pass:
        logger.info('user-input feature-choice was %d' % args.feature_choice)
        args.feature_choice &= 2038
        logger.info('feature-choice now is %d' % args.feature_choice)

    ################################################################################

    from multi_fofe_mention_net import *
    config = mention_config(args)
    from pprint import pprint
    logger.info("Here is config: ")
    pprint (vars(config))


    ################################################################################

    # TODO, try wikiNER
    if config.enable_distant_supervision:
        # folder = 'gigaword'
        # filelist =  [ f for f in os.listdir( folder ) \
        #                 if f.endswith('.txt') and \
        #                     os.path.getsize('gigaword/%s' % f) < 16 * 1024 * 1024 ]
        # random.shuffle( filelist )
        # logger.info( filelist )
        # logger.info( 'the smallest %d files are used' % len(filelist) )
        # config.max_iter = len(filelist)
        folder = '/eecs/research/asr/Shared/Reuters-RCV1/second-half/senna-labeled'
        filelist = os.listdir(folder)
        random.shuffle(filelist)
        logger.info(filelist)
        config.max_iter = min(len(filelist), config.max_iter)
        logger.info('There are %d machine-labeled files. %d will be used.' % \
                    (len(filelist), config.max_iter))

    ################################################################################

    mention_net = multi_fofe_mention_net(config, args.gpu_fraction)
    mention_net.tofile('./ontonotes-model/' + args.model)

    ################################################################################

    CONLL_N_LABELS = 4
    ONTONOTES_N_LABELS = 18
    KBP_N_LABELS = 10

    # there are 2 sets of vocabulary, case-insensitive and case sensitive
    nt = config.n_label_type if config.is_2nd_pass else 0

    # Vocabulary is an object that creates dicts of word to indices and fofe codes

    # QUESTION: Do they have to share the same vocabulary ? Yes, because they have
    # to share the same word embedding
    numericizer1 = vocabulary(config.word_embedding + '-case-insensitive.wordlist',
                              config.char_alpha, False,
                              n_label_type=nt)

    numericizer2 = vocabulary(config.word_embedding + '-case-sensitive.wordlist',
                              config.char_alpha, True,
                              n_label_type=nt)

    if args.feature_choice & 256 > 0: # NOT USED
        # Gazetteer is a list of names grouped by the pre-defined categories an NER
        # system is targeting at. Gazetteer is shown to be one of the most effective
        # external knowledge sources to improve NER performance
        conll2003_gazetteer = gazetteer( args.conll_datapath + '/ner-lst' )
        ontonotes_gazetteer = gazetteer(args.data_path + '/ner-lst', mode = "OntoNotes")
    else:
        conll2003_gazetteer = [set() for _ in xrange( CONLL_N_LABELS )]
        ontonotes_gazetteer = [set() for _ in xrange( ONTONOTES_N_LABELS )]

    kbp_gazetteer = gazetteer( args.kbp_gazetteer, mode = 'KBP' )

    

    # ==================================================================================
    # Official OntoNotes split
    # ==================================================================================

    ontonotes_directory = args.ontonotes_datapath
    if ontonotes_directory[-1] != '/':
        ontonotes_directory = ontonotes_directory + '/'

    ontonotes_training_path = ontonotes_directory + "train/conll"
    ontonotes_valid_path = ontonotes_directory + "development/conll"
    ontonotes_test_path = ontonotes_directory + "test-key/conll"

    # ==================================================================================
    # ----------------------------------------------------------------------------------
    # Training set
    # ----------------------------------------------------------------------------------
    # Batch constructor initializes sets of processed_sentence objects, sentence1
    # (case insensitive) and sentence2 (case sensitive)

    train_conll = batch_constructor(CoNLL2003( args.conll_datapath + '/eng.train' ), 
                               numericizer1, numericizer2, 
                               gazetteer = conll2003_gazetteer, 
                               alpha = config.word_alpha, 
                               window = config.n_window,
                               n_label_type = CONLL_N_LABELS,
                               is2ndPass = args.is_2nd_pass )

    train_ontonotes = batch_constructor(OntoNotes(ontonotes_training_path),
                              numericizer1, numericizer2,
                              gazetteer=ontonotes_gazetteer,
                              alpha=config.word_alpha,
                              window=config.n_window,
                              n_label_type = ONTONOTES_N_LABELS,
                              is2ndPass=args.is_2nd_pass)

    train_kbp = batch_constructor( 
                    # KBP(args.kbp_train_datapath, args.iflytek_checked_eng),
                    KBP(args.kbp_train_datapath),
                    numericizer1, 
                    numericizer2, 
                    gazetteer = kbp_gazetteer, 
                    alpha = config.word_alpha, 
                    window = config.n_window, 
                    n_label_type = KBP_N_LABELS,
                    language = config.language,
                    is2ndPass = args.is_2nd_pass 
                )

    logger.info('train conll: ' + str(train_conll))
    logger.info('train ontonotes: ' + str(train_ontonotes))
    logger.info('train kbp: ' + str(train_kbp))

    # ----------------------------------------------------------------------------------
    # Validation set
    # ----------------------------------------------------------------------------------

    valid_conll = batch_constructor(CoNLL2003( args.conll_datapath + '/eng.testa' ), 
                               numericizer1, numericizer2, 
                               gazetteer = conll2003_gazetteer, 
                               alpha = config.word_alpha, 
                               window = config.n_window,
                               n_label_type = CONLL_N_LABELS,
                               is2ndPass = args.is_2nd_pass)

    valid_ontonotes = batch_constructor(OntoNotes(ontonotes_valid_path),
                              numericizer1, numericizer2,
                              gazetteer=ontonotes_gazetteer,
                              alpha=config.word_alpha,
                              window=config.n_window,
                              n_label_type = ONTONOTES_N_LABELS,
                              is2ndPass=args.is_2nd_pass)

    valid_kbp = batch_constructor( 
                    KBP(args.kbp_valid_datapath), 
                    numericizer1, 
                    numericizer2, 
                    gazetteer = kbp_gazetteer, 
                    alpha = config.word_alpha, 
                    window = config.n_window, 
                    n_label_type = KBP_N_LABELS,
                    language = config.language,
                    is2ndPass = args.is_2nd_pass 
                )

    logger.info('valid conll: ' + str(valid_conll))
    logger.info('valid ontonotes: ' + str(valid_ontonotes))
    logger.info('valid kbp: ' + str(valid_kbp))

    # ----------------------------------------------------------------------------------
    # Test set
    # ----------------------------------------------------------------------------------

    test_conll  = batch_constructor( CoNLL2003( args.conll_datapath + '/eng.testb' ), 
                               numericizer1, numericizer2, 
                               gazetteer = conll2003_gazetteer, 
                               alpha = config.word_alpha, 
                               window = config.n_window,
                               n_label_type = CONLL_N_LABELS,
                               is2ndPass = args.is_2nd_pass)

    test_ontonotes = batch_constructor(OntoNotes(ontonotes_test_path),
                             numericizer1, numericizer2,
                             gazetteer=ontonotes_gazetteer,
                             alpha=config.word_alpha,
                             window=config.n_window,
                             n_label_type = ONTONOTES_N_LABELS,
                             is2ndPass=args.is_2nd_pass)

    test_kbp = batch_constructor( 
                    KBP(args.kbp_test_datapath),
                    numericizer1, 
                    numericizer2, 
                    gazetteer = kbp_gazetteer, 
                    alpha = config.word_alpha, 
                    window = config.n_window, 
                    n_label_type = KBP_N_LABELS,
                    language = config.language,
                    is2ndPass = args.is_2nd_pass 
                )

    logger.info('test conll: ' + str(test_conll))
    logger.info('test ontonotes: ' + str(test_ontonotes))
    logger.info('test kbp: ' + str(test_kbp))

    logger.info('data set loaded')

    # ==================================================================================

    ################### let's compute ####################

    prev_cost, decay_started = 2054, True if config.enable_distant_supervision else False
    best_test_fb1 = 0

    if config.enable_distant_supervision:
        machine = train
        infinite = machine.infinite_mini_batch_multi_thread(
            config.n_batch_size, True,
            config.overlap_rate, config.disjoint_rate,
            config.feature_choice, True)

    conll_task = TaskHolder(CoNLL2003, args.learning_rate,
                                         (train_conll, valid_conll, test_conll), 
                                       ("multitask-result/multitask-train-conll.predicted",
                                        'multitask-result/multitask-valid-conll.predicted',
                                        'multitask-result/multitask-test-conll.predicted'),
                                        (config.conll_datapath + '/eng.train',
                                         args.conll_datapath + '/eng.testa',
                                         args.conll_datapath + '/eng.testb'),
                                        CONLL_N_LABELS)

    ontonotes_task = TaskHolder(OntoNotes, args.learning_rate,
                                 (train_ontonotes, valid_ontonotes, test_ontonotes),
                                ("multitask-result/multitask-train-ontonotes.predicted",
                                 'multitask-result/multitask-valid-ontonotes.predicted',
                                  'multitask-result/multitask-test-ontonotes.predicted'),
                                (ontonotes_training_path, ontonotes_valid_path, ontonotes_test_path),
                                 ONTONOTES_N_LABELS)

    kbp_task = TaskHolder(KBP, args.learning_rate, (train_kbp, valid_kbp, test_kbp), 
                                ('multitask-result/multitask-train-kbp.predicted',
                                 'multitask-result/multitask-valid-kbp.predicted',
                                 'multitask-result/multitask-test-kbp.predicted'),
                                (args.kbp_train_datapath, args.kbp_valid_datapath, args.kbp_test_datapath),
                                 KBP_N_LABELS)

    for n_epoch in xrange(config.max_iter):
        if not os.path.exists('multitask-result'):
            os.makedirs('multitask-result')

        pick = random.choice([0, 1, 2])
        if pick == 0:
            # CoNLL 2003
            curr_task = conll_task
            logger.info("Epoch " + str(n_epoch) + ", random: " + str(pick))
        elif pick == 1:
            # OntoNotes
            curr_task = ontonotes_task
            logger.info("Epoch " + str(n_epoch) + ", random: " + str(pick))
        else:
            curr_task = kbp_task
            logger.info("Epoch " + str(n_epoch) + ", random: " + str(pick))

        mention_net.config.learning_rate = curr_task.lr

        # phar is used to observe training progress
        logger.info('epoch %2d, learning-rate: %f' % \
                    (n_epoch + 1, curr_task.lr))

        pbar = tqdm(total=len(curr_task.batch_constructors[0].positive) +
                          int(len(curr_task.batch_constructors[0].overlap) * config.overlap_rate) +
                          int(len(curr_task.batch_constructors[0].disjoint) * config.disjoint_rate))

        cost, cnt = 0, 0

        # example is batch of fragments from a sentence
        for example in ifilter(lambda x: x[-1].shape[0] == config.n_batch_size,
                               curr_task.batch_constructors[0].mini_batch_multi_thread(config.n_batch_size,
                                                             True,
                                                             config.overlap_rate,
                                                             config.disjoint_rate,
                                                             config.feature_choice)):

            c = mention_net.train(example, curr_task)
            cost += c * example[-1].shape[0]
            cnt += example[-1].shape[0]
            pbar.update(example[-1].shape[0])

        pbar.close()
        train_cost = cost / cnt

        # for plot
        curr_task.training_costs.append(train_cost)
        logger.info("training costs array: %s" % str(curr_task.training_costs))

        logger.info('training set iterated, %f' % train_cost)

        # just training from 1st to 9th iterations
        # if 0 < n_epoch < 10:
        #     continue

        ###############################################
        ########## go through training set ##########
        ###############################################

        train_predicted = open(curr_task.predicted_files[0], 'wb')
        to_print = []
        cost, cnt = 0, 0

        for example in curr_task.batch_constructors[0].mini_batch_multi_thread(
                512 if config.feature_choice & (1 << 9) > 0 else 1024,
                False, 1, 1, config.feature_choice):

            c, pi, pv = mention_net.eval(example, curr_task)

            cost += c * example[-1].shape[0]
            cnt += example[-1].shape[0]

            for exp, est, prob in zip(example[-1], pi, pv):
                to_print.append('%d  %d  %s' % \
                                (exp, est, '  '.join([('%f' % x) for x in prob.tolist()])))

        print >> train_predicted, '\n'.join(to_print)
        train_predicted.close()
        logger.info('training set passed for batch_num ' + str(curr_task.batch_num))

        ###############################################
        ########## go through validation set ##########
        ###############################################

        # if args.buffer_dir is None:
        #     validation_file = 'multitask-result/multitask-valid.predicted'
        # else:
        #     validation_file = os.path.join(args.buffer_dir, 'multitask-valid.predicted')

        valid_predicted = open(curr_task.predicted_files[1], 'wb')
        cost, cnt = 0, 0
        to_print = []

        for example in curr_task.batch_constructors[1].mini_batch_multi_thread(
                512 if config.feature_choice & (1 << 9) > 0 else 1024,
                False, 1, 1, config.feature_choice):

            c, pi, pv = mention_net.eval(example, curr_task)

            cost += c * example[-1].shape[0]
            cnt += example[-1].shape[0]

            for exp, est, prob in zip(example[-1], pi, pv):
                to_print.append('%d  %d  %s' % \
                                (exp, est, '  '.join([('%f' % x) for x in prob.tolist()])))

        print >> valid_predicted, '\n'.join(to_print)
        valid_predicted.close()
        valid_cost = cost / cnt
        curr_task.valid_cost = valid_cost
        logger.info('validation set passed for batch_num ' + str(curr_task.batch_num))

        #########################################
        ########## go through test set ##########
        #########################################

        # decode_test = (n_epoch >= config.max_iter / 2 or n_epoch == 0)
        decode_test = True

        # if args.buffer_dir is None:
        #     testing_file = 'multitask-result/multitask-test.predicted'
        # else:
        #     testing_file = os.path.join(args.buffer_dir, 'multitask-test.PredictionParsercted')

        test_predicted = open(curr_task.predicted_files[2], 'wb')
        cost, cnt = 0, 0
        to_print = []

        for example in curr_task.batch_constructors[2].mini_batch_multi_thread(
                512 if config.feature_choice & (1 << 9) > 0 else 1024,
                False, 1, 1, config.feature_choice):

            c, pi, pv = mention_net.eval(example, curr_task)

            cost += c * example[-1].shape[0]
            cnt += example[-1].shape[0]

            for exp, est, prob in zip(example[-1], pi, pv):
                to_print.append('%d  %d  %s' % \
                                (exp, est, '  '.join([('%f' % x) for x in prob.tolist()])))

        print >> test_predicted, '\n'.join(to_print)
        test_predicted.close()
        test_cost = cost / cnt

        curr_task.test_cost = test_cost

        logger.info('evaluation set passed for batch_num: ' + str(curr_task.batch_num))

        ###################################################################################
        ########## exhaustively iterate 3 decodding algrithms with 0.x cut-off ############
        ###################################################################################
        logger.info('cost: %f (train), %f (valid)', train_cost, valid_cost)
        # logger.info( 'cost: %f (train), %f (valid), %f (test)', train_cost, valid_cost, test_cost )

        algo_list = ['highest-first', 'longest-first', 'subsumption-removal']

        best_dev_fb1, best_threshold, best_algorithm = 0, 0.5, 1

        pp = [ p for p in PredictionParser(curr_task.generator( curr_task.data_loc[1] ), 
                                                curr_task.predicted_files[1], 
                                                config.n_window, n_label_type = curr_task.n_label ) ]

        for algorithm, name in zip([1, 2, 3], algo_list):
            for threshold in numpy.arange(0.3, 1, 0.1).tolist():
                precision, recall, f1, _ = evaluation(pp, threshold, algorithm, True, n_label_type = curr_task.n_label)
                logger.debug(('batch_num: %d, cut-off: %f, algorithm: %-20s' %
                              (curr_task.batch_num, threshold, name)) +
                             (', validation -- precision: %f,  recall: %f,  fb1: %f' % (precision, recall, f1)))
                if f1 > best_dev_fb1:
                    best_dev_fb1, best_threshold, best_algorithm = f1, threshold, algorithm
                    mention_net.config.threshold = best_threshold
                    mention_net.config.algorithm = best_algorithm

        curr_task.best_dev_fb1 = best_dev_fb1
        curr_task.best_threshold = best_threshold
        curr_task.best_algorithm = best_algorithm

        # training evaluation
        pp = [ p for p in PredictionParser(curr_task.generator( curr_task.data_loc[0] ), 
                                                curr_task.predicted_files[0], 
                                                config.n_window, n_label_type = curr_task.n_label ) ]
        
        _, _, train_fb1, info = evaluation(pp, curr_task.best_threshold, curr_task.best_algorithm, True, n_label_type = curr_task.n_label)
        logger.info('batch_num ' + str(curr_task.batch_num) + ' training:\n' + info)
        # fb1 score for validation
        curr_task.train_scores.append(train_fb1)

        logger.info("train scores array: %s" % str(curr_task.train_scores))

        # validation evaluation

        pp = [ p for p in PredictionParser(curr_task.generator( curr_task.data_loc[1] ), 
                                                curr_task.predicted_files[1], 
                                                config.n_window, n_label_type = curr_task.n_label ) ]

        _, _, test_fb1, info = evaluation(pp, curr_task.best_threshold, curr_task.best_algorithm, True, n_label_type = curr_task.n_label)
        logger.info('batch_num ' + str(curr_task.batch_num) + ', validation:\n' + info)
        curr_task.test_fb1 = test_fb1
        # fb1 score for validation
        curr_task.valid_scores.append(test_fb1)

        logger.info("valid scores array: %s" % str(curr_task.valid_scores))

        # test evaluation
            
        pp = [ p for p in PredictionParser(curr_task.generator( curr_task.data_loc[2] ), 
                                                curr_task.predicted_files[2], 
                                                config.n_window, n_label_type = curr_task.n_label ) ]

        _, _, fb1, out = evaluation(pp, curr_task.best_threshold, curr_task.best_algorithm, True, n_label_type = curr_task.n_label)
        logger.info('batch_num ' + str(curr_task.batch_num) + ', evaluation:\n' + out)
        curr_task.test_scores.append(fb1)
        curr_task.out = out

        curr_task.fb1 = fb1

        logger.info("test scores array: %s" % str(curr_task.test_scores))

        # Best so far 
        if curr_task.test_fb1 > curr_task.best_test_fb1:
            if decode_test:
                curr_task.best_test_info = curr_task.out
            curr_task.best_test_fb1 = curr_task.test_fb1
            mention_net.config.threshold = best_threshold
            mention_net.config.algorithm = best_algorithm
            mention_net.tofile('./multitask-model/' + args.model)
            
        logger.info('BEST SO FOR BATCH NUM ' + str(curr_task.batch_num) + ': threshold %f, algorithm %s\n%s' % \
                    (mention_net.config.threshold,
                     algo_list[mention_net.config.algorithm - 1],
                     curr_task.best_test_info))

        ##########################################
        ########## adjust learning rate ##########
        ##########################################

        if curr_task.valid_cost > curr_task.prev_cost or decay_started:
            curr_task.lr *= \
                0.5 ** ((4. / config.max_iter) if config.drop_rate > 0 else (1. / 2))
        else:
            curr_task.prev_cost = curr_task.valid_cost

        if config.drop_rate > 0:
            mention_net.config.drop_rate *= 0.5 ** (2. / config.max_iter)

    #===================
    #===== Plot ========
    #===================

    plt.figure(1)
    plt.plot(list(range(len(conll_task.training_costs))), conll_task.training_costs, 'g--')
    plt.title('Cost on training data')

    plt.savefig('/local/scratch/nana/mtl/fofe-ner/graphs/conll/training_costs_conll.png')

    plt.figure(2)
    plt.plot(list(range(len(conll_task.train_scores))), conll_task.train_scores, 'g--')
    plt.title('F-score on training data')

    plt.savefig('/local/scratch/nana/mtl/fofe-ner/graphs/conll/train_score_conll.png')

    plt.figure(3)
    plt.plot(list(range(len(conll_task.valid_scores))), conll_task.valid_scores, 'g--')
    plt.title('F-score on validation data')

    plt.savefig('/local/scratch/nana/mtl/fofe-ner/graphs/conll/validation_score_conll.png')

    plt.figure(4)
    plt.plot(list(range(len(conll_task.test_scores))), conll_task.test_scores, 'g--')
    plt.title('F-score on test data')

    plt.savefig('/local/scratch/nana/mtl/fofe-ner/graphs/conll/test_score_conll.png')

    plt.figure(5)
    plt.plot(list(range(len(ontonotes_task.training_costs))), ontonotes_task.training_costs, 'b--')
    plt.title('Cost on training data')

    plt.savefig('/local/scratch/nana/mtl/fofe-ner/graphs/ontonotes/training_costs_ontonotes.png')

    plt.figure(6)
    plt.plot(list(range(len(ontonotes_task.train_scores))), ontonotes_task.train_scores, 'b--')
    plt.title('F-score on training data')

    plt.savefig('/local/scratch/nana/mtl/fofe-ner/graphs/ontonotes/train_score_ontonotes.png')

    plt.figure(7)
    plt.plot(list(range(len(ontonotes_task.valid_scores))), ontonotes_task.valid_scores, 'b--')
    plt.title('F-score on validation data')

    plt.savefig('/local/scratch/nana/mtl/fofe-ner/graphs/ontonotes/validation_score_ontonotes.png')

    plt.figure(8)
    plt.plot(list(range(len(ontonotes_task.test_scores))), ontonotes_task.test_scores, 'b--')
    plt.title('F-score on test data')

    plt.savefig('/local/scratch/nana/mtl/fofe-ner/graphs/ontonotes/test_score_ontonotes.png')

    plt.figure(9)
    plt.plot(list(range(len(kbp_task.training_costs))), kbp_task.training_costs, 'r--')
    plt.title('Cost on training data')

    plt.savefig('/local/scratch/nana/mtl/fofe-ner/graphs/kbp/training_costs_kbp.png')

    plt.figure(10)
    plt.plot(list(range(len(kbp_task.train_scores))), kbp_task.train_scores, 'r--')
    plt.title('F-score on training data')

    plt.savefig('/local/scratch/nana/mtl/fofe-ner/graphs/kbp/train_score_kbp.png')

    plt.figure(11)
    plt.plot(list(range(len(kbp_task.valid_scores))), kbp_task.valid_scores, 'r--')
    plt.title('F-score on validation data')

    plt.savefig('/local/scratch/nana/mtl/fofe-ner/graphs/kbp/validation_score_kbp.png')

    plt.figure(12)
    plt.plot(list(range(len(kbp_task.test_scores))), kbp_task.test_scores, 'r--')
    plt.title('F-score on test data')

    plt.savefig('/local/scratch/nana/mtl/fofe-ner/graphs/kbp/test_score_kbp.png')


    #===================

    logger.info('results are written in multitask-{valid,test}.predicted')

