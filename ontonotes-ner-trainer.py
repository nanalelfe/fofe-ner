#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy, logging, argparse, time, copy, os, cPickle, sys
from subprocess import Popen, PIPE, call
from Queue import Queue
from threading import Thread
from random import shuffle
from math import floor

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
    parser.add_argument('word_embedding', type=str,
                        help='word_embedding.{-case-insensitive, -case-sensitive}.word2vec are assumed')

    parser.add_argument('data_path', type=str,
                        help='path to eng.{train, testa, testb} of CoNLL2003')

    # - Path to directory where the OntoNotes project is located
    parser.add_argument('dir_path', type=str,
                        help='path to the preparsed OntoNotes dataset')

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
    parser.add_argument('--n_label_type', type=int, default=18,
                        help='By default, OntoNotes')

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
                        filename=('log/ontonotes ' + time.ctime() + '.log').replace(' ', '-'),
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

    from fofe_mention_net import *
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

    mention_net = fofe_mention_net(config, args.gpu_fraction)
    mention_net.tofile('./ontonotes-model/' + args.model)

    ################################################################################

    # there are 2 sets of vocabulary, case-insensitive and case sensitive
    nt = config.n_label_type if config.is_2nd_pass else 0

    # Vocabulary is an object that creates dicts of word to indices and fofe codes
    numericizer1 = vocabulary(config.word_embedding + '-case-insensitive.wordlist',
                              config.char_alpha, False,
                              n_label_type=nt)
    numericizer2 = vocabulary(config.word_embedding + '-case-sensitive.wordlist',
                              config.char_alpha, True,
                              n_label_type=nt)

    ontonotes_gazetteer = [set() for _ in xrange(args.n_label_type)]

    # ==================================================================================
    # Official OntoNotes split
    # ==================================================================================

    # Data splitting
    training_rate = 0.8
    validation_rate = 0.1
    test_rate = 0.1

    directory = args.dir_path
    if directory[-1] != '/':
        directory = directory + '/'

    training_path = directory + "train/conll"
    valid_path = directory + "development/conll"
    test_path = directory + "test/conll"

    # ==================================================================================
    # ----------------------------------------------------------------------------------
    # Training set
    # ----------------------------------------------------------------------------------
    # Batch constructor initializes sets of processed_sentence objects, sentence1
    # (case insensitive) and sentence2 (case sensitive)
    train = batch_constructor(OntoNotes(training_path),
                              numericizer1, numericizer2,
                              gazetteer=ontonotes_gazetteer,
                              alpha=config.word_alpha,
                              window=config.n_window,
                              n_label_type = config.n_label_type,
                              is2ndPass=args.is_2nd_pass)
    logger.info('train: ' + str(train))

    # ----------------------------------------------------------------------------------
    # Validation set
    # ----------------------------------------------------------------------------------
    valid = batch_constructor(OntoNotes(valid_path),
                              numericizer1, numericizer2,
                              gazetteer=ontonotes_gazetteer,
                              alpha=config.word_alpha,
                              window=config.n_window,
                              n_label_type = config.n_label_type,
                              is2ndPass=args.is_2nd_pass)
    logger.info('valid: ' + str(valid))

    # ----------------------------------------------------------------------------------
    # Test set
    # ----------------------------------------------------------------------------------

    test = batch_constructor(OntoNotes(test_path),
                             numericizer1, numericizer2,
                             gazetteer=ontonotes_gazetteer,
                             alpha=config.word_alpha,
                             window=config.n_window,
                             n_label_type = config.n_label_type,
                             is2ndPass=args.is_2nd_pass)
    logger.info('test: ' + str(test))

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

    #===================
    # ==== Plot =======
    #===================

    # F1 scores
    train_scores = []
    valid_scores = []
    test_scores = []

    # Train cost
    training_costs = []

    for n_epoch in xrange(config.max_iter):
        if not os.path.exists('ontonotes-result'):
            os.makedirs('ontonotes-result')

        #############################################
        ########## go through training set ##########
        #############################################

        # phar is used to observe training progress
        logger.info('epoch %2d, learning-rate: %f' % \
                    (n_epoch + 1, mention_net.config.learning_rate))

        if config.enable_distant_supervision:
            logger.info("Distant supervision enabled")
            # train = batch_constructor(  # gigaword( 'gigaword/' + filelist[n_epoch] ),
            #     CoNLL2003(os.path.join(folder, filelist[n_epoch])),
            #     numericizer1, numericizer2,
            #     gazetteer=conll2003_gazetteer,
            #     alpha=config.word_alpha,
            #     window=config.n_window,
            #     is2ndPass=args.is_2nd_pass)
            # logger.info('train: ' + str(train))

        pbar = tqdm(total=len(train.positive) +
                          int(len(train.overlap) * config.overlap_rate) +
                          int(len(train.disjoint) * config.disjoint_rate))

        cost, cnt = 0, 0

        # example is batch of fragments from a sentence
        for example in ifilter(lambda x: x[-1].shape[0] == config.n_batch_size,
                               train.mini_batch_multi_thread(config.n_batch_size,
                                                             True,
                                                             config.overlap_rate,
                                                             config.disjoint_rate,
                                                             config.feature_choice)):

            c = mention_net.train(example)

            cost += c * example[-1].shape[0]
            cnt += example[-1].shape[0]
            pbar.update(example[-1].shape[0])

            if config.enable_distant_supervision:
                mention_net.train(infinite.next())

        pbar.close()
        train_cost = cost / cnt

        # for plot
        training_costs.append(train_cost)
        logger.info("training costs array: %s" % str(training_costs))

        logger.info('training set iterated, %f' % train_cost)

        # just training from 1st to 9th iterations
        # if 0 < n_epoch < 10:
        #     continue

        ###############################################
        ########## go through training set ##########
        ###############################################

        training_file = "ontonotes-result/ontonotes-train.predicted"
        train_predicted = open(training_file, 'wb')
        to_print = []

        cost, cnt = 0, 0
        to_print = []

        for example in train.mini_batch_multi_thread(
                512 if config.feature_choice & (1 << 9) > 0 else 1024,
                False, 1, 1, config.feature_choice):

            c, pi, pv = mention_net.eval(example)

            cost += c * example[-1].shape[0]
            cnt += example[-1].shape[0]

            for exp, est, prob in zip(example[-1], pi, pv):
                to_print.append('%d  %d  %s' % \
                                (exp, est, '  '.join([('%f' % x) for x in prob.tolist()])))

        print >> train_predicted, '\n'.join(to_print)
        train_predicted.close()
        valid_cost = cost / cnt
        logger.info('training set passed')

        ###############################################
        ########## go through validation set ##########
        ###############################################

        if args.buffer_dir is None:
            validation_file = 'ontonotes-result/ontonotes-valid.predicted'
        else:
            validation_file = os.path.join(args.buffer_dir, 'ontonotes-valid.predicted')
        valid_predicted = open(validation_file, 'wb')
        cost, cnt = 0, 0
        to_print = []

        for example in valid.mini_batch_multi_thread(
                512 if config.feature_choice & (1 << 9) > 0 else 1024,
                False, 1, 1, config.feature_choice):

            c, pi, pv = mention_net.eval(example)

            cost += c * example[-1].shape[0]
            cnt += example[-1].shape[0]

            for exp, est, prob in zip(example[-1], pi, pv):
                to_print.append('%d  %d  %s' % \
                                (exp, est, '  '.join([('%f' % x) for x in prob.tolist()])))

        print >> valid_predicted, '\n'.join(to_print)
        valid_predicted.close()
        valid_cost = cost / cnt
        logger.info('validation set passed')

        #########################################
        ########## go through test set ##########
        #########################################

        decode_test = True#(n_epoch >= config.max_iter / 2 or n_epoch == 0)

        if args.offical_eval or decode_test:
            if args.buffer_dir is None:
                testing_file = 'ontonotes-result/ontonotes-test.predicted'
            else:
                testing_file = os.path.join(args.buffer_dir, 'ontonotes-test.predicted')

            test_predicted = open(testing_file, 'wb')
            cost, cnt = 0, 0
            to_print = []

            for example in test.mini_batch_multi_thread(
                    512 if config.feature_choice & (1 << 9) > 0 else 1024,
                    False, 1, 1, config.feature_choice):

                c, pi, pv = mention_net.eval(example)

                cost += c * example[-1].shape[0]
                cnt += example[-1].shape[0]

                for exp, est, prob in zip(example[-1], pi, pv):
                    to_print.append('%d  %d  %s' % \
                                    (exp, est, '  '.join([('%f' % x) for x in prob.tolist()])))

            print >> test_predicted, '\n'.join(to_print)
            test_predicted.close()
            test_cost = cost / cnt
            logger.info('evaluation set passed')

        ###################################################################################
        ########## exhaustively iterate 3 decodding algrithms with 0.x cut-off ############
        ###################################################################################
        logger.info('cost: %f (train), %f (valid)', train_cost, valid_cost)
        # logger.info( 'cost: %f (train), %f (valid), %f (test)', train_cost, valid_cost, test_cost )

        algo_list = ['highest-first', 'longest-first', 'subsumption-removal']
        best_dev_fb1, best_threshold, best_algorithm = 0, 0.5, 1

        if decode_test:

            pp = [p for p in PredictionParser(OntoNotes(valid_path),
                                              validation_file,
                                              config.n_window, n_label_type = config.n_label_type)]

            for algorithm, name in zip([1, 2, 3], algo_list):
                for threshold in numpy.arange(0.3, 1, 0.1).tolist():
                    precision, recall, f1, _ = evaluation(pp, threshold, algorithm, True, n_label_type = config.n_label_type)
                    logger.debug(('cut-off: %f, algorithm: %-20s' %
                                  (threshold, name)) +
                                 (', validation -- precision: %f,  recall: %f,  fb1: %f' % (precision, recall, f1)))
                    if f1 > best_dev_fb1:
                        best_dev_fb1, best_threshold, best_algorithm = f1, threshold, algorithm
                        mention_net.config.threshold = best_threshold
                        mention_net.config.algorithm = best_algorithm

        ###############################################
        ########## invoke official evaluator ##########
        ###############################################

        if args.offical_eval:
            logger.info("Inside the official eval if-statement")
            # cmd = ('CoNLL2003eval.py --threshold=%f --algorithm=%d --n_window=%d --config=%s ' \
            #                 % ( best_threshold, best_algorithm, config.n_window,
            #                     'conll2003-model/%s.config' % args.model ) ) + \
            #       ('%s/eng.testa %s | conlleval' % (config.data_path, validation_file) )
            # process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
            # (out, err) = process.communicate()
            # exit_code = process.wait()
            # logger.info( 'validation\n' + out )

            # cmd = ('CoNLL2003eval.py --threshold=%f --algorithm=%d --n_window=%d ' \
            #                 % ( best_threshold, best_algorithm, config.n_window ) ) + \
            #       ('%s/eng.testb %s | conlleval' % (config.data_path, testing_file) )
            # process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
            # (out, err) = process.communicate()
            # logger.info( 'test, global threshold\n' + out )
            # test_fb1 = float(out.split('\n')[1].split()[-1])

        else:
            # training
            pp = [p for p in TrainingPredictionParser(OntoNotes(training_path), training_file, config.n_window, n_label_type = config.n_label_type)]

            _, _, train_fb1, info = evaluation(pp, best_threshold, best_algorithm, True, n_label_type = config.n_label_type)
            logger.info('training:\n' + info)
            # fb1 score for validation
            train_scores.append(train_fb1)

            logger.info("train scores array: %s" % str(train_scores))

            # validation
            pp = [p for p in PredictionParser(OntoNotes(valid_path),
                                              validation_file,
                                              config.n_window, n_label_type = config.n_label_type)]
            _, _, test_fb1, info = evaluation(pp, best_threshold, best_algorithm, True, n_label_type = config.n_label_type)
            logger.info('validation:\n' + info)
            # fb1 score for validation
            valid_scores.append(test_fb1)

            logger.info("valid scores array: %s" % str(valid_scores))

            if decode_test:
                pp = [p for p in PredictionParser(OntoNotes(test_path),
                                                  testing_file,
                                                  config.n_window, n_label_type = config.n_label_type)]
                _, _, fb1, out = evaluation(pp, best_threshold, best_algorithm, True, n_label_type = config.n_label_type)
                logger.info('evaluation:\n' + out)
                test_scores.append(fb1)

                logger.info("test scores array: %s" % str(test_scores))

        if test_fb1 > best_test_fb1:
            if decode_test:
                best_test_info = out
            best_test_fb1 = test_fb1
            mention_net.config.threshold = best_threshold
            mention_net.config.algorithm = best_algorithm
            mention_net.tofile('./ontonotes-model/' + args.model)

        # cmd = ('CoNLL2003eval.py --threshold=%f --algorithm=%d --n_window=%d --config=%s ' \
        #                 % ( best_threshold, best_algorithm, config.n_window,
        #                     'conll2003-model/%s.config' % args.model ) ) + \
        #       ('%s/eng.testb conll2003-result/conll2003-test.predicted | conlleval' \
        #                 % config.data_path)
        # process = Popen( cmd, shell = True, stdout = PIPE, stderr = PIPE)
        # (out, err) = process.communicate()
        # logger.info( 'test, individual thresholds\n' + out )

        if decode_test:
            logger.info('BEST SO FOR: threshold %f, algorithm %s\n%s' % \
                        (mention_net.config.threshold,
                         algo_list[mention_net.config.algorithm - 1],
                         best_test_info))

        ##########################################
        ########## adjust learning rate ##########
        ##########################################

        if valid_cost > prev_cost or decay_started:
            mention_net.config.learning_rate *= \
                0.5 ** ((4. / config.max_iter) if config.drop_rate > 0 else (1. / 2))
        else:
            prev_cost = valid_cost

        if config.drop_rate > 0:
            mention_net.config.drop_rate *= 0.5 ** (2. / config.max_iter)


    #===================
    #===== Plot ========
    #===================

    epochs = list(range(1, n_epoch + 1))

    plt.figure(1)
    plt.plot(epochs, training_costs, 'r--')
    plt.title('Cost on training data')

    plt.savefig('/local/scratch/nana/fofe-ner/training_costs.png')

    plt.figure(2)
    plt.plot(epochs, valid_scores, 'r--')
    plt.title('F-score on validation data')

    plt.savefig('/local/scratch/nana/fofe-ner/validation_score.png')

    plt.figure(3)
    plt.plot(epochs, test_scores, 'r--')
    plt.title('F-score on test data')

    plt.savefig('/local/scratch/nana/fofe-ner/test_score.png')
    #===================

    logger.info('results are written in ontonotes-{valid,test}.predicted')
