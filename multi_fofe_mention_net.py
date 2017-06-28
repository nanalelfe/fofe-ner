#!/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

import numpy, logging, time, copy, os, cPickle

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# from CoNLL2003eval import evaluation
from gigaword2feature import * 
from LinkingUtil import *

from tqdm import tqdm
from itertools import ifilter, izip, imap
from random import choice

logger = logging.getLogger( __name__ )

########################################################################

def load_word_embedding( filename ):
    """
    Parameters
    ----------
        filename : str
            path to the word embedding binary file (trained by skipgram-train.py)

    Returns
    -------
        embedding : ndarray
            2D matrix where each row is a word vector
    """
    with open( filename, 'rb' ) as fp:
        shape = numpy.fromfile( fp, dtype = numpy.int32, count = 2 )
        embedding = numpy.fromfile( fp, dtype = numpy.float32 ).reshape( shape )
    return embedding

########################################################################


class mention_config( object ):
    def __init__( self, args = None ):
        # default config
        self.word_embedding = 'word2vec/reuters128'
        self.data_path = 'processed-data'
        self.n_char_embedding = 64
        self.n_char = 128
        self.n_batch_size = 512
        self.learning_rate = 0.1024
        self.momentum = 0.9
        self.layer_size = '512,512,512'
        self.max_iter = 64
        self.feature_choice = 511
        self.overlap_rate = 0.08
        self.disjoint_rate = 0.016
        self.dropout = True
        self.n_ner_embedding = 32
        self.char_alpha = 0.8
        self.word_alpha = 0.5
        self.n_window = 7
        self.strictly_one_hot = True
        self.hope_out = 0
        self.n_label_type = 7
        self.kernel_height = range(2, 10)
        self.kernel_depth = [16] * 8
        self.enable_distant_supervision = False
        self.initialize_method = 'uniform'
        self.kernel_depth = ','.join( ['16'] * 8 )
        self.kernel_height = '2,3,4,5,6,7,8,9'
        self.l1 = 0
        self.l2 = 0
        self.n_pattern = 0

        # KBP-specific config
        self.language = 'eng'
        self.average = False
        self.is_2nd_pass = False

        if args is not None:
            self.__dict__.update( args.__dict__ )

        self.kernel_depth = [ int(d) for d in self.kernel_depth.split(',') ]
        self.kernel_height = [ int(h) for h in self.kernel_height.split(',') ]
        
        # these parameters are not decided by the input to the program
        # I put some placeholders here; they will be eventually modified
        self.algorithm = 1              # highest first, decided by training
        self.threshold = 0.5            # decided by training
        self.drop_rate = 0.4096 if self.dropout else 0
        self.n_word1 = 100000           # decided by self.word_embedding
        self.n_word2 = 100000           # decided by self.word_embedding
        self.n_word_embedding1 = 256    # decided by self.word_embedding
        self.n_word_embedding2 = 256    # decided by self.word_embedding
        self.customized_threshold = None    # not used any more
        assert len( self.kernel_height ) == len( self.kernel_depth )


########################################################################


class multi_fofe_mention_net( object ):

    def __init__( self, config = None, gpu_option = 0.96 ):
        """
        Parameters
        ----------
            config : mention_config
        """

        # most code is lengacy, let's put some alias here
        word_embedding = config.word_embedding
        data_path = config.data_path
        n_char_embedding = config.n_char_embedding
        n_char = config.n_char
        n_batch_size = config.n_batch_size
        learning_rate = config.learning_rate
        momentum = config.momentum
        layer_size = config.layer_size
        feature_choice = config.feature_choice
        overlap_rate = config.overlap_rate
        disjoint_rate = config.disjoint_rate
        dropout = config.dropout
        n_ner_embedding = config.n_ner_embedding
        char_alpha = config.char_alpha
        word_alpha = config.word_alpha
        n_window = config.n_window
        hope_out = config.hope_out
        n_label_type = config.n_label_type
        kernel_height = config.kernel_height
        kernel_depth = config.kernel_depth
        enable_distant_supervision = config.enable_distant_supervision
        initialize_method = config.initialize_method
        n_pattern = config.n_pattern
 
        # if config is not None:
        #     self.config = copy.deepcopy( config )
        # else:
        #     self.config = mention_config()

        self.config = mention_config()
        if config is not None:
            self.config.__dict__.update( config.__dict__ )

        self.graph = tf.Graph()

        # TODO: create a graph instead of using default graph
        #       otherwise, we cannot instantiate multiple fofe_mention_nets
        # tf.reset_default_graph()

        if gpu_option is not None:
            gpu_option = tf.GPUOptions( per_process_gpu_memory_fraction = gpu_option )
            configg = tf.ConfigProto(gpu_options = gpu_option )
            configg.gpu_options.allow_growth = True
            self.session = tf.Session( config = configg,
                                                # log_device_placement = True ),
                                       graph = self.graph )

        else:
             self.session = tf.Session( graph = self.graph )

        # NON-CHINESE WORD EMBEDDINGS
        if os.path.exists( self.config.word_embedding + '-case-insensitive.word2vec' ) \
            and os.path.exists( self.config.word_embedding + '-case-sensitive.word2vec' ):

            # Matrix with row word vectors for insensitive case
            projection1 = load_word_embedding( self.config.word_embedding + \
                                               '-case-insensitive.word2vec' )

            # Matrix with row word vectors for sensitive case
            projection2 = load_word_embedding( self.config.word_embedding + \
                                               '-case-sensitive.word2vec' )

            # Number of words in insensitive case
            self.n_word1 = projection1.shape[0]

            # Number of words in sensitive case
            self.n_word2 = projection2.shape[0]

            # Number of words in insensitive case
            n_word1 = projection1.shape[0]
            # Number of words in sensitive case
            n_word2 = projection2.shape[0]

            # Length of the word embedding for insensitive case
            n_word_embedding1 = projection1.shape[1]

            # Length of the word embedding for sensitive case
            n_word_embedding2 = projection2.shape[1]

            # Update the configuation
            self.config.n_word1 = self.n_word1
            self.config.n_word2 = self.n_word2
            self.config.n_word_embedding1 = n_word_embedding1
            self.config.n_word_embedding2 = n_word_embedding2
            logger.info( 'non-Chinese embeddings loaded' )

        # CHINESE CHARACTER EMBEDDINGS
        elif os.path.exists( self.config.word_embedding + '-char.word2vec' ) \
            and os.path.exists( self.config.word_embedding + '-word.word2vec' ):

            projection1 = load_word_embedding( self.config.word_embedding + \
                                               '-char.word2vec' )
            projection2 = load_word_embedding( self.config.word_embedding + \
                            ('-avg.word2vec' if self.config.average else '-word.word2vec') )

            self.n_word1 = projection1.shape[0]
            self.n_word2 = projection2.shape[0]

            n_word_embedding1 = projection1.shape[1]
            n_word_embedding2 = projection2.shape[1]

            self.config.n_word1 = self.n_word1
            self.config.n_word2 = self.n_word2
            self.config.n_word_embedding1 = n_word_embedding1
            self.config.n_word_embedding2 = n_word_embedding2
            logger.info( 'Chinese embeddings loaded' )

        else:
            self.n_word1 = self.config.n_word1
            self.n_word2 = self.config.n_word2
            n_word_embedding1 = self.config.n_word_embedding1
            n_word_embedding2 = self.config.n_word_embedding2

            projection1 = numpy.random.uniform( -1, 1, 
                            (self.n_word1, n_word_embedding1) ).astype( numpy.float32 )
            projection2 = numpy.random.uniform( -1, 1, 
                            (self.n_word2, n_word_embedding2) ).astype( numpy.float32 )
            logger.info( 'embedding is randomly initialized' )

        if config.is_2nd_pass:
            logger.info( 'In 2nd pass, substitute the last few entries with label types.' )
            projection1[-1 - n_label_type: -1, :] = \
                            numpy.random.uniform( 
                                    projection1.min(), projection1.max(),
                                    (n_label_type, n_word_embedding1) 
                            ).astype( numpy.float32 )
            sub = numpy.random.uniform( 
                                projection2.min(), projection2.max(),
                                (n_label_type, n_word_embedding2) 
                        ).astype( numpy.float32 )
            if self.config.language == 'cmn':
                projection2[-1 - n_label_type: -1, :] = sub
            else:
                projection2[-2 - n_label_type: -2, :] = sub

        # dimension of x in the HOPE paper
        hope_in = 0
        for ith, name in enumerate( ['case-insensitive bidirectional-context-with-focus', \
                                     'case-insensitive bidirectional-context-without-focus', \
                                     'case-insensitive focus-bow', \
                                     'case-sensitive bidirectional-context-with-focus', \
                                     'case-sensitive bidirectional-context-without-focus', \
                                     'case-sensitive focus-bow', \
                                     'left-char & right-char', 'left-initial & right-initial', \
                                     'gazetteer', 'char-conv', 'char-bigram' ] ):
            if (1 << ith) & self.config.feature_choice > 0: 
                logger.info( '%s used' % name )
                if ith in [0, 1]:
                    hope_in += n_word_embedding1 * 2
                elif ith in [3, 4]:
                    hope_in += n_word_embedding2 * 2
                elif ith == 2:
                    hope_in += n_word_embedding1
                elif ith == 5:
                    hope_in += n_word_embedding1
                elif ith in [6, 7]:
                    hope_in += n_char_embedding * 2
                elif ith == 8: 
                    hope_in += n_ner_embedding
                elif ith == 9:
                    hope_in += sum( kernel_depth )
                elif ith == 10:
                    hope_in += n_char_embedding * 2

        CONLL_N_LABELS = 4
        ONTONOTES_N_LABELS = 18

        # add a U matrix between projected feature and fully-connected layers

        n_in_shared = [ hope_out if hope_out > 0 else hope_in ] + [ int(s) for s in layer_size.split(',') ]
        n_out_shared = n_in_shared[1:] + [n_in_shared[-1]]

        n_in_conll = n_out_shared
        n_out_conll = n_in_conll[1:] + [ CONLL_N_LABELS + 1 ]

        n_in_ontonotes = n_out_shared
        n_out_ontonotes = n_in_ontonotes[1:] + [ ONTONOTES_N_LABELS + 1 ]

        logger.info( 'n_in_shared: ' + str(n_in_shared) )
        logger.info( 'n_out_shared: ' + str(n_out_shared) )
        logger.info( 'n_in_ontonotes: ' + str(n_in_ontonotes) )
        logger.info( 'n_out_ontonotes: ' + str(n_out_ontonotes) )
        logger.info( 'n_in_conll: ' + str(n_in_conll) )
        logger.info( 'n_out_conll: ' + str(n_out_conll) )

        with self.graph.as_default():

            ###########################
            ### WORD-LEVEL FEATURES ###
            ###########################

            # CASE INSENSITIVE
            # ----------------

            # case insensitive excluding fragment
            self.lw1_values = tf.placeholder( dtype = tf.float32, shape = [None], 
                                              name = 'left-context-values' )
            self.lw1_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'left-context-indices' )

            # case insensitive excluding fragment
            self.rw1_values = tf.placeholder( tf.float32, [None], 
                                              name = 'right-context-values' )
            self.rw1_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'right-context-indices' )

            # case insensitive including fragment
            self.lw2_values = tf.placeholder( tf.float32, [None], 
                                              name = 'left-context-values' )
            self.lw2_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'left-context-indices' )

            # case insensitive including fragment
            self.rw2_values = tf.placeholder( tf.float32, [None], 
                                              name = 'right-context-values' )
            self.rw2_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'right-context-indices' )

            # case insensitive bow fragment
            self.bow1_values = tf.placeholder( tf.float32, [None], 
                                               name = 'bow-values' )
            self.bow1_indices = tf.placeholder( tf.int64, [None, 2], 
                                                name = 'bow-indices' )

            # CASE SENSITIVE
            # --------------
            # value vectors in FOFE code

            # case sensitive excluding fragment
            self.lw3_values = tf.placeholder( tf.float32, [None], 
                                              name = 'left-context-values' )
            self.lw3_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'left-context-indices' )

            # case sensitive excluding fragment
            self.rw3_values = tf.placeholder( tf.float32, [None], 
                                              name = 'right-context-values' )
            self.rw3_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'right-context-indices' )

            # case sensitive including fragment
            self.lw4_values = tf.placeholder( tf.float32, [None], 
                                              name = 'left-context-values' )
            self.lw4_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'left-context-indices' )

            # case sensitive including fragment
            self.rw4_values = tf.placeholder( tf.float32, [None], 
                                              name = 'right-context-values' )
            self.rw4_indices = tf.placeholder( tf.int64, [None, 2], 
                                               name = 'right-context-indices' )

            # case sensitive bow fragment
            self.bow2_values = tf.placeholder( tf.float32, [None], 
                                               name = 'bow-values' )
            self.bow2_indices = tf.placeholder( tf.int64, [None, 2], 
                                                name = 'bow-indices' )

            # Shape of the index matrices
            # ===========================
            # case insensitive bow shape, vector of size 2
            # first value # of rows and second value # of cols
            self.shape1 = tf.placeholder( tf.int64, [2], name = 'bow-shape1' )
            # case sensitive
            self.shape2 = tf.placeholder( tf.int64, [2], name = 'bow-shape2' )


            ################################
            ### CHARACTER-LEVEL FEATURES ###
            ################################

            self.lc_fofe = tf.placeholder( tf.float32, [None, n_char], name = 'left-char' )
            self.rc_fofe = tf.placeholder( tf.float32, [None, n_char], name = 'right-char' )

            self.li_fofe = tf.placeholder( tf.float32, [None, n_char], name = 'left-initial' )
            self.ri_fofe = tf.placeholder( tf.float32, [None, n_char], name = 'right-initial' )

            ################################################################################
            # A grid like matrix where each row represents a token, and a 0/1 in a particular
            # column means that the token is a mention with entity type corresponding to 
            # column number:
            self.ner_cls_match = tf.placeholder( tf.float32, [None, n_label_type + 1], name = 'gazetteer' )

            # Each entity type is associated with a label
            self.label = tf.placeholder( tf.int64, [None], 'label' )

            # A constant value
            self.lr = tf.placeholder( tf.float32, [], 'learning-rate' )

            # 1 - dropout rate
            self.keep_prob = tf.placeholder( tf.float32, [], 'keep-prob' )
            
            self.char_idx = tf.placeholder( tf.int32, [None, None], name = 'char-idx' )

            # Bigrams
            # left context character level
            self.lbc_values = tf.placeholder( tf.float32, [None], name = 'bigram-values' )
            self.lbc_indices = tf.placeholder( tf.int64, [None, 2], name = 'bigram-indices' )

            # right context character level
            self.rbc_values = tf.placeholder( tf.float32, [None], name = 'bigram-values' )
            self.rbc_indices = tf.placeholder( tf.int64, [None, 2], name = 'bigram-indices' )

            ################################################################################

            self.shape3 = tf.placeholder( tf.int64, [2], name = 'shape3' )

            logger.info( 'placeholder defined' )
            #################### model parameters ##########################################
            ################################################################################

            # projection1 and projection2 are one-hot word vectors (?)
            self.word_embedding_1 = tf.Variable( projection1 )
            self.word_embedding_2 = tf.Variable( projection2 )
            del projection1, projection2

            # weights & bias of fully-connected layers
            # self.W contains weight matrices for each layer
            self.shared_layer_weights = []
            self.ontonotes_layer_weights = []
            self.conll_layer_weights = []

            # Bias
            self.shared_layer_b = []   
            self.ontonotes_layer_b = []
            self.conll_layer_b = []
            self.param = []

            # network weights are randomly initialized based on the uniform distribution
            if initialize_method == 'uniform':
                # Character-level network weights initialization
                val_rng = numpy.float32(2.5 / numpy.sqrt(n_char + n_char_embedding))
                self.char_embedding = tf.Variable( tf.random_uniform( 
                                        [n_char, n_char_embedding], minval = -val_rng, maxval = val_rng ) )
            
                self.conv_embedding = tf.Variable( tf.random_uniform( 
                                        [n_char, n_char_embedding], minval = -val_rng, maxval = val_rng ) )

                # Word-level network weights initialization
                # value range
                val_rng = numpy.float32(2.5 / numpy.sqrt(n_label_type + n_ner_embedding + 1))

                # random initialization of word embeddings
                self.ner_embedding = tf.Variable( tf.random_uniform( 
                                        [n_label_type + 1 ,n_ner_embedding], minval = -val_rng, maxval = val_rng ) )
                
                val_rng = numpy.float32(2.5 / numpy.sqrt(96 * 96 + n_char_embedding))
                self.bigram_embedding = tf.Variable( tf.random_uniform( 
                                        [96 * 96, n_char_embedding], minval = -val_rng, maxval = val_rng ) )

                self.kernels = [ tf.Variable( tf.random_uniform( 
                                    [h, n_char_embedding, 1, d], 
                                    minval = -2.5 / numpy.sqrt(1 + h * n_char_embedding * d), 
                                    maxval = 2.5 / numpy.sqrt(1 + h * n_char_embedding * d) ) ) for \
                                (h, d) in zip( kernel_height, kernel_depth ) ]

                self.kernel_bias = [ tf.Variable( tf.zeros( [d] ) ) for d in kernel_depth ]

                if hope_out > 0:
                    val_rng = 2.5 / numpy.sqrt( hope_in + hope_out )
                    u_matrix = numpy.random.uniform( -val_rng, val_rng, 
                                                    [hope_in, hope_out] ).astype( numpy.float32 )
                    u_matrix = u_matrix / (u_matrix ** 2).sum( 0 )
                    self.U = tf.Variable( u_matrix )
                    del u_matrix

                # Initialize the weights of each module using uniform
                for i, o in zip( n_in_shared, n_out_shared ):
                    logger.info("shared shape: " + str(i) + ' ' + str(o) + '\n')
                    val_rng = numpy.float32(2.5 / numpy.sqrt(i + o))
                    # random_uniform : Returns a tensor of the specified shape filled with random uniform values.
                    test = tf.random_uniform( [i, o], minval = -val_rng, maxval = val_rng )
                    logger.info("shared shape 2 : " + str(test.get_shape()))
                    self.shared_layer_weights.append( tf.Variable( tf.random_uniform( [i, o], minval = -val_rng, maxval = val_rng ) ) )
                    self.shared_layer_b.append( tf.Variable( tf.zeros( [o] ) )  )

                for i, o in zip( n_in_ontonotes, n_out_ontonotes ):
                    logger.info("ontonotes shape: " + str(i) + ' ' + str(o) + '\n')
                    val_rng = numpy.float32(2.5 / numpy.sqrt(i + o))

                    self.ontonotes_layer_weights.append( tf.Variable( tf.random_uniform( [i, o], minval = -val_rng, maxval = val_rng ) ) )
                    self.ontonotes_layer_b.append( tf.Variable( tf.zeros( [o] ) )  )

                for i, o in zip( n_in_conll, n_out_conll ):
                    logger.info("conll shape: " + str(i) + ' ' + str(o) + '\n')
                    val_rng = numpy.float32(2.5 / numpy.sqrt(i + o))

                    self.conll_layer_weights.append( tf.Variable( tf.random_uniform( [i, o], minval = -val_rng, maxval = val_rng ) ) )
                    self.conll_layer_b.append( tf.Variable( tf.zeros( [o] ) )  )


                if n_pattern > 0:
                    # case-insensitive patterns
                    val_rng = numpy.sqrt(3)

                    # val_rng = numpy.float32(2.5 / numpy.sqrt(n_pattern + n_word1)) 
                    self.pattern1 = []
                    self.pattern1_bias = []
                    for _ in xrange(4):
                        self.pattern1.append(
                            tf.Variable(
                                tf.random_uniform( [n_pattern, n_word1],
                                                    minval = -val_rng, maxval = val_rng )
                            )
                        )
                        self.pattern1_bias.append(
                            tf.Variable(
                                tf.random_uniform( [n_pattern],
                                                    minval = -val_rng, maxval = val_rng )
                            )
                        )

                    # case-sensitive patterns
                    # val_rng = numpy.float32(2.5 / numpy.sqrt(n_pattern + n_word2))
                    self.pattern2 = []
                    self.pattern2_bias = []
                    for _ in xrange(4):
                        self.pattern2.append(
                            tf.Variable(
                                tf.random_uniform( [n_pattern, n_word2],
                                                    minval = -val_rng, maxval = val_rng )
                            )
                        )
                        self.pattern2_bias.append(
                            tf.Variable(
                                tf.random_uniform( [n_pattern],
                                                    minval = -val_rng, maxval = val_rng )
                            )
                        )

                del val_rng
                
            else:
                self.char_embedding = tf.Variable( tf.truncated_normal( [n_char, n_char_embedding], 
                                                stddev = numpy.sqrt(2./(n_char * n_char_embedding)) ) )
                
                self.conv_embedding = tf.Variable( tf.truncated_normal( [n_char, n_char_embedding], 
                                                stddev = numpy.sqrt(2./(n_char * n_char_embedding)) ) )

                self.ner_embedding = tf.Variable( tf.truncated_normal( [n_label_type + 1, n_ner_embedding], 
                                                stddev = numpy.sqrt(2./(n_label_type * n_ner_embedding)) ) )

                self.bigram_embedding = tf.Variable( tf.truncated_normal( [96 * 96, n_char_embedding],
                                                stddev = numpy.sqrt(2./(96 * 96 * n_char_embedding)) ) )

                self.kernels = [ tf.Variable( tf.truncated_normal( [h, n_char_embedding, 1, d], 
                                                              stddev = numpy.sqrt(2./(h * n_char_embedding * d)) ) ) for \
                            (h, d) in zip( kernel_height, kernel_depth ) ]

                self.kernel_bias = [ tf.Variable( tf.zeros( [d] ) ) for d in kernel_depth ]

                # the U matrix in the HOPE paper
                if hope_out > 0:
                    self.U = tf.Variable( tf.truncated_normal( [hope_in, hope_out],
                                                          stddev = numpy.sqrt(2./(hope_in * hope_out)) ) )

                for i, o in zip( n_in_shared, n_out_shared ):
                    self.shared_layer_weights.append( tf.Variable( tf.truncated_normal( [i, o], stddev = numpy.sqrt(2./(i * o)) ) ) )
                    self.shared_layer_b.append( tf.Variable( tf.zeros( [o] ) )  )

                for i, o in zip( n_in_ontonotes, n_out_ontonotes ):
                    self.ontonotes_layer_weights.append( tf.Variable( tf.truncated_normal( [i, o], stddev = numpy.sqrt(2./(i * o)) ) ) )
                    self.ontonotes_layer_b.append( tf.Variable( tf.zeros( [o] ) )  )

                for i, o in zip( n_in_conll, n_out_conll ):
                    self.conll_layer_weights.append( tf.Variable( tf.truncated_normal( [i, o], stddev = numpy.sqrt(2./(i * o)) ) ) )
                    self.conll_layer_b.append( tf.Variable( tf.zeros( [o] ) )  )

                if n_pattern > 0:
                    # case-insensitive patterns
                    # stddev = numpy.sqrt(2./(n_pattern * n_word1))
                    stddev = numpy.sqrt(2./n_word1 )
                    self.pattern1 = []
                    self.pattern1_bias = []
                    for _ in xrange(4):
                        self.pattern1.append(
                            tf.Variable(
                                tf.truncated_normal( [n_pattern, n_word1], stddev = stddev )
                            )
                        )
                        self.pattern1_bias.append(
                            tf.Variable(
                                tf.truncated_normal( [n_pattern], stddev = stddev )
                            )
                        )

                    # case-sensitive patterns
                    # stddev = numpy.sqrt(2./(n_pattern * n_word2))
                    stddev = numpy.sqrt(2./n_word2)
                    self.pattern2 = []
                    self.pattern2_bias = []
                    for _ in xrange(4):
                        self.pattern2.append(
                            tf.Variable(
                                tf.truncated_normal( [n_pattern, n_word2], stddev = stddev )
                            )
                        )
                        self.pattern2_bias.append(
                            tf.Variable(
                                tf.truncated_normal( [n_pattern], stddev = stddev )
                            )
                        )

                    del stddev

            # parameters that need calculation for the cost function
            if hope_out > 0:
                self.param.append( self.U )
            self.param.append( self.char_embedding )
            self.param.append( self.conv_embedding )
            self.param.append( self.ner_embedding )
            self.param.append( self.bigram_embedding )
            self.param.extend( self.kernels )
            self.param.extend( self.kernel_bias )
            self.param.extend( self.shared_layer_weights )
            self.param.extend( self.shared_layer_b )

            self.ontonotes_param = self.param 
            self.conll_param = self.param
            if n_pattern > 0:
                self.param.extend( self.pattern1 )
                self.param.extend( self.pattern2 )
                self.param.extend( self.pattern1_bias )
                self.param.extend( self.pattern2_bias )


            self.conll_param.extend(self.conll_layer_weights)
            self.conll_param.extend(self.conll_layer_b)

            self.ontonotes_param.extend(self.ontonotes_layer_weights)
            self.ontonotes_param.extend(self.ontonotes_layer_b)
            # add KBP later

            logger.info( 'variable defined' )

            ################################################################################

            char_cube = tf.expand_dims( tf.gather( self.conv_embedding, self.char_idx ), 3 )

            # char-level CNN
            char_conv = [ tf.reduce_max( tf.nn.tanh( tf.nn.conv2d( char_cube, 
                                                                   kk, 
                                                                   [1, 1, 1, 1], 
                                                                   'VALID' ) + bb ),
                                                     reduction_indices = [1, 2] ) \
                            for kk,bb in zip( self.kernels, self.kernel_bias) ]

            ###########################
            ### WORD-LEVEL FEATURES ###
            ###########################

            # case insensitive excluding fragment
            lw1 = tf.SparseTensor( self.lw1_indices, self.lw1_values, self.shape1 )
            rw1 = tf.SparseTensor( self.rw1_indices, self.rw1_values, self.shape1 )

            # case insensitive including fragment
            lw2 = tf.SparseTensor( self.lw2_indices, self.lw2_values, self.shape1 )
            rw2 = tf.SparseTensor( self.rw2_indices, self.rw2_values, self.shape1 )

            # case insensitive bow fragment
            bow1 = tf.SparseTensor( self.bow1_indices, self.bow1_values, self.shape1 )

            # CASE SENSITIVE
            # --------------
            # case sensitive excluding fragment
            lw3 = tf.SparseTensor( self.lw3_indices, self.lw3_values, self.shape2 )
            rw3 = tf.SparseTensor( self.rw3_indices, self.rw3_values, self.shape2 )

            # case sensitive including fragment
            lw4 = tf.SparseTensor( self.lw4_indices, self.lw4_values, self.shape2 )
            rw4 = tf.SparseTensor( self.rw4_indices, self.rw4_values, self.shape2 )

            # case sensitive bow fragment
            bow2 = tf.SparseTensor( self.bow2_indices, self.bow2_values, self.shape2 )

            # left and right bigram context
            lbc = tf.SparseTensor( self.lbc_indices, self.lbc_values, self.shape3 )
            rbc = tf.SparseTensor( self.rbc_indices, self.rbc_values, self.shape3 )

            if n_pattern > 0:
                def sparse_fofe( fofe, pattern, pattern_bias ):
                    indices_f32 = tf.to_float( fofe.indices )
                    # n_pattern * n_example
                    sp_w = tf.transpose(
                                # tf.nn.relu( 
                                tf.nn.softmax(
                                    tf.sparse_tensor_dense_matmul( 
                                            fofe, pattern, adjoint_b = True,
                                            name = 'sparse-weight' )
                                    + pattern_bias
                                )
                            )
                    # replicate the indices
                    indices_rep = tf.tile( indices_f32, [n_pattern, 1] )
                    # add n_pattern as highest dimension
                    nnz = tf.to_int64( tf.shape(fofe.indices)[0] )
                    indices_high = tf.to_float(
                                        tf.reshape(
                                            tf.range( n_pattern * nnz) / nnz,
                                            [-1, 1]
                                        )
                                    )
                    # effectively n_pattern * n_example * n_word
                    indices_ext = tf.concat( [ indices_high, indices_rep ], 1 )
                    # pattern * fofe
                    values_ext = tf.multiply( 
                                        tf.tile( fofe.values, [n_pattern] ),
                                        tf.gather_nd( 
                                            pattern,
                                            tf.to_int64(
                                                tf.concat( [ indices_ext[:,0:1], indices_ext[:,2:3] ], 1 )
                                            )
                                        )
                                    )
                    # re-weight by pattern importance
                    values_ext_2d = tf.multiply(
                                        values_ext,
                                        tf.gather_nd( sp_w, tf.to_int64(indices_ext[:,0:2]) )
                                    )
                    values_att = tf.reduce_sum( tf.reshape( values_ext_2d, [n_pattern, -1] ), 0 )
                    # re-group it as a SparseTensor
                    return tf.SparseTensor( fofe.indices, values_att, fofe.dense_shape )

                lw1 = sparse_fofe( lw1, self.pattern1[0], self.pattern1_bias[0] )
                rw1 = sparse_fofe( rw1, self.pattern1[1], self.pattern1_bias[1] )
                lw2 = sparse_fofe( lw2, self.pattern1[2], self.pattern2_bias[2] )
                rw2 = sparse_fofe( rw2, self.pattern1[3], self.pattern2_bias[3] )

                lw3 = sparse_fofe( lw3, self.pattern2[0], self.pattern2_bias[0] )
                rw3 = sparse_fofe( rw3, self.pattern2[1], self.pattern2_bias[1] )
                lw4 = sparse_fofe( lw4, self.pattern2[2], self.pattern2_bias[2] )
                rw4 = sparse_fofe( rw4, self.pattern2[3], self.pattern2_bias[3] )

            # all sparse feature after projection

            # case-insensitive in English / word embedding in Chinese
            # case-insensitive bfofe with candidate word(s)

            # 1st layer: word embedding, char embedding, ner embedding
            lwp1 = tf.sparse_tensor_dense_matmul( lw1, self.word_embedding_1,
                                                  name = 'emb1-left-fofe-excl-proj' )
            rwp1 = tf.sparse_tensor_dense_matmul( rw1, self.word_embedding_1,
                                                  name = 'emb1-right-fofe-excl-proj' )

            # case-insensitive bfofe without candidate word(s)
            lwp2 = tf.sparse_tensor_dense_matmul( lw2, self.word_embedding_1,
                                                  name = 'emb1-left-fofe-incl-proj' )
            rwp2 = tf.sparse_tensor_dense_matmul( rw2, self.word_embedding_1,
                                                  name = 'emb1-right-fofe-incl-proj' )

            # case-insensitive bag-of-words
            bowp1 = tf.sparse_tensor_dense_matmul( bow1, self.word_embedding_1 )

            # case-sensitive in English / character embedding in Chinese
            # case-sensitive bfofe with candidate word(s)
            lwp3 = tf.sparse_tensor_dense_matmul( lw3, self.word_embedding_2,
                                                  name = 'emb2-left-fofe-excl-proj' )
            rwp3 = tf.sparse_tensor_dense_matmul( rw3, self.word_embedding_2,
                                                  name = 'emb2-right-fofe-excl-proj' )

            # case-sensitive bfofe without candidate word(s)
            lwp4 = tf.sparse_tensor_dense_matmul( lw4, self.word_embedding_2,
                                                  name = 'emb2-left-fofe-incl-proj' )
            rwp4 = tf.sparse_tensor_dense_matmul( rw4, self.word_embedding_2,
                                                  name = 'emb2-right-fofe-incl-proj' )

            # case-sensitive bag-of-words
            bowp2 = tf.sparse_tensor_dense_matmul( bow2, self.word_embedding_2 )

            # dense features after projection
            # char-level bfofe of candidate word(s)
            lcp = tf.matmul( self.lc_fofe, self.char_embedding )
            rcp = tf.matmul( self.rc_fofe, self.char_embedding )
            
            lip = tf.matmul( self.li_fofe, self.char_embedding )
            rip = tf.matmul( self.ri_fofe, self.char_embedding )

            # bigram char-fofe
            lbcp = tf.sparse_tensor_dense_matmul( lbc, self.bigram_embedding )
            rbcp = tf.sparse_tensor_dense_matmul( rbc, self.bigram_embedding )

            ner_projection = tf.matmul( self.ner_cls_match, self.ner_embedding )

            # all possible features
            feature_list = [ [lwp1, rwp1], [lwp2, rwp2], [bowp1],
                             [lwp3, rwp3], [lwp4, rwp4], [bowp2],
                             [lcp, rcp], [lip, rip], [ner_projection],
                             char_conv, [lbcp, rbcp] ]

            # divide up the used and unused features
            used, not_used = [], [] 

            # decide what feature to use
            for ith, f in enumerate( feature_list ):
                if (1 << ith) & feature_choice > 0: 
                    used.extend( f )
                else:
                    not_used.extend( f )

            # only use the features requested for use
            feature_list = used #+ not_used

            # a tensor containing all the feature vectors
            # 2nd layer: concatenate
            feature = tf.concat( feature_list, 1 )

            if hope_out > 0:
                hope = tf.matmul( feature, self.U )
                shared_layer_output = [ hope ]
            else:
                # layer 1 and 2
                shared_layer_output = [ tf.nn.dropout( feature, self.keep_prob ) ]

            #=======================
            #==== Shared layers ====
            #=======================

            # calculate the output by multiplying the input by the weights
            # use ReLU as an activation function
            # 3rd layer to 11th layer: linear, relu, dropout
            for i in xrange( len(self.shared_layer_weights) ):
                logger.info("test shared0: " + str(self.shared_layer_weights[i].get_shape()))
                test = tf.matmul( shared_layer_output[-1], self.shared_layer_weights[i] )
                logger.info("test shared1: " + str(test.get_shape()))
                # linear layer (also 12th layer: linear)
                shared_layer_output.append( tf.matmul( shared_layer_output[-1], self.shared_layer_weights[i] ) + self.shared_layer_b[i] )
                # ReLU layer
                shared_layer_output[-1] = tf.nn.relu(shared_layer_output[-1] )
                # Dropout layer
                shared_layer_output[-1] = tf.nn.dropout(shared_layer_output[-1], self.keep_prob )

            conll_layer_output = shared_layer_output
            ontonotes_layer_output = shared_layer_output

            #============================
            #==== OntoNotes layers ======
            #============================

            for i in xrange(len(self.ontonotes_layer_weights)):
                test = tf.matmul( ontonotes_layer_output[-1], self.ontonotes_layer_weights[i] )
                logger.info("test ontonotes: " + str(test.get_shape()))
                ontonotes_layer_output.append( tf.matmul( ontonotes_layer_output[-1], self.ontonotes_layer_weights[i] ) + self.ontonotes_layer_b[i] )
                if i < len(self.ontonotes_layer_weights) - 1:
                    # ReLU layer
                    ontonotes_layer_output[-1] = tf.nn.relu(ontonotes_layer_output[-1] )
                if i < len(self.ontonotes_layer_weights) - 2:
                    # Dropout layer
                    ontonotes_layer_output[-1] = tf.nn.dropout(ontonotes_layer_output[-1], self.keep_prob )

            #=============================
            #==== CoNLL 2003 layers ======
            #=============================

            for i in xrange(len(self.conll_layer_weights)):
                test = tf.matmul(conll_layer_output[-1], self.conll_layer_weights[i] )
                logger.info("test conll: " + str(test.get_shape()))
                conll_layer_output.append( tf.matmul(conll_layer_output[-1], self.conll_layer_weights[i] ) + self.conll_layer_b[i] )
                if i < len(self.conll_layer_weights) - 1:
                    # ReLU layer
                    conll_layer_output[-1] = tf.nn.relu(conll_layer_output[-1] )
                if i < len(self.conll_layer_weights) - 2:
                    # Dropout layer
                    conll_layer_output[-1] = tf.nn.dropout(conll_layer_output[-1], self.keep_prob )

            #=============================

            # 13th layer: log_softmax
            self.ontonotes_xent = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( 
                                            logits = ontonotes_layer_output[-1], labels = self.label ) )

            self.conll_xent = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( 
                                            logits = conll_layer_output[-1], labels = self.label ) )

            if config.l1 > 0:
                # self.param contains weights, bias, character conv embedding variables etc
                for param in self.ontonotes_param:
                    self.ontonotes_xent = self.ontonotes_xent + config.l1 * tf.reduce_sum( tf.abs( param ) )

                for param in self.conll_param:
                    self.conll_xent = self.conll_xent + config.l1 * tf.reduce_sum( tf.abs( param ) )


            if config.l2 > 0:
                for param in self.ontonotes_param:
                    self.ontonotes_xent = self.ontonotes_xent + config.l2 * tf.nn.l2_loss( param )

                for param in self.conll_param:
                    self.conll_xent = self.conll_xent + config.l2 * tf.nn.l2_loss( param )

            self.ontonotes_predicted_values = tf.nn.softmax( ontonotes_layer_output[-1] )
            _, top_ontonotes_indices = tf.nn.top_k( self.ontonotes_predicted_values )
            self.ontonotes_predicted_indices = tf.reshape( top_ontonotes_indices, [-1] )

            self.conll_predicted_values = tf.nn.softmax( conll_layer_output[-1] )
            _, top_conll_indices = tf.nn.top_k( self.conll_predicted_values )
            self.conll_predicted_indices = tf.reshape( top_conll_indices, [-1] )

            varlist = self.shared_layer_weights + self.shared_layer_b
            # add KBP later

            # fully connected layers are must-trained layers
            # Want to change the weights and bias terms only for minimization of cross entropy 
            # cost function
            ontonotes_varlist = varlist + self.ontonotes_layer_weghts + self.ontonotes_layer_b
            ontonotes_fully_connected_train_step = tf.train.MomentumOptimizer( self.lr, 
                                                                     self.config.momentum, 
                                                                     use_locking = False ) \
                                           .minimize(self.ontonotes_xent, var_list = ontonotes_varlist)

            conll_varlist = varlist + self.conll_layer_weights + self.conll_layer_b
            conll_fully_connected_train_step = tf.train.MomentumOptimizer(self.lr,
                                                                self.config.momentum,
                                                                use_locking = False) \
                                        .minimize(self.conll_xent, var_list = conll_varlist)

            # a list of things to train
            self.conll_train_step = [ conll_fully_connected_train_step ]
            self.ontonotes_train_step = [ ontonotes_fully_connected_train_step ]

            if n_pattern > 0:
                __lambda = 0.001
                self.xentPlusSparse = self.xent
                # L1-norm, not working
                # for i in xrange(4):
                #     self.xentPlusSparse += __lambda * tf.reduce_sum( tf.abs( self.pattern1[i] ) )
                #     self.xentPlusSparse += __lambda * tf.reduce_sum( tf.abs( self.pattern2[i] ) )
                
                # orthogonality
                for p in self.pattern1 + self.pattern2:
                    self.xentPlusSparse += tf.reduce_sum(
                                                tf.abs(
                                                    tf.matmul( p, p, 
                                                               transpose_a = False,
                                                               transpose_b = True ) 
                                                    - tf.eye( n_pattern )
                                                )
                                            ) * __lambda
                sparse_fofe_step = tf.train.GradientDescentOptimizer(
                                        self.lr,
                                        use_locking = True
                                    ).minimize( 
                                        self.xentPlusSparse,
                                        var_list = self.pattern1 + self.pattern2 +
                                                   self.pattern1_bias +
                                                   self.pattern2_bias
                                    )
                self.train_step.append( sparse_fofe_step )

            # train the word embedding for insensitive case
            if feature_choice & 0b111 > 0:
                # returns an Operation that updates the variables in var_list.
                insensitive_train_step = tf.train.GradientDescentOptimizer( self.lr / 4, 
                                                                            use_locking = True ) \
                                          .minimize( self.xent, var_list = [ self.word_embedding_1 ] )
                self.conll_train_step.append( insensitive_train_step )
                self.ontonotes_train_step.append( insensitive_train_step )

            # train the word embedding for sensitive case
            if feature_choice & (0b111 << 3) > 0:
                sensitive_train_step = tf.train.GradientDescentOptimizer( self.lr / 4, 
                                                                          use_locking = True ) \
                                          .minimize( self.xent, var_list = [ self.word_embedding_2 ] )
                self.conll_train_step.append( sensitive_train_step )
                self.ontonotes_train_step.append( sensitive_train_step )

            # train the char embedding for insensitive case
            if feature_choice & (0b11 << 6) > 0:
                char_embedding_train_step = tf.train.GradientDescentOptimizer( self.lr / 2, 
                                                                               use_locking = True ) \
                                              .minimize( self.xent, var_list = [ self.char_embedding ] )
                self.conll_train_step.append( char_embedding_train_step )
                self.ontonotes_train_step.append( char_embedding_train_step )

            # train the NER embedding
            if feature_choice & (1 << 8) > 0:
                ner_embedding_train_step = tf.train.GradientDescentOptimizer( self.lr, 
                                                                              use_locking = True ) \
                                          .minimize( self.xent, var_list = [ self.ner_embedding ] )
                self.conll_train_step.append( ner_embedding_train_step )
                self.ontonotes_train_step.append( ner_embedding_train_step )

            if feature_choice & (1 << 9) > 0:
                char_conv_train_step = tf.train.MomentumOptimizer( self.lr, momentum )\
                                             .minimize( self.xent, 
                                                var_list = [ self.conv_embedding ] + \
                                                             self.kernels + self.kernel_bias )
                self.conll_train_step.append( char_conv_train_step )
                self.ontonotes_train_step.append( char_conv_train_step )

            if feature_choice & (1 << 10) > 0:
                bigram_train_step = tf.train.GradientDescentOptimizer( self.lr / 2, use_locking = True )\
                                            .minimize( self.xent, var_list = [ self.bigram_embedding ] )
                self.conll_train_step.append( bigram_train_step )
                self.ontonotes_train_step.append( bigram_train_step )

            if hope_out > 0:
                __lambda = 0.001
                orthogonal_penalty = tf.reduce_sum(
                                            tf.abs(
                                                tf.matmul( self.U, self.U, transpose_a = True ) - tf.eye(hope_out)
                                            ) 
                                        ) * __lambda
                U_train_step_1 = tf.train.GradientDescentOptimizer(self.lr, use_locking = True) \
                                         .minimize( orthogonal_penalty , var_list = [ self.U ] )
                U_train_step_2 = tf.train.MomentumOptimizer( self.lr, momentum, use_locking = True ) \
                                         .minimize( self.xent, var_list = [ self.U ] )
                self.train_step.append( U_train_step_1 )
                self.train_step.append( U_train_step_2 )

        logger.info( 'computational graph built\n' )

        with self.graph.as_default():
            self.session.run( tf.global_variables_initializer() )
            # self.session.run( tf.variables_initializer( self.param ) )
            self.saver = tf.train.Saver()


    def train( self, mini_batch, dataset, profile = False ):
        """
        Parameters
        ----------
            mini_batch : tuple
            dataset:    0 - CoNLL2003 
                        1 - OntoNotes 
                        2 - KBP

        Returns
        -------
            c : float
        """ 
        l1_values, r1_values, l1_indices, r1_indices, \
        l2_values, r2_values, l2_indices, r2_indices, \
        bow1i, \
        l3_values, r3_values, l3_indices, r3_indices, \
        l4_values, r4_values, l4_indices, r4_indices, \
        bow2i, \
        dense_feature,\
        conv_idx,\
        l5_values, l5_indices, r5_values, r5_indices, \
        target = mini_batch

        if not self.config.strictly_one_hot:
            dense_feature[:,-1] = 0

        if profile:
            options = tf.RunOptions( trace_level = tf.RunOptions.FULL_TRACE )
            run_metadata = tf.RunMetadata()
        else:
            options, run_metadata = None, None

        if dataset == 0:
            train = self.conll_train_step + [self.conll_xent]
        else:
            train = self.ontonotes_train_step + [self.ontonotes_xent]

        c = self.session.run(  
            train,
            feed_dict = {   self.lw1_values: l1_values,
                            self.lw1_indices: l1_indices,
                            self.rw1_values: r1_values,
                            self.rw1_indices: r1_indices,
                            self.lw2_values: l2_values,
                            self.lw2_indices: l2_indices,
                            self.rw2_values: r2_values,
                            self.rw2_indices: r2_indices,
                            self.bow1_indices: bow1i,
                            self.bow1_values: numpy.ones( bow1i.shape[0], dtype = numpy.float32 ),
                            self.lw3_values: l3_values,
                            self.lw3_indices: l3_indices,
                            self.rw3_values: r3_values,
                            self.rw3_indices: r3_indices,
                            self.lw4_values: l4_values,
                            self.lw4_indices: l4_indices,
                            self.rw4_values: r4_values,
                            self.rw4_indices: r4_indices,
                            self.bow2_indices: bow2i,
                            self.bow2_values: numpy.ones( bow2i.shape[0], dtype = numpy.float32 ),
                            self.shape1: (target.shape[0], self.n_word1),
                            self.shape2: (target.shape[0], self.n_word2),
                            self.lc_fofe: dense_feature[:,:128],
                            self.rc_fofe: dense_feature[:,128:256],
                            self.li_fofe: dense_feature[:,256:384],
                            self.ri_fofe: dense_feature[:,384:512],
                            self.ner_cls_match: dense_feature[:,512:],
                            self.char_idx: conv_idx,
                            self.lbc_values : l5_values,
                            self.lbc_indices : l5_indices,
                            self.rbc_values : r5_values,
                            self.rbc_indices : r5_indices,
                            self.shape3 : (target.shape[0], 96 * 96),
                            self.label: target,
                            self.lr: self.config.learning_rate,
                            self.keep_prob: 1 - self.config.drop_rate },
            options = options, 
            run_metadata = run_metadata
        )[-1]

        if profile:
            tf.contrib.tfprof.model_analyzer.print_model_analysis(
                self.graph,
                run_meta = run_metadata,
                tfprof_options = tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY
            )

        return c
        

    def eval( self, mini_batch, dataset ):
        """
        Parameters
        ----------
            mini_batch : tuple

        Returns:
            c : float
            pi : numpy.ndarray
            pv : numpy.ndarray
        """
        l1_values, r1_values, l1_indices, r1_indices, \
        l2_values, r2_values, l2_indices, r2_indices, \
        bow1i, \
        l3_values, r3_values, l3_indices, r3_indices, \
        l4_values, r4_values, l4_indices, r4_indices, \
        bow2i, \
        dense_feature,\
        conv_idx,\
        l5_values, l5_indices, r5_values, r5_indices, \
        target = mini_batch

        if not self.config.strictly_one_hot:
            dense_feature[:,-1] = 0

        if dataset == 0:
            train = [self.conll_xent, self.conll_predicted_indices, self.conll_predicted_values]
        else:
            train = [self.ontonotes_xent, self.ontonotes_predicted_indices, self.ontonotes_predicted_values]


        c, pi, pv = self.session.run( train, 
                                        feed_dict = {   self.lw1_values: l1_values,
                                                        self.lw1_indices: l1_indices,
                                                        self.rw1_values: r1_values,
                                                        self.rw1_indices: r1_indices,
                                                        self.lw2_values: l2_values,
                                                        self.lw2_indices: l2_indices,
                                                        self.rw2_values: r2_values,
                                                        self.rw2_indices: r2_indices,
                                                        self.bow1_indices: bow1i,
                                                        self.bow1_values: numpy.ones( bow1i.shape[0], dtype = numpy.float32 ),
                                                        self.lw3_values: l3_values,
                                                        self.lw3_indices: l3_indices,
                                                        self.rw3_values: r3_values,
                                                        self.rw3_indices: r3_indices,
                                                        self.lw4_values: l4_values,
                                                        self.lw4_indices: l4_indices,
                                                        self.rw4_values: r4_values,
                                                        self.rw4_indices: r4_indices,
                                                        self.bow2_indices: bow2i,
                                                        self.bow2_values: numpy.ones( bow2i.shape[0], dtype = numpy.float32 ),
                                                        self.shape1: (target.shape[0], self.n_word1),
                                                        self.shape2: (target.shape[0], self.n_word2),
                                                        self.lc_fofe: dense_feature[:,:128],
                                                        self.rc_fofe: dense_feature[:,128:256],
                                                        self.li_fofe: dense_feature[:,256:384],
                                                        self.ri_fofe: dense_feature[:,384:512],
                                                        self.ner_cls_match: dense_feature[:,512:],
                                                        self.char_idx: conv_idx,
                                                        self.lbc_values : l5_values,
                                                        self.lbc_indices : l5_indices,
                                                        self.rbc_values : r5_values,
                                                        self.rbc_indices : r5_indices,
                                                        self.shape3 : (target.shape[0], 96 * 96),
                                                        self.label: target,
                                                        self.keep_prob: 1 } ) 

        return c, pi, pv

    def tofile( self, filename ):
        """
        Parameters
        ----------
            filename : str
                The current model will be stored in basename.{tf,config}
        """
        self.saver.save( self.session, filename )
        with open( filename + '.config', 'wb' ) as fp:
            cPickle.dump( self.config, fp )


    def fromfile( self, filename ):
        """
            filename : str
                The current model will be restored from basename.{tf,config}
        """
        self.saver.restore( self.session, filename )


    def __del__( self ):
        self.session.close()

########################################################################

