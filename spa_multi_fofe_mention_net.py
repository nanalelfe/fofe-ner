#!/home/chwang/anaconda2/envs/tensorflow/bin/python

#/eecs/research/asr/mingbin/python-workspace/hopeless/bin/python

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

RICH_N_LABELS = 10
LIGHT_N_LABELS = 11
KBP_N_LABELS = 10

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
        self.language = 'spa'
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

        # add a U matrix between projected feature and fully-connected layers

        middle = [ int(s) for s in layer_size.split(',') ]

        n_in_shared = [ hope_out if hope_out > 0 else hope_in ] + middle[:1]
        # n_out_shared = n_in_shared[1:] + [n_in_shared[-1]]
        n_out_shared = middle[-2:]

        n_in_rich = n_out_shared[-1:]
        n_out_rich = [ RICH_N_LABELS + 1 ]

        n_in_light = n_out_shared[-1:]
        n_out_light = [ LIGHT_N_LABELS + 1 ]

        n_in_kbp = n_out_shared[-2:]
        n_out_kbp = n_in_kbp[1:] +  [KBP_N_LABELS + 1]

        logger.info( 'n_in_shared: ' + str(n_in_shared) )
        logger.info( 'n_out_shared: ' + str(n_out_shared) )
        logger.info( 'n_in_light: ' + str(n_in_light) )
        logger.info( 'n_out_light: ' + str(n_out_light) )
        logger.info( 'n_in_rich: ' + str(n_in_rich) )
        logger.info( 'n_out_rich: ' + str(n_out_rich) )
        logger.info( 'n_in_kbp' + str(n_in_kbp))
        logger.info( 'n_out_kbp'+ str(n_out_kbp))

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
            self.ner_cls_match_rich = tf.placeholder( tf.float32, [None, RICH_N_LABELS + 1], name = 'gazetteer_rich' )
            self.ner_cls_match_light = tf.placeholder( tf.float32, [None, LIGHT_N_LABELS + 1], name = 'gazetteer_light' )
            self.ner_cls_match_kbp = tf.placeholder(tf.float32, [None, KBP_N_LABELS + 1], name = 'gazetteer_kbp')

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
            self.rich_layer_weights = []
            self.light_layer_weights = []
            self.kbp_layer_weights = []

            # Bias
            self.shared_layer_b = []   
            self.rich_layer_b = []
            self.light_layer_b = []
            self.param = []
            self.kbp_layer_b = []

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
                val_rng_rich = numpy.float32(2.5 / numpy.sqrt(RICH_N_LABELS + n_ner_embedding + 1))
                val_rng_light = numpy.float32(2.5 / numpy.sqrt(LIGHT_N_LABELS + n_ner_embedding + 1))
                var_rng_kbp = numpy.float32(2.5 / numpy.sqrt(KBP_N_LABELS + n_ner_embedding + 1))

                # random initialization of word embeddings
                self.ner_embedding_rich = tf.Variable( tf.random_uniform( 
                                        [RICH_N_LABELS + 1 , n_ner_embedding], minval = -val_rng_rich, maxval = val_rng_rich ) )

                self.ner_embedding_light = tf.Variable( tf.random_uniform( 
                                        [LIGHT_N_LABELS + 1 ,n_ner_embedding], minval = -val_rng_light, maxval = val_rng_light ) )

                self.ner_embedding_kbp = tf.Variable( tf.random_uniform( 
                                        [KBP_N_LABELS + 1 ,n_ner_embedding], minval = -var_rng_kbp, maxval = var_rng_kbp ) )
                
                val_rng = numpy.float32(2.5 / numpy.sqrt(96 * 96 + n_char_embedding))
                self.bigram_embedding = tf.Variable( tf.random_uniform( 
                                        [96 * 96, n_char_embedding], minval = -val_rng, maxval = val_rng ) )

                self.kernels = [ tf.Variable( tf.random_uniform( 
                                    [h, n_char_embedding, 1, d], 
                                    minval = -2.5 / numpy.sqrt(1 + h * n_char_embedding * d), 
                                    maxval = 2.5 / numpy.sqrt(1 + h * n_char_embedding * d) ) ) for \
                                (h, d) in zip( kernel_height, kernel_depth ) ]

                self.kernel_bias = [ tf.Variable( tf.zeros( [d] ) ) for d in kernel_depth ]

                # Initialize the weights of each module using uniform
                for i, o in zip( n_in_shared, n_out_shared ):
                    val_rng = numpy.float32(2.5 / numpy.sqrt(i + o))
                    # random_uniform : Returns a tensor of the specified shape filled with random uniform values.
                    self.shared_layer_weights.append( tf.Variable( tf.random_uniform( [i, o], minval = -val_rng, maxval = val_rng ) ) )
                    self.shared_layer_b.append( tf.Variable( tf.zeros( [o] ) )  )

                for i, o in zip( n_in_rich, n_out_rich ):
                    val_rng = numpy.float32(2.5 / numpy.sqrt(i + o))

                    self.rich_layer_weights.append( tf.Variable( tf.random_uniform( [i, o], minval = -val_rng, maxval = val_rng ) ) )
                    self.rich_layer_b.append( tf.Variable( tf.zeros( [o] ) )  )

                for i, o in zip( n_in_light, n_out_light ):
                    val_rng = numpy.float32(2.5 / numpy.sqrt(i + o))

                    self.light_layer_weights.append( tf.Variable( tf.random_uniform( [i, o], minval = -val_rng, maxval = val_rng ) ) )
                    self.light_layer_b.append( tf.Variable( tf.zeros( [o] ) )  )

                for i, o in zip( n_in_kbp, n_out_kbp ):
                    val_rng = numpy.float32(2.5 / numpy.sqrt(i + o))

                    self.kbp_layer_weights.append( tf.Variable( tf.random_uniform( [i, o], minval = -val_rng, maxval = val_rng ) ) )
                    self.kbp_layer_b.append( tf.Variable( tf.zeros( [o] ) )  )


                
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


                for i, o in zip( n_in_shared, n_out_shared ):
                    self.shared_layer_weights.append( tf.Variable( tf.truncated_normal( [i, o], stddev = numpy.sqrt(2./(i * o)) ) ) )
                    self.shared_layer_b.append( tf.Variable( tf.zeros( [o] ) )  )

                for i, o in zip( n_in_rich, n_out_rich ):
                    self.rich_layer_weights.append( tf.Variable( tf.truncated_normal( [i, o], stddev = numpy.sqrt(2./(i * o)) ) ) )
                    self.rich_layer_b.append( tf.Variable( tf.zeros( [o] ) )  )

                for i, o in zip( n_in_light, n_out_light ):
                    self.light_layer_weights.append( tf.Variable( tf.truncated_normal( [i, o], stddev = numpy.sqrt(2./(i * o)) ) ) )
                    self.light_layer_b.append( tf.Variable( tf.zeros( [o] ) )  )

                for i, o in zip( n_in_kbp, n_out_kbp ):
                    self.kbp_layer_weights.append( tf.Variable( tf.truncated_normal( [i, o], stddev = numpy.sqrt(2./(i * o)) ) ) )
                    self.kbp_layer_b.append( tf.Variable( tf.zeros( [o] ) )  )

            # parameters that need calculation for the cost function

            self.param.append( self.char_embedding )
            self.param.append( self.conv_embedding )
            self.param.append( self.bigram_embedding )
            self.param.extend( self.kernels )
            self.param.extend( self.kernel_bias )
            self.param.extend( self.shared_layer_weights )
            self.param.extend( self.shared_layer_b )

            self.light_param = self.param[:]
            self.rich_param = self.param[:]
            self.kbp_param = self.param[:]

            self.rich_param.append( self.ner_embedding_rich )
            self.rich_param.extend(self.rich_layer_weights)
            self.rich_param.extend(self.rich_layer_b)

            self.light_param.append( self.ner_embedding_light )
            self.light_param.extend(self.light_layer_weights)
            self.light_param.extend(self.light_layer_b)

            self.kbp_param.append( self.ner_embedding_kbp )
            self.kbp_param.extend(self.kbp_layer_weights)
            self.kbp_param.extend(self.kbp_layer_b)
            
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

            ner_projection_rich = tf.matmul( self.ner_cls_match_rich, self.ner_embedding_rich )
            ner_projection_light = tf.matmul( self.ner_cls_match_light, self.ner_embedding_light )
            ner_projection_kbp = tf.matmul(self.ner_cls_match_kbp, self.ner_embedding_kbp)

            # all possible features
            feature_list = [ [lwp1, rwp1], [lwp2, rwp2], [bowp1],
                             [lwp3, rwp3], [lwp4, rwp4], [bowp2],
                             [lcp, rcp], [lip, rip], [ner_projection_rich,
                             ner_projection_light, ner_projection_kbp], char_conv, [lbcp, rbcp] ]

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

            # layer 1 and 2
            shared_layer_output = [ tf.nn.dropout( feature, self.keep_prob ) ]

            #=======================
            #==== Shared layers ====
            #=======================

            # calculate the output by multiplying the input by the weights
            # use ReLU as an activation function
            # 3rd layer to 11th layer: linear, relu, dropout
            for i in xrange( len(self.shared_layer_weights) ):
                # linear layer (also 12th layer: linear)
                shared_layer_output.append( tf.matmul( shared_layer_output[-1], self.shared_layer_weights[i] ) + self.shared_layer_b[i] )
                # ReLU layer
                shared_layer_output[-1] = tf.nn.relu(shared_layer_output[-1] )
                # Dropout layer
                shared_layer_output[-1] = tf.nn.dropout(shared_layer_output[-1], self.keep_prob )

            rich_layer_output = [shared_layer_output[-1]]
            light_layer_output = [shared_layer_output[-1]]
            kbp_layer_output = [shared_layer_output[-1]]

            #===========================
            #==== Rich ERE layers ======
            #===========================

            for i in xrange(len(self.rich_layer_weights)):
                rich_layer_output.append( tf.matmul( rich_layer_output[-1], self.rich_layer_weights[i] ) + self.rich_layer_b[i] )
                if i < len(self.rich_layer_weights) - 1:
                    # ReLU layer
                    rich_layer_output[-1] = tf.nn.relu(rich_layer_output[-1] )
                if i < len(self.rich_layer_weights) - 2:
                    # Dropout layer
                    rich_layer_output[-1] = tf.nn.dropout(rich_layer_output[-1], self.keep_prob )

            #============================
            #==== Light ERE layers ======
            #============================

            for i in xrange(len(self.light_layer_weights)):
                light_layer_output.append( tf.matmul(light_layer_output[-1], self.light_layer_weights[i] ) + self.light_layer_b[i] )
                if i < len(self.light_layer_weights) - 1:
                    # ReLU layer
                    light_layer_output[-1] = tf.nn.relu(light_layer_output[-1] )
                if i < len(self.light_layer_weights) - 2:
                    # Dropout layer
                    light_layer_output[-1] = tf.nn.dropout(light_layer_output[-1], self.keep_prob )

            #===========================
            #==== KBP 2003 layers ======
            #===========================

            for i in xrange(len(self.kbp_layer_weights)):
                kbp_layer_output.append(tf.matmul(kbp_layer_output[-1], self.kbp_layer_weights[i]) + self.kbp_layer_b[i])
                # ReLU layer
                if i < len(self.kbp_layer_weights) - 1:
                    kbp_layer_output[-1] = tf.nn.relu(kbp_layer_output[-1])
                # Dropout layer
                if i < len(self.kbp_layer_weights) - 2:
                    kbp_layer_output[-1] = tf.nn.dropout(kbp_layer_output[-1], self.keep_prob)

            #===========================

            # 13th layer: log_softmax
            self.light_xent = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( 
                                            logits = light_layer_output[-1], labels = self.label ) )

            self.rich_xent = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( 
                                            logits = rich_layer_output[-1], labels = self.label ) )

            self.kbp_xent = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( 
                                            logits = kbp_layer_output[-1], labels = self.label ) )

            self.light_predicted_values = tf.nn.softmax( light_layer_output[-1] )
            _, top_light_indices = tf.nn.top_k( self.light_predicted_values )
            self.light_predicted_indices = tf.reshape( top_light_indices, [-1] )

            self.rich_predicted_values = tf.nn.softmax( rich_layer_output[-1] )
            _, top_rich_indices = tf.nn.top_k( self.rich_predicted_values )
            self.rich_predicted_indices = tf.reshape( top_rich_indices, [-1] )

            self.kbp_predicted_values = tf.nn.softmax( kbp_layer_output[-1] )
            _, top_kbp_indices = tf.nn.top_k( self.kbp_predicted_values )
            self.kbp_predicted_indices = tf.reshape( top_kbp_indices, [-1] )

            varlist = self.shared_layer_weights + self.shared_layer_b
            # add KBP later

            # fully connected layers are must-trained layers
            # Want to change the weights and bias terms only for minimization of cross entropy 
            # cost function
            light_varlist = varlist + self.light_layer_weights + self.light_layer_b
            light_fully_connected_train_step = tf.train.MomentumOptimizer( self.lr, 
                                                                     self.config.momentum, 
                                                                     use_locking = False ) \
                                           .minimize(self.light_xent, var_list = light_varlist)

            rich_varlist = varlist + self.rich_layer_weights + self.rich_layer_b
            rich_fully_connected_train_step = tf.train.MomentumOptimizer(self.lr,
                                                                self.config.momentum,
                                                                use_locking = False) \
                                        .minimize(self.rich_xent, var_list = rich_varlist)

            kbp_varlist = varlist + self.kbp_layer_weights + self.kbp_layer_b
            kbp_fully_connected_train_step = tf.train.MomentumOptimizer(self.lr,
                                                                self.config.momentum,
                                                                use_locking = False) \
                                        .minimize(self.kbp_xent, var_list = kbp_varlist)


            # a list of things to train
            self.rich_train_step = [ rich_fully_connected_train_step ]
            self.light_train_step = [ light_fully_connected_train_step ]
            self.kbp_train_step = [kbp_fully_connected_train_step]

            # train the word embedding for insensitive case
            if feature_choice & 0b111 > 0:
                # returns an Operation that updates the variables in var_list.
                rich_insensitive_train_step = tf.train.GradientDescentOptimizer( self.lr / 4, 
                                                                            use_locking = True ) \
                                          .minimize( self.rich_xent, var_list = [ self.word_embedding_1 ] )
                light_insensitive_train_step = tf.train.GradientDescentOptimizer( self.lr / 4, 
                                                                            use_locking = True ) \
                                          .minimize( self.light_xent, var_list = [ self.word_embedding_1 ] )

                kbp_insensitive_train_step = tf.train.GradientDescentOptimizer( self.lr / 4, 
                                                                            use_locking = True ) \
                                          .minimize( self.kbp_xent, var_list = [ self.word_embedding_1 ] )

                self.rich_train_step.append( rich_insensitive_train_step )
                self.light_train_step.append( light_insensitive_train_step )
                self.kbp_train_step.append( kbp_insensitive_train_step )

            # train the word embedding for sensitive case
            if feature_choice & (0b111 << 3) > 0:
                rich_sensitive_train_step = tf.train.GradientDescentOptimizer( self.lr / 4, 
                                                                          use_locking = True ) \
                                          .minimize( self.rich_xent, var_list = [ self.word_embedding_2 ] )
                light_sensitive_train_step = tf.train.GradientDescentOptimizer( self.lr / 4, 
                                                                          use_locking = True ) \
                                          .minimize( self.light_xent, var_list = [ self.word_embedding_2 ] )

                kbp_sensitive_train_step = tf.train.GradientDescentOptimizer(self.lr / 4,   
                                                                    use_locking = True) \
                                            .minimize( self.kbp_xent, var_list = [ self.word_embedding_2 ] )

                self.rich_train_step.append( rich_sensitive_train_step )
                self.light_train_step.append( light_sensitive_train_step )
                self.kbp_train_step.append(kbp_sensitive_train_step)

            # train the char embedding for insensitive case
            if feature_choice & (0b11 << 6) > 0:
                rich_char_embedding_train_step = tf.train.GradientDescentOptimizer( self.lr / 2, 
                                                                               use_locking = True ) \
                                              .minimize( self.rich_xent, var_list = [ self.char_embedding ] )
                light_char_embedding_train_step = tf.train.GradientDescentOptimizer( self.lr / 2, 
                                                                               use_locking = True ) \
                                              .minimize( self.light_xent, var_list = [ self.char_embedding ] )
                kbp_char_embedding_train_step = tf.train.GradientDescentOptimizer(self.lr / 2,
                                                                                use_locking = True) \
                                                .minimize(self.kbp_xent, var_list = [self.char_embedding])

                self.rich_train_step.append( rich_char_embedding_train_step )
                self.light_train_step.append( light_char_embedding_train_step )
                self.kbp_train_step.append(kbp_char_embedding_train_step)

            # train the NER embedding
            if feature_choice & (1 << 8) > 0:
                rich_ner_embedding_train_step = tf.train.GradientDescentOptimizer( self.lr, 
                                                                              use_locking = True ) \
                                          .minimize( self.rich_xent, var_list = [ self.ner_embedding_rich ] )
                light_ner_embedding_train_step = tf.train.GradientDescentOptimizer( self.lr, 
                                                                              use_locking = True ) \
                                          .minimize( self.light_xent, var_list = [ self.ner_embedding_light ] )
                kbp_ner_embedding_train_step = tf.train.GradientDescentOptimizer(self.lr,
                                                                                use_locking = True) \
                                            .minimize(self.kbp_xent, var_list = [self.ner_embedding_kbp])

                self.rich_train_step.append( rich_ner_embedding_train_step )
                self.light_train_step.append( light_ner_embedding_train_step )
                self.kbp_train_step.append(kbp_ner_embedding_train_step)

            if feature_choice & (1 << 9) > 0:
                rich_char_conv_train_step = tf.train.MomentumOptimizer( self.lr, momentum )\
                                             .minimize( self.rich_xent, 
                                                var_list = [ self.conv_embedding ] + \
                                                             self.kernels + self.kernel_bias )
                light_char_conv_train_step = tf.train.MomentumOptimizer( self.lr, momentum )\
                                             .minimize( self.light_xent, 
                                                var_list = [ self.conv_embedding ] + \
                                                             self.kernels + self.kernel_bias )
                kbp_char_conv_train_step = tf.train.MomentumOptimizer(self.lr, momentum) \
                                            .minimize(self.kbp_xent, var_list = [self.conv_embedding] + \
                                                self.kernels + self.kernel_bias)

                self.rich_train_step.append( rich_char_conv_train_step )
                self.light_train_step.append( light_char_conv_train_step )
                self.kbp_train_step.append(kbp_char_conv_train_step)

            if feature_choice & (1 << 10) > 0:
                rich_bigram_train_step = tf.train.GradientDescentOptimizer( self.lr / 2, use_locking = True )\
                                            .minimize( self.rich_xent, var_list = [ self.bigram_embedding ] )
                light_bigram_train_step = tf.train.GradientDescentOptimizer( self.lr / 2, use_locking = True )\
                                            .minimize( self.light_xent, var_list = [ self.bigram_embedding ] )
                kbp_bigram_train_step = tf.train.GradientDescentOptimizer(self.lr / 2, use_locking = True) \
                                            .minimize(self.kbp_xent, var_list = [self.bigram_embedding])
                self.rich_train_step.append( rich_bigram_train_step )
                self.light_train_step.append( light_bigram_train_step )
                self.kbp_train_step.append(kbp_bigram_train_step)

        logger.info( 'computational graph built\n' )

        with self.graph.as_default():
            self.session.run( tf.global_variables_initializer() )
            # self.session.run( tf.variables_initializer( self.param ) )
            self.saver = tf.train.Saver()


    def train( self, mini_batch, curr_task, profile = False ):
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

        if curr_task.batch_num == 0:
            train = self.rich_train_step + [self.rich_xent]
            ner_cls_match_rich = dense_feature[:,512:]
            ner_cls_match_light = numpy.zeros((512, LIGHT_N_LABELS + 1))
            ner_cls_match_kbp = numpy.zeros((512, KBP_N_LABELS + 1))
        elif curr_task.batch_num == 1:
            train = self.light_train_step + [self.light_xent]
            ner_cls_match_rich = numpy.zeros((512, RICH_N_LABELS + 1))
            ner_cls_match_light = dense_feature[:,512:]
            ner_cls_match_kbp = numpy.zeros((512, KBP_N_LABELS + 1))
        else: 
            train = self.kbp_train_step + [self.kbp_xent]
            ner_cls_match_rich = numpy.zeros((512, RICH_N_LABELS + 1))
            ner_cls_match_light = numpy.zeros((512, LIGHT_N_LABELS + 1))
            ner_cls_match_kbp = dense_feature[:,512:]

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
                            self.ner_cls_match_rich: ner_cls_match_rich,
                            self.ner_cls_match_light: ner_cls_match_light,
                            self.ner_cls_match_kbp: ner_cls_match_kbp,
                            self.char_idx: conv_idx,
                            self.lbc_values : l5_values,
                            self.lbc_indices : l5_indices,
                            self.rbc_values : r5_values,
                            self.rbc_indices : r5_indices,
                            self.shape3 : (target.shape[0], 96 * 96),
                            self.label: target,
                            self.lr: curr_task.lr,
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
        

    def eval( self, mini_batch, curr_task ):
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

        if curr_task.batch_num == 0:
            train = [self.rich_xent, self.rich_predicted_indices, self.rich_predicted_values]
            ner_cls_match_rich = dense_feature[:,512:]
            ner_cls_match_light = numpy.zeros((512, LIGHT_N_LABELS + 1))
            ner_cls_match_kbp = numpy.zeros((512, KBP_N_LABELS + 1))
        elif curr_task.batch_num == 1:
            train = [self.light_xent, self.light_predicted_indices, self.light_predicted_values]
            ner_cls_match_rich = numpy.zeros((512, RICH_N_LABELS + 1))
            ner_cls_match_light = dense_feature[:,512:]
            ner_cls_match_kbp = numpy.zeros((512, KBP_N_LABELS + 1))
        else:
            train = [self.kbp_xent, self.kbp_predicted_indices, self.kbp_predicted_values]
            ner_cls_match_rich = numpy.zeros((512, RICH_N_LABELS + 1))
            ner_cls_match_light = numpy.zeros((512, LIGHT_N_LABELS + 1))
            ner_cls_match_kbp = dense_feature[:,512:]


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
                                                        self.ner_cls_match_rich: ner_cls_match_rich,
                                                        self.ner_cls_match_light: ner_cls_match_light,
                                                        self.ner_cls_match_kbp : ner_cls_match_kbp,
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

