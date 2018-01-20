import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * log_l + p * logN
        Low = good
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

            L is the likelihood of the fitted model
            p is the number of parameters,
            N is the number of data points
        :return: GaussianHMM object
        """
        # implement model selection based on BIC scores
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        n = self.X.shape[0]
        best_bic = float('Inf')
        best_model = None
        for p in range(self.min_n_components, self.max_n_components):
            model = self.base_model(num_states=p)
            log_l = model.score(self.X, self.lengths)
            bic = -2 * log_l + p * np.log(n)
            if bic < best_bic:
                best_bic = bic
                best_model = model
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
        M: number of words
        log(P(X(i)): log score of self.this_word -> log_l
        log(P(X(all but i): log score of all words in self.word except .this_word -> log_l_all (all words)
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on DIC scores
        m = len(self.words)
        best_dic = float('-Inf') # Higher = Better
        best_model = None
        for p in range(self.min_n_components, self.max_n_components):
            model = self.base_model(num_states=p)
            log_l = model.score(self.X, self.lengths)
            # calculate all scores
            log_l_all = 0
            for word in self.words:
                word_x, word_length = self.hwords[word]
                log_l_all += model.score(word_x, word_length)
            dic = log_l - 1 / (m - 1) * (log_l_all - log_l)
            if dic > best_dic:
                best_dic = dic
                best_model = model
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection using CV
        best_cv = float('-Inf')
        best_model = None
        for p in range(self.min_n_components, self.max_n_components + 1):
            split_method = KFold(min(3, len(self.sequences)))
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                train_x, train_length = combine_sequences(cv_train_idx, self.sequences)
                test_x, test_length = combine_sequences(cv_test_idx, self.sequences)
                try:
                    model = self.base_model(num_states=p).fit(train_x, train_length)
                    cv = model.score(test_x, test_length)

                    if cv > best_cv:
                        best_cv = cv
                        best_model = model
                except:
                    continue
        return best_model
