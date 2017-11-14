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
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("inf")
        best_model = None

        for num_components in range(self.min_n_components, self.max_n_components + 1):

            try:
                hmm_model = self.base_model(num_components)
                log_l = hmm_model.score(self.X, self.lengths)
            except:
                if self.verbose:
                    print("Model train error on {} with {} states".format(self.this_word, num_components))
                hmm_model = None
                break  #No need to increase complexity of model when current model failed.

            num_features = self.X.shape[1]
            log_n = math.log(self.X.shape[0])
            num_params = num_components * (num_components - 1) + 2 * num_components * num_features
            score = -2 * log_l + num_params * log_n

            if best_score > score:
                best_score = score
                best_model = hmm_model

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''


    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):

        ModelSelector.__init__(self, all_word_sequences, all_word_Xlengths, this_word, n_constant, min_n_components, max_n_components,
                               random_state, verbose)  #Call base constructor
        self.preprocessed_models = {}  #Key is num components(int)  value is a dictionary of word and model_scores


    def get_scores_for_all_words(self, num_components):
        """ For each word get the score for HMM model
        
        Args:
        num_components (int): The number of states in a HMM model.

        Returns:
        A dictionary of tuples (GaussianHMM, float) where the key of the dictionary is word(str)        
        """

        #If the models were processed before for the given num_components then return that instead of recalculating again
        if num_components in self.preprocessed_models:            
            return self.preprocessed_models[num_components]

        model_scores = {}
        for word, _ in self.words.items():
            try:
                x_sequence, seq_lengths = self.hwords[word]
                hmm_model = GaussianHMM(n_components=num_components, covariance_type="diag",
                                        n_iter=1000, random_state=self.random_state,
                                        verbose=False).fit(x_sequence, seq_lengths)
                log_l = hmm_model.score(x_sequence, seq_lengths)
            except:
                if self.verbose:
                    print("failure on {} with {} states".format(word, num_components))                
                hmm_model = None

            if hmm_model is not None:
                model_scores[word] = (hmm_model, log_l)

        self.preprocessed_models[num_components] = model_scores
        return model_scores



    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score, best_model = float("-inf"), None

        for num_components in range(self.min_n_components, self.max_n_components + 1):
            if self.verbose:
                print("     Current model number of components: {}  Word: {}".format(num_components, self.this_word))
            all_model_scores = self.get_scores_for_all_words(num_components)
            if self.this_word not in all_model_scores:
                if self.verbose:
                    print("Model train error on {} with {} states".format(self.this_word, num_components))
                break

            othwer_words_scores = [model_score[1] for word, model_score in all_model_scores.items()
                                   if word != self.this_word]
            othwer_words_score_avg = np.average(othwer_words_scores)
            current_model, current_word_score = all_model_scores[self.this_word]

            score_dic = current_word_score - othwer_words_score_avg
            if self.verbose:
                print("     DIC Score: {}  othwer_words_score_avg: {}  models in dictionary: {}".format(score_dic,othwer_words_score_avg,len(all_model_scores)))
            if score_dic > best_score:
                best_model = current_model
                best_score = score_dic

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        kfold_splits = min(len(self.sequences), 3) #Use 3 split unless we have fewer sequences
        best_score = float("-inf")
        best_num_components = 3
        word_sequences = self.sequences

        if kfold_splits >= 2:            
            if kfold_splits < 3:
                if self.verbose:
                    print ("For {} using a kfold split of {}.".format(self.this_word, kfold_splits))
            split_method = KFold(random_state=self.random_state, n_splits=kfold_splits)
            fold_indices = list(split_method.split(word_sequences))
        else:
            if self.verbose:
                print("Sequences for {} is less than 2.  Creating model with {} states.".format(self.this_word, best_num_components))
            hmm_model = self.base_model(best_num_components)
            return hmm_model

        for num_components in range(self.min_n_components, self.max_n_components + 1):
            scores = []

            for cv_train_idx, cv_test_idx in fold_indices:
                train_x, train_x_lengths = combine_sequences(cv_train_idx, word_sequences)
                test_x, test_x_lengths = combine_sequences(cv_test_idx, word_sequences)
                try:
                    hmm_model = GaussianHMM(n_components=num_components, covariance_type="diag",
                                            n_iter=1000,
                                            random_state=self.random_state,
                                            verbose=False).fit(train_x, train_x_lengths)

                    log_l = hmm_model.score(test_x, test_x_lengths)
                    scores.append(log_l)
                except Exception as e:
                    if self.verbose:
                        print("Model train error on {} with {} states".format(self.this_word, num_components))
                    
                    #Discard this model.
                    hmm_model = None
                    break

            if hmm_model is None:
                # Stop increasing complexity since the current model failed.
                break

            if len(scores) == 1:
                avg = scores[0]
            else:
                avg = np.average(scores)

            if best_score < avg:
                best_score = avg                
                best_num_components = num_components

        #Train the model with the full set of data
        model = self.base_model(best_num_components)

        return model
