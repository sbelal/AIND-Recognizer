import warnings
from hmmlearn.hmm import GaussianHMM
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    test_x_lengths = test_set.get_all_Xlengths()

    test_set.num_items

    for word_id in range(len(test_x_lengths)):
        test_word_probabilities = {}
        best_guess_word = ""
        best_log_l = float('-inf')

        seq, lengths = test_set.get_item_Xlengths(word_id)

        for word, model in models.items():
            try:
                log_l = model.score(seq, lengths)
            except:
                log_l = float('-inf')

            test_word_probabilities[word] = log_l

            if log_l > best_log_l:
                best_log_l = log_l
                best_guess_word = word

        probabilities.append(test_word_probabilities)
        guesses.append(best_guess_word)

    return probabilities, guesses


def get_WER(guesses: list, test_set: SinglesData):
    """ Print WER

    :param guesses: list of test item answers, ordered
    :param test_set: SinglesData object
    :return:
        float of WER

    WER = (S+I+D)/N  but we have no insertions or deletions for isolated words so WER = S/N
    """
    S = 0
    N = len(test_set.wordlist)
    num_test_words = len(test_set.wordlist)
    if len(guesses) != num_test_words:
        print("Size of guesses must equal number of test words ({})!".format(num_test_words))
    for word_id in range(num_test_words):
        if guesses[word_id] != test_set.wordlist[word_id]:
            S += 1

    return float(S) / float(N)