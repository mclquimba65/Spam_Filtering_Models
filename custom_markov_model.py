# Model Built By Mark Clayton Quimba

import numpy as np
from collections import defaultdict, Counter

class Custom_Markov_Model:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = None

    def build_model_markov(self):
        X_tokens_flat = [token for email_tokens in self.X for token in email_tokens]
        vocabulary = set(token for token in X_tokens_flat)
        vocabulary.add("<UNDEFINED>")

        spam_bigram_count = defaultdict(Counter)
        no_spam_bigram_count = defaultdict(Counter)

        spam_first_count = Counter()
        no_spam_first_count = Counter()

        # Add comment
        for i in range(len(self.X)):
            email_tokenized = self.X[i]
            label = self.y[i]

            if label == 1:
                spam_first_count[email_tokenized[0]] += 1
                for x in range(len(email_tokenized) - 1):
                    spam_bigram_count[email_tokenized[x]][email_tokenized[x + 1]] += 1
            else:
                no_spam_first_count[email_tokenized[0]] += 1
                for x in range(len(email_tokenized) - 1):
                    no_spam_bigram_count[email_tokenized[x]][email_tokenized[x + 1]] += 1

        self.model = {
            "spam": {
                "bigram": spam_bigram_count,
                "first": spam_first_count
            },
            "no_spam": {
                "bigram": no_spam_bigram_count,
                "first": no_spam_first_count
            },
            "vocabulary": vocabulary
        }

        return self.model
    
    def X_given_y(self, email_tokenized, alpha=1):
        # Equation of P(X_0=w_0 | Y=y) = (count(w_0,Y) + alpha) / (count(X,Y) + alpha(Vocabulary))
        # Equation of P(X_(n+1)=w_(n+1) | X_n=w_n,Y=y) = (count(w_n -> w_(n+1), Y) + alpha) / (count(X_n -> X_(n+1), Y) + alpha(Vocabulary))
        email_tokenized_copy = [token if token in self.model['vocabulary'] else "<UNDEFINED>" for token in email_tokenized]
        first_word = email_tokenized_copy[0]
        Xgiveny = {}

        for cls in ("spam", "no_spam"):
            log_bigram_prob_summation = 0
            
            # Finding probability of the first word
            # First, finding total count of all first words for both classes
            # Then, finding frequency of email's first word
            # Calculate probability of first word with applied laplace smoothing of alpha=1 by default
            # Take the log of this probability
            first_word_total = sum(self.model[cls]['first'].values())
            first_word_count = self.model[cls]['first'].get(first_word, 0)
            first_word_prob = (first_word_count + alpha) / (first_word_total + (alpha * len(self.model['vocabulary'])))
            log_first_word_prob = np.log(first_word_prob)

            # Finding probability of the next word given a word
            # First, retrieve the current word and the next word
            # Find the total count of the transition from the current word to the next word
            # Find frequency of the transition from the current word to the next word
            # Calculate probability of the transition with applied laplace smoothing of alpha=1 by default
            # Take the log of this probability
            # Add to log summation of next given current
            for i in range(len(email_tokenized_copy) - 1):
                current_word = email_tokenized_copy[i]
                next_word = email_tokenized_copy[i + 1]
                bigram_total = sum(self.model[cls]['bigram'].get(current_word, {}).values())
                bigram_count = self.model[cls]['bigram'].get(current_word, {}).get(next_word, 0)
                bigram_prob = (bigram_count + alpha) / (bigram_total + (alpha * len(self.model['vocabulary'])))
                log_bigram_prob_summation += np.log(bigram_prob)

            Xgiveny[cls] = log_first_word_prob + log_bigram_prob_summation

        return Xgiveny['spam'], Xgiveny['no_spam']
    
    def y_prior(self):
        # Calculates prior probability of both spam and not spam
        # Labels are either 1 or 0 and so summing up all would be count of spam
        y_prior_spam = sum(self.y) / len(self.y)
        y_prior_no_spam = 1 - y_prior_spam

        log_y_prior_spam = np.log(y_prior_spam)
        log_y_prior_no_spam = np.log(y_prior_no_spam)

        return log_y_prior_spam, log_y_prior_no_spam
    
    def yhat_classify(self, X):
        # For each email use yhat equation and predict
        yhat_vector = []

        log_spam_y_prior, log_no_spam_y_prior = self.y_prior()

        for i in range(len(X)):
            log_spam_Xgiveny, log_no_spam_Xgiveny = self.X_given_y(X[i])

            yhat_spam = log_spam_y_prior + log_spam_Xgiveny
            yhat_no_spam = log_no_spam_y_prior + log_no_spam_Xgiveny

            yhat_vector.append(1 if yhat_spam > yhat_no_spam else 0)

        return yhat_vector

    def classification_error(self, yhat, ytruth):
        error = 0
        for i in range(len(yhat)):
            if yhat[i] != ytruth[i]:
                error += 1

        error_prob = error / len(yhat)

        return error_prob
    
    def bigrams_high_to_low(self):
        bigram_count = {
            "spam": {
                "bigram": [],
                "count": [],
                "sorted": None
            },
            "no_spam": {
                "bigram": [],
                "count": [],
                "sorted": None
            }
        }

        for cls in ("spam", "no_spam"):
            for current_word, counter in self.model[cls]['bigram'].items():
                for next_word, count in counter.items():
                    bigram_count[cls]['bigram'].append((current_word, next_word))
                    bigram_count[cls]['count'].append(count)

            sorted_indexes = np.argsort(bigram_count[cls]['count'])[::-1]
            bigram_count[cls]['sorted'] = [bigram_count[cls]['bigram'][i] for i in sorted_indexes]

        return bigram_count['spam']['sorted'], bigram_count['no_spam']['sorted']
    
    def bigrams_high_to_low_ratio(self, alpha=1):
        bigrams_high_to_low = {
            "spam": {
                "bigram_prob": [],
                "bigram_ratio": [],
                "sorted": None
            },
            "no_spam": {
                "bigram_prob": [],
                "bigram_ratio": [],
                "sorted": None
            }
        }

        vocab_size = len(self.model['vocabulary'])
        observed_bigrams_set = set()

        for cls in ("spam", "no_spam"):
            for current_word, counter in self.model[cls]['bigram'].items():
                for next_word in counter:
                    observed_bigrams_set.add((current_word, next_word))

        observed_bigrams = list(observed_bigrams_set)

        for cls in ("spam", "no_spam"):
            for current_word, next_word in observed_bigrams:
                bigram_count = self.model[cls]['bigram'].get(current_word, {}).get(next_word, 0)
                bigram_total = sum(self.model[cls]['bigram'].get(current_word, {}).values())
                bigram_prob = (bigram_count + alpha) / (bigram_total + (alpha * vocab_size))

                bigrams_high_to_low[cls]['bigram_prob'].append(bigram_prob)

        for i in range(len(observed_bigrams)):
            bigrams_high_to_low['spam']['bigram_ratio'].append((bigrams_high_to_low['spam']['bigram_prob'][i]) / (bigrams_high_to_low['no_spam']['bigram_prob'][i]))
            bigrams_high_to_low['no_spam']['bigram_ratio'].append((bigrams_high_to_low['no_spam']['bigram_prob'][i]) / (bigrams_high_to_low['spam']['bigram_prob'][i]))

        for cls in ("spam", "no_spam"):
            sorted_indexes = np.argsort(bigrams_high_to_low[cls]['bigram_ratio'])[::-1]
            bigrams_high_to_low[cls]['sorted'] = [observed_bigrams[i] for i in sorted_indexes]

        return bigrams_high_to_low['spam']['sorted'], bigrams_high_to_low['no_spam']['sorted']