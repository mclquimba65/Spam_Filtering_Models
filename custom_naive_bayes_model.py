# Model Built By Mark Clayton Quimba

import numpy as np
from collections import Counter

class Custom_Naive_Bayes_Model:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def build_model_nb(self):
        X_tokens_flat = [token for email_tokens in self.X for token in email_tokens]
        vocabulary = set(token for token in X_tokens_flat)
        vocabulary.add("<UNDEFINED>")

        spam_word_count = Counter()
        no_spam_word_count = Counter()

        for i in range(len(self.X)):
            email_tokenized = self.X[i]
            label = self.y[i]

            if label == 1:
                for x in range(len(email_tokenized)):
                    spam_word_count[email_tokenized[x]] += 1
            else:
                for x in range(len(email_tokenized)):
                    no_spam_word_count[email_tokenized[x]] += 1
        
        self.model = {
            "spam": spam_word_count,
            "no_spam": no_spam_word_count,
            "vocabulary": vocabulary
        }

        return self.model
    
    def X_given_y(self, email_tokenized, alpha=1):
        # Equation of P(X|Y) = (count(x_w | Y) + alpha) / (count(X | Y) + (alpha(Vocabulary)))
        email_tokenized_copy = [token if token in self.model['vocabulary'] else "<UNDEFINED>" for token in email_tokenized]
        Xgiveny = {}
        for cls in ("spam", "no_spam"):
            log_token_prob_summation = 0
            total_count = sum(self.model[cls].values())
            for token in email_tokenized_copy:
                token_count = self.model[cls].get(token, 0)
                token_prob = (token_count + alpha) / (total_count + (alpha * len(self.model['vocabulary'])))
                log_token_prob_summation += np.log(token_prob)
            Xgiveny[cls] = log_token_prob_summation

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
    
    def token_frequency_high_to_low(self):
        # For both classes, first create token list, then create count list
        tokens_high_to_low = {
            "spam": {
                "tokens": [],
                "count": [],
                "sorted": None
            },
            "no_spam": {
                "tokens": [],
                "count": [],
                "sorted": None
            }
        }

        for cls in ("spam", "no_spam"):
            for token, count in self.model[cls].items():
                tokens_high_to_low[cls]['tokens'].append(token)
                tokens_high_to_low[cls]['count'].append(count)

            sorted_indexes = np.argsort(tokens_high_to_low[cls]['count'])[::-1]
            tokens_high_to_low[cls]['sorted'] = [tokens_high_to_low[cls]['tokens'][i] for i in sorted_indexes]

        return tokens_high_to_low['spam']['sorted'], tokens_high_to_low['no_spam']['sorted']
    
    def token_frequency_high_to_low_ratio(self, alpha=1):
        tokens_high_to_low = {
            "spam": {
                "token_prob": None,
                "token_prob_ratio": [],
                "sorted": None
            },
            "no_spam": {
                "token_prob": None,
                "token_prob_ratio": [],
                "sorted": None
            }
        }

        vocab_size = len(self.model['vocabulary'])
        vocabulary_list = list(self.model['vocabulary'])
        
        for cls in ("spam", "no_spam"):
            total_count = sum(self.model[cls].values())
            tokens_high_to_low[cls]['token_prob'] = [(self.model[cls].get(token, 0) + alpha) / (total_count + (alpha * vocab_size)) for token in vocabulary_list]

        for i in range(len(vocabulary_list)):
            tokens_high_to_low['spam']['token_prob_ratio'].append((tokens_high_to_low['spam']['token_prob'][i] / tokens_high_to_low['no_spam']['token_prob'][i]))
            tokens_high_to_low['no_spam']['token_prob_ratio'].append((tokens_high_to_low['no_spam']['token_prob'][i] / tokens_high_to_low['spam']['token_prob'][i]))

        for cls in ("spam", "no_spam"):
            sorted_indexes = np.argsort(tokens_high_to_low[cls]['token_prob_ratio'])[::-1]
            tokens_high_to_low[cls]['sorted'] = [vocabulary_list[i] for i in sorted_indexes]

        return tokens_high_to_low['spam']['sorted'], tokens_high_to_low['no_spam']['sorted']