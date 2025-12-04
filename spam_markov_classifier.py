# Might import cudf.pandas if calculations take too long
import numpy as np
import pandas as pd     # Dataframes
import nltk             # Tokenizer
nltk.download("punkt_tab")
nltk.download("stopwords")
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import re               # This is module for regular expressions

# Reads csv file and assign data to pandas dataframe
data = pd.read_csv("C:/Users/Mark/Documents/MATH_452/Python_Code/Ling.csv")
print(data.head(10))

# Cleans the data from unneccesary words
data["body"] = data["body"].str.replace(r"content\s*-\s*length\s*:\s*\d+\s*", "", regex=True)
print(data["body"].head(10))
print(f"Number of rows: {data.shape[0]}\nNumber of columns: {data.shape[1]}")

# Splitting data into a training set and a testing set
# X_train contains content of emails for training set, y_train contains labels for training set
# X_test contains content of emails for testing set, y_test contains labels for testing set
X = data["body"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100, shuffle=True)

print(X_train.head())

# Create a vocabulary that redacts any rare words
# Create nested dictionaries for both spam and ham that contain a word paired with observed next words, use defaultdict and Counter from collections
# Create first word dictionaries using Counter
# For each email, tally up next words given current word frequencies, also keep track of first words

def clean_vocab(text):
    # Replace repeating symbols/punctuation with PRESET tokens
    text = re.sub(r'(\-\s*\-)+', ' DASHPLUS ', text)
    text = re.sub(r'(\*\s\*)+', ' STARPLUS ', text)
    text = re.sub(r'(=\s*=)+', ' EQUALSPLUS ', text)
    text = re.sub(r'(/\s*/)+', ' SLASHPLUS ', text)
    text = re.sub(r'(\+\s*\+)+', ' PLUSPLUS ', text)
    text = re.sub(r'(~\s*~)+', ' TILDEPLUS ', text)
    text = re.sub(r'(_\s*_)+', ' UNDERSCORE ', text)

    # Match multiple PRESET tokens in a row to one token
    text = re.sub(r'(DASHPLUS\s*)+', ' DASHPLUS ', text)
    text = re.sub(r'(STARPLUS\s*)+', ' STARPLUS ', text)
    text = re.sub(r'(EQUALSPLUS\s*)+', ' EQUALSPLUS ', text)
    text = re.sub(r'(SLASHPLUS\s*)+', ' SLASHPLUS ', text)
    text = re.sub(r'(PLUSPLUS\s*)+', ' PLUSPLUS ', text)
    text = re.sub(r'(TILDEPLUS\s*)+', ' TILDEPLUS ', text)
    text = re.sub(r'(UNDERSCORE\s*)+', ' UNDERSCORE ', text)

    return text

def tokenize(text):
    # Note: Clean up vocabulary
    clean_text = clean_vocab(text)
    tokens = word_tokenize(clean_text)
    tokens = [word.lower() for word in tokens]

    # Remove unnecessary indeterministic single punctuation
    redact_punct = {",", "."}
    tokens = [word for word in tokens if word not in redact_punct]

    return tokens

# This function builds a model that returns the vocabulary, spam bigrams, no spam bigrams, and first word dictionaries
def build_model_frequencies(X, y):
    # Flatten tokens and create dictionary containing word and their frequencies
    # Add <UNDEFINED>
    X_tokens_flat = [word for email in X for word in email]

    word_frequencies = Counter(X_tokens_flat)

    minimum_frequency = 1
    vocabulary = set(word for word, frequency in word_frequencies.items() if frequency >= minimum_frequency)
    vocabulary.add("<UNDEFINED>")

    spam_bigram = defaultdict(Counter)
    no_spam_bigram = defaultdict(Counter)

    spam_first = Counter()
    no_spam_first = Counter()

    # From here, build the bigrams and first words, building transitions and distributions
    # Traverse through each email changing words not in vocabulary to undefined
    # Then count current word and next word transitions 
    for i in range(len(X)):
        email = X[i]
        email = [word if word in vocabulary else '<UNDEFINED>' for word in email]
        label = y.iloc[i]

        if label == 1:
            spam_first[email[0]] += 1
            for x in range(len(email) - 1):
                spam_bigram[email[x]][email[x + 1]] += 1
        else:
            no_spam_first[email[0]] += 1
            for x in range(len(email) - 1):
                no_spam_bigram[email[x]][email[x + 1]] += 1

    model_frequencies = {
        "spam": {
            "bigram": spam_bigram,
            "first": spam_first
        },
        "no_spam": {
            "bigram": no_spam_bigram,
            "first": no_spam_first
        }
    }
    
    return vocabulary, model_frequencies

def Xgiveny_markov(email, model_frequencies, vocabulary, alpha=1):
    # Equation of P(X_0=w_0 | Y=y) = (count(w_0,Y) + alpha) / (count(X,Y) + alpha(Vocabulary))
    # Equation of P(X_(n+1)=w_(n+1) | X_n=w_n,Y=y) = (count(w_n -> w_(n+1), Y) + alpha) / (count(X_n -> X_(n+1), Y) + alpha(Vocabulary))
    email_tokenized = [word if word in vocabulary else '<UNDEFINED>' for word in email]
    first_word = email_tokenized[0]
    Xgiveny = {}
    
    for cls in ("spam", "no_spam"):
        log_transition_prob_summation = 0

        # Finding probability of the first word
        # First, finding total count of first word instances of both classes
        # Then, finding frequency of the email's first word
        # Calculate probability of first word with applied laplacian smoothing of alpha=1
        # Take the log of this probability
        first_word_total = sum(model_frequencies[cls]['first'].values())
        first_word_count = model_frequencies[cls]['first'].get(first_word, 0)
        first_word_prob = (first_word_count + alpha) / (first_word_total + (alpha * len(vocabulary)))
        log_first_word_prob = np.log(first_word_prob)

        # Finding probability of the next word given a word
        # First, retrieve the current word and the next word
        # Find total count of the transition from current word to next word
        # Find frequency of the transition from current word to next word in the email
        # Calculate probabilty of the transition with applied laplacian smoothing of alpha=1
        # Take the log of this probability
        for i in range(len(email_tokenized) - 1):
            current_word = email_tokenized[i]
            next_word = email_tokenized[i + 1]
            transition_total = sum(model_frequencies[cls]['bigram'][current_word].values())
            transition_count = model_frequencies[cls]['bigram'][current_word].get(next_word, 0)
            transition_prob = (transition_count + alpha) / (transition_total + (alpha * len(vocabulary)))
            log_transition_prob_summation += np.log(transition_prob)

        Xgiveny[cls] = log_first_word_prob + log_transition_prob_summation

    return Xgiveny["spam"], Xgiveny["no_spam"]

def y_prior_markov(y):
    # Calculates prior probability of both spam and not spam
    # Labels are either 1 or 0 and so summing up all would be count of spam
    y_prior_spam = sum(y) / len(y)
    y_prior_no_spam = 1 - y_prior_spam

    log_y_prior_spam = np.log(y_prior_spam)
    log_y_prior_no_spam = np.log(y_prior_no_spam)

    return log_y_prior_spam, log_y_prior_no_spam

def yhat_classifier_markov(X_tokens, y, vocabulary, model_frequencies):
    yhat_vector = []

    log_spam_y_prior, log_no_spam_y_prior = y_prior_markov(y)

    for i in range(len(X_tokens)):
        log_spam_Xgiveny, log_no_spam_Xgiveny = Xgiveny_markov(X_tokens[i], model_frequencies, vocabulary)

        yhat_spam = log_spam_y_prior + log_spam_Xgiveny
        yhat_no_spam = log_no_spam_y_prior + log_no_spam_Xgiveny

        yhat_vector.append(1 if yhat_spam > yhat_no_spam else 0)

    return yhat_vector

def classification_error(yhat, y):
    error = 0
    for i in range(len(yhat)):
        if yhat[i] != y.iloc[i]:
            error += 1

    error_probability = error / len(yhat)
    return error_probability

def transitions_high_to_low(bigram):
    transitions = []
    counts = []

    for current_word, counter in bigram.items():
        for next_word, count in counter.items():
            transitions.append((current_word, next_word))
            counts.append(count)

    arr_counts = np.array(counts)
    sorted_indexes = np.argsort(arr_counts)[::-1]

    sorted_transitions = [transitions[i] for i in sorted_indexes]

    return sorted_transitions


X_train_tokens = [tokenize(X_train.iloc[i]) for i in range(len(X_train))]
train_vocabulary_markov, train_model_frequencies_markov = build_model_frequencies(X_train_tokens, y_train)
yhat_vector_train_markov = yhat_classifier_markov(X_train_tokens, y_train, train_vocabulary_markov, train_model_frequencies_markov)
error_probability_train_markov = classification_error(yhat_vector_train_markov, y_train)

X_test_tokens = [tokenize(X_test.iloc[i]) for i in range(len(X_test))]
yhat_vector_test_markov = yhat_classifier_markov(X_test_tokens, y_test, train_vocabulary_markov, train_model_frequencies_markov)
error_probability_test_markov = classification_error(yhat_vector_test_markov, y_test)

print(f"Train error: {error_probability_train_markov}")
print(f"Test error: {error_probability_test_markov}")

transitions_spam = transitions_high_to_low(train_model_frequencies_markov["spam"]["bigram"])
transitions_no_spam = transitions_high_to_low(train_model_frequencies_markov["no_spam"]["bigram"])

top_10_spam_bigrams = transitions_spam[:10]
top_10_no_spam_bigrams = transitions_no_spam[:10]

for current_word, next_word in top_10_spam_bigrams:
    print(current_word, next_word)

print("\n")

for current_word, next_word in top_10_no_spam_bigrams:
    print(current_word, next_word)