# Model Built By Mark Clayton Quimba

from custom_markov_model import Custom_Markov_Model
from custom_naive_bayes_model import Custom_Naive_Bayes_Model

import pandas as pd
import nltk
nltk.download("punkt_tab")
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import re

data = pd.read_csv("C:/Users/Mark/Documents/MATH_452/Python_Code/Ling.csv")
print(data.head(10))    # Debug purposes

# Partly cleans the data from unnecessary string
data["body"] = data["body"].str.replace(r"content\s*-\s*length\s*:\s*\d+\s*", "", regex=True)
print(data["body"].head(10))    # Debug purposes
print(f"Number of rows: {data.shape[0]}\nNumber of columns: {data.shape[1]}")   # Debug purposes

# Splitting data into training set and testing set
# X_train contains contents of emails for the training set, y_train contains labels for training set
# X_test contains contents of emails for the testing set, y_train contains labels for testing set
X = data["body"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=180, shuffle=True)

print(X_train.head())   # Debug purposes
print("\n")

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
    # Clean up vocabulary
    clean_text = clean_vocab(text)
    tokens = word_tokenize(clean_text)
    tokens = [word.lower() for word in tokens]

    # Remove unnecessary indeterministic single punctuation
    redact_punct = {",", "."}
    tokens = [word for word in tokens if word not in redact_punct]

    return tokens


# Custom_Markov_Model requires X and y as lists
X_train_reformat = [tokenize(X_train.iloc[i]) for i in range(len(X_train))]
y_train_reformat = [y_train.iloc[i] for i in range(len(y_train))]

X_test_reformat = [tokenize(X_test.iloc[i]) for i in range(len(X_test))]
y_test_reformat = [y_test.iloc[i] for i in range(len(y_test))]

sum_spam = sum(y)
sum_ham = len(y) - sum_spam

sum_spam_train = sum(y_train_reformat)
sum_ham_train = len(y_train_reformat) - sum_spam_train

sum_spam_test = sum(y_test_reformat)
sum_ham_test = len(y_test_reformat) - sum_spam_test

# Building the Markov model
markov = Custom_Markov_Model(X_train_reformat, y_train_reformat)
model_markov = markov.build_model_markov()

# Classifying emails and retrieving classification error of training data
yhat_markov_train = markov.yhat_classify(X_train_reformat)
markov_model_error_train = markov.classification_error(yhat_markov_train, y_train_reformat)

# Classifying emails and retrieving classification error of testing data
yhat_markov_test = markov.yhat_classify(X_test_reformat)
markov_model_error_test = markov.classification_error(yhat_markov_test, y_test_reformat)

# Retrieving most common bigrams for both spam and no spam
top_bigrams_spam, top_bigrams_no_spam = markov.bigrams_high_to_low()
top_10_bigrams_spam = top_bigrams_spam[:10]
top_10_bigrams_no_spam = top_bigrams_no_spam[:10]

# Retrieving relative most common bigrams for both spam and no spam
top_bigrams_relative_spam, top_bigrams_relative_no_spam = markov.bigrams_high_to_low_ratio()
top_10_bigrams_relative_spam = top_bigrams_relative_spam[:10]
top_10_bigrams_relative_no_spam = top_bigrams_relative_no_spam[:10]

print("- Markov Model Errors -")
print(f"Markov Train Error: {markov_model_error_train}")
print(f"Markov Test Error: {markov_model_error_test}\n")

print("Top 10 bigrams for Spam emails:")
for current_word, next_word in top_10_bigrams_spam:
    print(current_word, next_word)

print("\n")

print("Top 10 bigrams for Non Spam emails:")
for current_word, next_word in top_10_bigrams_no_spam:
    print(current_word, next_word)

print("\n")

print("Top 10 bigrams relative for Spam emails:")
for current_word, next_word in top_10_bigrams_relative_spam:
    print(current_word, next_word)

print("\n")

print("Top 10 bigrams relative for Non Spam emails:")
for current_word, next_word in top_10_bigrams_relative_no_spam:
    print(current_word, next_word)

print("\n\n")

# Building the Naive Bayes model
naive_bayes = Custom_Naive_Bayes_Model(X_train_reformat, y_train_reformat)
nb_model = naive_bayes.build_model_nb()

# Classifying emails and retrieving classification error of training data
yhat_nb_train = naive_bayes.yhat_classify(X_train_reformat)
nb_model_error_train = naive_bayes.classification_error(yhat_nb_train, y_train_reformat)

# Classifying emails and retrieving classification error of testing data
yhat_nb_test = naive_bayes.yhat_classify(X_test_reformat)
nb_model_error_test = naive_bayes.classification_error(yhat_nb_test, y_test_reformat)

# Retrieving most common tokens for both spam and no spam
top_tokens_spam, top_tokens_no_spam = naive_bayes.token_frequency_high_to_low()
top_10_tokens_spam = top_tokens_spam[:10]
top_10_tokens_no_spam = top_tokens_no_spam[:10]

# Retrieving relative common tokens for both spam and no spam
top_tokens_relative_spam, top_tokens_relative_no_spam = naive_bayes.token_frequency_high_to_low_ratio()
top_10_tokens_relative_spam = top_tokens_relative_spam[:10]
top_10_tokens_relative_no_spam = top_tokens_relative_no_spam[:10]

print("- Naive Bayes Model Errors -")
print(f"Naive Bayes Train Error: {nb_model_error_train}")
print(f"Naive Bayes Test Error: {nb_model_error_test}\n")

print("Top 10 tokens for Spam emails:")
for token in top_10_tokens_spam:
    print(token)

print("\n")

print("Top 10 tokens for Non Spam emails:")
for token in top_10_tokens_no_spam:
    print(token)

print("\n")

print("Top 10 tokens relative for Spam emails:")
for token in top_10_tokens_relative_spam:
    print(token)

print("\n")

print("Top 10 tokens relative for Non Spam emails:")
for token in top_10_tokens_relative_no_spam:
    print(token)

print(len(model_markov['vocabulary']))

print("\n")

print(sum_spam)
print(sum_ham)

print("\n")

print(sum_spam_train)
print(sum_ham_train)

print("\n")

print(sum_spam_test)
print(sum_ham_test)