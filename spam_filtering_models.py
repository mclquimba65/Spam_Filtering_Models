from custom_markov_model import Custom_Markov_Model

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100, shuffle=True)

print(X_train.head())   # Debug purposes

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

