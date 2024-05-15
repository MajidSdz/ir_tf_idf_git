# Libraries ->
import sys
import warnings
import random
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Configurations ->
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

# Set random seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)

# to ignore warnings
warnings.filterwarnings('ignore')

# Function to separate first n words
def get_first_n_words(text, n=1000):
    words = text.split()
    first_n_words = words[:n]
    return " ".join(first_n_words)


# Get words count
def get_words_count(text):
    words = text.split()
    return len(words)

# Read Data
file = open('dataset/SciFi/data.txt', 'r')
content = file.read()
file.close()
print("\n\n--------------------------------------------\n\n", end="")
print("Corpus Length(Original): ", len(content), "chars /", str(get_words_count(content)), "words", end="")
print("\n\n--------------------------------------------\n\n", end="")

# First n words for new content
n_words = 1000
content = get_first_n_words(content, n_words)
print("\n\n--------------------------------------------\n\n", end="")
print("Corpus Length(N Words): ", len(content), "chars /", str(get_words_count(content)), "words", end="")
print("\n\n--------------------------------------------\n\n", end="")

# Preprocess Steps
# Lowercase
content = content.lower()

# Punctuations Removal
pattern = r'[^\w\s]'
content = re.sub(pattern, "", content)

# Numbers Removal
pattern = r'[\d]'
content = re.sub(pattern, "", content)

# More Than 1 Space Removal
pattern = r'\s{2,}'
content = re.sub(pattern, " ", content)

print("\n\n--------------------------------------------\n\n", end="")
print("Corpus Length(After Preprocess): ", len(content), "chars /", str(get_words_count(content)), "words", end="")
print("\n\n--------------------------------------------\n\n", end="")
