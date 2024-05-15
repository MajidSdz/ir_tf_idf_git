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
