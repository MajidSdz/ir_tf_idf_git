# Libraries ->
import sys
import warnings
import random
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# import nltk
# nltk.download('stopwords')

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

# Read Data
content = pd.read_csv("dataset/Tweets/data.csv")
print("\n\n--------------------------------------------\n\n", end="")
print("Corpus Length(Original): ", len(content))
print(content.head())
print(content.tail())
print("\n\n--------------------------------------------\n\n", end="")

# First n first tweets
tweets_count = 1000
content = content[:tweets_count]
content = content.drop(
    ["tweet_id", "author_id", "inbound", "created_at", "response_tweet_id", "in_response_to_tweet_id"], axis=1)
print("\n\n--------------------------------------------\n\n", end="")
print("Corpus Length(N Tweets and Only Text Column): ", len(content))
print(content.head())
print(content.tail())
print("\n\n--------------------------------------------\n\n", end="")

# Preprocess Steps
# Lowercase
content["text"] = content["text"].apply(lambda x: x.lower())

# Stopword Removal
stop = stopwords.words('english')
content['text'] = content['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))

# Punctuations Removal
pattern = r'[^\w\s]'
content["text"] = content['text'].str.replace(pattern, '', regex=True)

# Numbers Removal
pattern = r'[\d]'
content["text"] = content['text'].str.replace(pattern, '', regex=True)

# More Than 1 Space Removal
pattern = r'\s{2,}'
content["text"] = content['text'].str.replace(pattern, ' ', regex=True)

print("\n\n--------------------------------------------\n\n", end="")
print("Corpus Length(After Preprocess): ", len(content))
print(content.head())
print(content.tail())
print("\n\n--------------------------------------------\n\n", end="")
