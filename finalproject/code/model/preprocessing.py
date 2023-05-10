from concurrent.futures import process
from sre_parse import Tokenizer
import pandas as pd
import numpy as np
import pickle
import nltk
import re
import string
import torch
import tensorflow
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from torch.nn.utils.rnn import pad_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# Here, we want to remove any punctuation as well as any common words before
# retrieving the stem of each word and lemmatizing the words.

def get_labels_data():
    #Read from the csv file to create a pandas dataframe
    data = pd.read_csv('finalproject/code/data/yelp_reviews_Hotels_categories.csv')

    #Retrieving each review's text and consequent star rating
    reviews = data['text']
    labels = data['review_stars'].values
    
    #removing punctuation
    reviews = data['text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))
    
    #We are using 80,000 reviews from the dataset
    return labels[0:80000], reviews[0:80000]
    
# Binary classification function
def binary_label(labels):

    labels_list= []
    for label in labels:
        if label >= 3:
            labels_list.append(0)
        else:
            labels_list.append(1)
    
    return np.array(labels_list)

#For 5-way classification or star rating
def five_classes(labels):
    labels_list= []
    for label in labels:
        if label == 5:
            labels_list.append(0)
        if label == 4:
            labels_list.append(1)
        if label == 3:
            labels_list.append(2)
        if label == 2:
            labels_list.append(3)
        if label == 1:
            labels_list.append(4)
    
    return np.array(labels_list)

def process_text(reviews):
    #Get the common or unnecessary words 
    stop_words = set(stopwords.words('english'))
    filtered_reviews = []

    for review in reviews:
        
        tokenized_review = word_tokenize(review)

        #Remove the stop words
        filtered_sentence = [w for w in tokenized_review if not w.lower() in stop_words]

        #Here, we lemmatize
        ps = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        filtered = [lemmatizer.lemmatize(w) if lemmatizer.lemmatize(w).endswith('e') else ps.stem(w) for w in filtered_sentence]
        filtered_reviews.append(filtered)

    t = Tokenizer()
    t.fit_on_texts(filtered_reviews)
    sequenced_reviews = t.texts_to_sequences(filtered_reviews)
    
    #Set max length of review and make sure all reviews are same length
    padded_reviews = pad_sequences(sequenced_reviews, maxlen=50) 
    
    return padded_reviews

def preprocess(classification=5):

    labels, reviews = get_labels_data()

    #Binary classification
    if classification == 2:
        labels = binary_label(labels)
    elif classification == 5:
    #5-way classification for star rating
        labels = five_classes(labels)

    reviews = process_text(reviews)
    
    train_inputs = reviews[:64000]
    test_inputs = reviews[64000:]
    train_labels = labels[:64000]
    test_labels = labels[64000:]

    return train_inputs, test_inputs, train_labels, test_labels




