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


#In order to preprocess data we have a number of things we want to do:
# 1. remove punctuation
# 2. remove stop_words (common words)
# 3. stem - get the stem of words
# 4. lemmatize  

def get_labels_data():
    #Read from the csv file to create a pandas dataframe
    data = pd.read_csv('finalproject/code/data/yelp_reviews_Hotels_categories.csv')

    #get the labels - corresponds to number of stars 
    labels = data['review_stars'].values
    reviews = data['text']
    
    #get rid of all punctuation
    reviews = data['text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '' , x))
    
    #Determines how many reviews we want running through the model
    return labels[0:80000], reviews[0:80000]
    
#Function only for use if classifying as positive or negative
def binary_label(labels):

    labels_list= []
    for label in labels:
        if label >= 3:
            labels_list.append(0)
        else:
            labels_list.append(1)
    
    return np.array(labels_list)

#Function for classifying 3 classes: positive, neutral or negative
def ternary_label(labels):
    labels_list= []
    for label in labels:
        if label > 3:
            labels_list.append(0)
        if label == 3:
            labels_list.append(1)
        if label < 3:
            labels_list.append(2)
    return np.array(labels_list)

#For classifying all 5 star rating classes - sets them from 0-4 as this is what is used for loss
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
    #Get the stop words 
    stop_words = set(stopwords.words('english'))
    filtered_reviews = []

    for review in reviews:
        
        #Tokenize the input
        tokenized_review = word_tokenize(review)

        #Remove the stop words
        filtered_sentence = [w for w in tokenized_review if not w.lower() in stop_words]

        #Stem the sentence or lemmatize, this helps to stop 'trailing e' issue:
        ps = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        filtered = [lemmatizer.lemmatize(w) if lemmatizer.lemmatize(w).endswith('e') else ps.stem(w) for w in filtered_sentence]
        filtered_reviews.append(filtered)

    #Now we have removed all words we do not want we need to convert our reviews into sequence
    t = Tokenizer()
    #Updates internal vocabulary based on a list of texts. 
    t.fit_on_texts(filtered_reviews)
    #Converts reviews 
    sequenced_reviews = t.texts_to_sequences(filtered_reviews)
    
    #Finally we need to add padding to ensure all our reviews are the same length
    #Also set the max length of the review here, parameter that can be altered
    padded_reviews = pad_sequences(sequenced_reviews, maxlen=50) 
    
    return padded_reviews

def preprocess(classification=5):

    labels, reviews = get_labels_data()

    #Call if binary classification to convert labels
    if classification == 2:
        labels = binary_label(labels)

    #Call if ternary classification
    if classification == 3:
        labels = ternary_label(labels)

    #Call if full multi-class classification
    if classification == 5:
        labels = five_classes(labels)

    #process the reviews so they can ba parsed through the model
    reviews = process_text(reviews)
    
    #Split the reviews into train and test  
    train_inputs = reviews[:64000]
    test_inputs = reviews[64000:]

    #Split corresponding labels into train and test
    train_labels = labels[:64000]
    test_labels = labels[64000:]

    return train_inputs, test_inputs, train_labels, test_labels




