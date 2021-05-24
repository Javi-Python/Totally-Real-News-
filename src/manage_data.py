from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import regex as re
import pandas as pd
from config import n_tokens, keep_n, embedding_dim
import pandas as pd
import numpy as np
#This class allows to vectorize a text corpus, by turning each text into either a sequence of integers (each integer being the index of a token in a dictionary) 
#or into a vector where the coefficient for each token 
from tensorflow.keras.preprocessing.text import Tokenizer

#Converts a text to a sequence of words (or tokens).
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

#lets us create embedding of words that represent the meaning of the words in relation to other words.
from keras_preprocessing.text import  tokenizer_from_json

import json
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st


with open('../tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)


path_m = '../Keras_Model/LSTM_92%ACC.h5'
model = load_model(path_m)


def text_preprocessor(text):
    #Processes text and prepares it for encoding
    nlp = spacy.load('en_core_web_md')
    text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    doc = nlp(text)
    #Lemmatization, which is the process of reducing a word to its lemma or dictionary form. 
    #For example, the word run is the lemma for the words runs, ran, and running.
    text = ' '.join([token.lemma_ for token in doc if token.text not in STOP_WORDS])
    return text

def import_text(text):
    #to import a news article text in order to attempt classification
    df = pd.DataFrame()
    df['text'] = [text]
    df['text'] = df.text.apply(text_preprocessor)
    return df


def text_to_sequences(df):
    #changes text to a sequence of numbers based on tokens
    sequence = tokenizer.texts_to_sequences(df.text)
    return sequence

def sec_to_padded(sequence):
    #Addds padding and enables same shape matrixes
    padded = pad_sequences(sequence, maxlen = keep_n, padding = 'post')
    return padded

def make_prediction(padded):
    #Makes prediction based on the loaded model
    model = load_model(path_m)
    model._make_predict_function()
    graph = tf.get_default_graph()
    with graph.as_default():
        prediction = model.predict(padded)
        if prediction > .65:
            st.write('# FAKE NEWS')
            st.write(f'probability of fake news {prediction}')
        else:
            st.write('# REAL')
            st.write(f'probability of fake news {prediction}')
        
    
def make_prediction_text(text):
    df = import_text(text)
    sequence = text_to_sequences(df)
    padded = sec_to_padded(sequence)
    
    return make_prediction(padded)




