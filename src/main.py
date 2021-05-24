
import streamlit as st
from PIL import Image

from manage_data import make_prediction_text
from GPT2 import generatefakes


st.write('''# Detecting Fake News''')

fake_news = Image.open('FAKE NEWS ALERT.jpg').convert('RGB').save('new.jpeg')
images = Image.open('new.jpeg')
st.image(images)

st.write('''

The aim of this project was to explore machine-learning algorithms that pertain to the creation and classification of Fake News. 

For the creation of Fake News I explored GPT-2, a large scale unsupervised language model that generates coherent (or semi-coherent) text and achieves high level performance in many language modeling benchmarks.
Although most "fake-news' or heavily biased articles seem to be written by actual humans, it is not hard to imagine a world where someone could use this "neural fake-news" technology to try to influence public opinion regarding sensitive topics from geo-political disputes to manipulation of public opinion on social networks like reddit and twitter.

On the classification side, I attempted to create a neural network capable of decomposing the article text, process it 
For this project, we gathered over 78,000 news articles distributed around 50/50 between Real and Fake. 

We proceeded to feed the training dataset with some of our algorithmically created fake news, to try to improve its accuracy when confronted witht these neural fake news.

''')


st.write('''# What is GPT-2?

GPT2 is a unsupervised learning model that was trained on 40gb of text from the internet. It's main learning goal was simple : predict the next word, given all of the previous words within some text.
GPT2 creates synthetic text examples in response to the model being given an arbitrary input, the model is very chameleon like, as it responds to the style and content of the text its been given.
The model is far from perfect, it sometimes will switch from topic to topic and seems to perform better with topics its been trained on as opposedd to very esoterical topics.

Although its far from perfect, the team responsible for creating it felt scared about its potential for missuse and only released the smaller versions of the model. As this technology advances, it has high potential for missuse, from generating fake news to generating fake personalities on social media.

''')

st.write('''# Fake News Creator: Using GPT-2 To Write Fake News Text

I finetuned the GPT-2 Model with a bunch of news text (both fake and real) in order to try and have a bit more control on what the output text would be. 

I was able to create a model capable of creating semi-coherent news articles out of an initial sentence. 

Wanna try it out? Input some parameters and lets generate some fake news.
''')

article_length = st.number_input('How long should the article be? (int)', min_value = 100)
article_creativity = st.number_input('How creative should the article be? input a number between 0 and 1, 1 being more creative', min_value = 0.1)
article_sentence = st.text_input('Input a sentence the algorithm should be created around.')
article_number = st.number_input('How many articles should I write?', min_value = 1)


if st.button("Create Neural Fake Text"):
    generate = generatefakes(article_length, article_creativity, article_sentence, article_number)
    st.write(f'{generate}')


st.write('''
# Creating a Fake News Classifier:

For this project, I tried to create a recurent convulutional neural network model that took into account the body of text from an article and spet out a clasiffication of either Real (0) or Fake (1).

The pillars of this model are as follows:

1. Tokenizer = We created a tokenizer that read through all the words in all the articles in our data and transformed each article into a vector of indexed words.
2. Word Embedding = The first layer in our LSTM model. This layer seeks to create a vector of M dimensions for each word, it tries to represent the meaning of words for each word in relation to other words present in the article. Words with similar embedding vectors tend to have similar meanings and represent similar concepts.
3. Convolutional 1D = A 1-dimensional filter that is passed through our word embedding matrixes, and outputs a new vector based on the filter. 
4. Maxpool = operates on vectors but reduces the size of the input by selecting the maximum values from local regions of the input.
5. LSTM Layers = LSTM stands for Long Short Term Memory. This a network archetype that is capable of remembering and forgeting information when its no longer needed. This is usefull for the task of classifying text, as it is able to play with different weights and optimize for the best result.
6. Dense = The fully connected layer, it chages the vectors outputed by the rest of the model into numbers from 0 to 1, 1 being classified as Fake.
''')




st.write('''# Fake News Analyzer
We created a LSTM model that analyzes the text in the article makes a prediction of wether a text is real or fake... Try it out!
''')

user_input = st.text_area("Article Text Goes Here:")

if st.button("Predict"):
    if user_input:
        make_prediction_text(user_input)
    else:
        st.error('You should enter an article text body')
