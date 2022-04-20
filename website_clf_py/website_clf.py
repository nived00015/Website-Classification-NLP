import pandas as pd
import sklearn.metrics as sm
import nltk
import string
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np


def clean_text(text):
    #### cleaning the text 
    ###1. Convert the text to lower case
    text= text.lower()

    ###2. tokenize the sentences to words
    text_list= word_tokenize(text)

    ###3. Removes the special charcters
    special_char_non_text= [re.sub(f'[{string.punctuation}]+','',i) for i in text_list]

    ###4.  remove stopwords
    non_stopwords_text= [i for i in special_char_non_text if i not in stopwords.words('english')]

    ###5. lemmatize the words
    lemmatizer= WordNetLemmatizer()
    lemmatized_words= [lemmatizer.lemmatize(i) for i in non_stopwords_text]

    cleaned_text= ' '.join(lemmatized_words)

    return cleaned_text


text_input= input('Please enter the text: ')
cleaned_text= clean_text(text_input)

#temp_df= pd.DataFrame({'input_text':[cleaned_text]})
vectorizer_filepath= 'tf_idf_vectorizer.pkl'
tf_idf_vectorizer= pickle.load(open(vectorizer_filepath,'rb'))
temp_df= tf_idf_vectorizer.transform([cleaned_text])
input_df= pd.DataFrame(temp_df.toarray(),columns=tf_idf_vectorizer.get_feature_names())

### load the model

model_path='multinomial_clf.pkl'
model_clf= pickle.load(open(model_path,'rb'))

y_pred= model_clf.predict(input_df)
y_pred_prob= model_clf.predict_proba(input_df)

### load the label encoder
label_encoder_file= 'label_encoder.pkl'
label_encoder= pickle.load(open(label_encoder_file,'rb'))

label_class= label_encoder.inverse_transform(y_pred)
probability_class= y_pred_prob[0][np.argmax(y_pred_prob[0])]*100


print(f'{label_class[0]} is the predicted class/category where website belongs to..........')










