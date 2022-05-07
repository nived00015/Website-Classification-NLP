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
from flask import Flask,render_template,request

app= Flask(__name__)
@app.route('/')
def home():
    return "<h1>This is homepage !!!! </h1>"
@app.route('/form',methods=['GET','POST'])
def website_class_predictor():
    if request.method == 'POST':
        text= request.form.get("website_text")
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
        vectorizer_filepath= 'tf_idf_vectorizer.pkl'
        tf_idf_vectorizer= pickle.load(open(vectorizer_filepath,'rb'))
        temp_df= tf_idf_vectorizer.transform([cleaned_text])
        input_df= pd.DataFrame(temp_df.toarray(),columns=tf_idf_vectorizer.get_feature_names())
        model_path='multinomial_clf.pkl'
        model_clf= pickle.load(open(model_path,'rb'))
        y_pred= model_clf.predict(input_df)
        y_pred_prob= model_clf.predict_proba(input_df)

        ### load the label encoder
        label_encoder_file= 'label_encoder.pkl'
        label_encoder= pickle.load(open(label_encoder_file,'rb'))

        label_class= label_encoder.inverse_transform(y_pred)
        probability_class= y_pred_prob[0][np.argmax(y_pred_prob[0])]*100

        return f"Predicted website class: {label_class[0]}"
    return render_template('index.html')
        


app.run()



def predict_webclass(text):
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
    vectorizer_filepath= 'tf_idf_vectorizer.pkl'
    tf_idf_vectorizer= pickle.load(open(vectorizer_filepath,'rb'))
    temp_df= tf_idf_vectorizer.transform([cleaned_text])
    input_df= pd.DataFrame(temp_df.toarray(),columns=tf_idf_vectorizer.get_feature_names())
    model_path='multinomial_clf.pkl'
    model_clf= pickle.load(open(model_path,'rb'))
    y_pred= model_clf.predict(input_df)
    y_pred_prob= model_clf.predict_proba(input_df)

    ### load the label encoder
    label_encoder_file= 'label_encoder.pkl'
    label_encoder= pickle.load(open(label_encoder_file,'rb'))

    label_class= label_encoder.inverse_transform(y_pred)
    probability_class= y_pred_prob[0][np.argmax(y_pred_prob[0])]*100


    return label_class[0]











