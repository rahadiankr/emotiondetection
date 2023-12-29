# app.py

import streamlit as st

import pandas as pd
import altair as alt
import tensorflow as tf
import seaborn as sns
import numpy as np
import joblib

# Text Cleaning
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Label Encoder
from sklearn.preprocessing import LabelEncoder
nltk.download("stopwords")
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))
lemmatizer= WordNetLemmatizer()
nltk.download('omw-1.4')
sns.set(font_scale=1.3)

# Load the tokenizer used during training
tokenizer = tf.keras.preprocessing.text.Tokenizer()

emotions_emoji_dict = {"anger": "😠", "fear": "😨", "joy": "😂", "sadness": "😔"}

# Target labels
target_labels = ["anger", "fear", "joy", "sadness"]
label_encoder = LabelEncoder()
label_encoder.fit(target_labels)

# Load ML Models .h5
model_path = "model.h5"
model = tf.keras.models.load_model(model_path)

# Text Cleaning 
# Penjelasan Fungsi-fungsi yang digunakan untuk membersihkan text
#     @lemmatization : mengubah kata-kata menjadi kata dasar
#     @remove_stop_words : menghapus kata-kata yang tidak memiliki makna
#     @Removing_numbers : menghapus angka-angka
#     @lower_case : mengubah huruf menjadi huruf kecil
#     @Removing_punctuations : menghapus tanda baca
#     @Removing_urls : menghapus url
#     @remove_small_sentences : menghapus kalimat yang memiliki panjang kurang dari 3
#     @normalized_sentence : menggabungkan semua fungsi diatas untuk membersihkan text
def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text):

    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    df.Text=df.Text.apply(lambda text : lower_case(text))
    df.Text=df.Text.apply(lambda text : remove_stop_words(text))
    df.Text=df.Text.apply(lambda text : Removing_numbers(text))
    df.Text=df.Text.apply(lambda text : Removing_punctuations(text))
    df.Text=df.Text.apply(lambda text : Removing_urls(text))
    df.Text=df.Text.apply(lambda text : lemmatization(text))
    return df

def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    sentence= Removing_numbers(sentence)
    sentence= Removing_punctuations(sentence)
    sentence= Removing_urls(sentence)
    sentence= lemmatization(sentence)
    return sentence
# Predict Emotions
#     @predict_emotions : memprediksi emosi dari text yang diinputkan
#     @get_prediction_proba : memprediksi probabilitas dari text yang diinputkan

def predict_emotions(sentence):
    sentence = normalized_sentence(sentence)
    token_path = "tokenizer_sentiment.joblib"
    tokenizer = joblib.load(token_path)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = tf.keras.preprocessing.sequence.pad_sequences(sentence, maxlen=229, truncating='pre')
    result = label_encoder.inverse_transform(np.argmax(model.predict(sentence), axis=-1))[0]
    proba =  np.max(model.predict(sentence))
    return result, proba, model.predict(sentence)[0]
    

def main():  
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)
            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction[0]]
            st.write("{} : {}".format(prediction[0], emoji_icon))
            st.write("Confidence : {:5.2f}%" .format(100 * prediction[1]))

        with col2:
            st.success("Prediction Probability")
            df = pd.DataFrame(data=prediction[2], index=target_labels, columns=['values'])
            chart = alt.Chart(df.reset_index()).mark_bar().encode(x='index', y='values', color='index')
            st.altair_chart(chart, use_container_width=True)
            

if __name__ == '__main__':
    main()