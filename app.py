import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stm=PorterStemmer()
stop=set(stopwords.words("english"))

def clean(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()

    for i in text:
        if i not in stop and i not in string.punctuation:
            y.append(stm.stem(i))
    return " ".join(y)

st.title("Spam Classifier")
dat=st.text_input("Enter your msg/text")
btn=st.button("predict")

model=pickle.load(open("model.pkl","rb"))
cv=pickle.load(open("Cv.pkl","rb"))
l=['Ham','Spam']
if(btn):
    dt=cv.transform([clean(dat)]).toarray()
    pd=model.predict(dt)[0]
    st.write(l[pd])