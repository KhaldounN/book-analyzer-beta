#!/usr/bin/env python
# coding: utf-8

# # Defining Analysis Functions

# In[9]:
import plotly.graph_objects as go
from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn")
import math
import plotly.express as px
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import spacy
nlp_md = spacy.load('en_core_web_md')
import pickle
from nltk.tokenize import RegexpTokenizer
import streamlit as st


# load book , clean data and split into sentences 
def Load_book(path):
  book = open(path, 'r').read()
  book = book.replace("_",' ')
  book = book.replace("\n",'')
  sentences  = book.split('.')

  return sentences 

# run sentiment analysis on each sentence in the book and save them into df_setiment 
def Analyzse_sentiment(sentences):
  tagger = TextClassifier.load('sentiment')
  df_sentiment = pd.DataFrame((np.zeros((4,int(len(sentences))))))
  for i, sentence in enumerate(sentences):
    warnings.filterwarnings('ignore')
    sentence = Sentence(sentence)
    tagger.predict(sentence)
    df_sentiment[i].update(sentence.labels)


  for i , sentence in enumerate(df_sentiment.iloc[0,:]):
      try:
        df_sentiment.iloc[1,i] = sentence.value
        if sentence.value == 'NEGATIVE':
          df_sentiment.iloc[2,i] = sentence.score * (-1)
        else:
          df_sentiment.iloc[2,i] = sentence.score
      except:
        pass

  return df_sentiment

# run emotions dedection on each sentence 
def Analyzse_Emotions(sentences):
  token = RegexpTokenizer(r'[a-zA-Z0-9]+')
  cv = pickle.load(open(r'vectorizer.pickle', 'rb'))
  df_emotions = pd.DataFrame((np.zeros((1,int(len(sentences))))))
  model = pickle.load(open(r"emotions_detector.sav", 'rb'))
  for i , sentence in enumerate( sentences):
    try:
        result = model.predict(cv.transform([sentence]))
        df_emotions[i].update(result)
    except:
      pass
  return df_emotions


# The Below Function retrive a Data Frame containing all the Entities detected and entity label ( index is sentence in text)
def Analyze_entities(sentences):
  df_ent = pd.DataFrame((np.zeros((100,int(len(sentences))))))
  df_ents = pd.DataFrame((np.zeros((3,int(len(sentences))))))
  l = []
  for i , sentence in enumerate(sentences):
    doc = nlp_md(sentence)
    df_ent[i].update(doc.ents)
  for  j , raw in df_ent.iteritems():
    for i , item in enumerate(raw):
      item2 = item
      if str(item2) != '0.0':
        l = [j , item.text , item.label_]
        df_ents[j].update(l)
  df_ents = df_ents.T
  df_ents = df_ents[df_ents[1] != 0]
  df_ents.columns = ['index','entity','label']
  df_ents = df_ents[df_ents.label.isin([ 'PERSON' ,'NORP','GPE', 'LOC','EVENT', 'LAW'])]
  
  return df_ents
  

    
#Main Function to run program 
def Analyze_Book(path,booktitle="Book Title"):
  sentences = Load_book(path)
  print("analyzing")
  df_sent = Analyzse_sentiment(sentences)
  df_emo = Analyzse_Emotions(sentences)
  print('wait for it...')
  df_ent = Analyze_entities(sentences)
  fig_sentiment = Display_sentiment(df_sent , booktitle)
  Display_entities(df_ent , booktitle)
  Display_emotions(df_emo , booktitle )


# calcualte Major and Minor sentiment and vizualize it in a dashabord style

def Display_sentiment(df_sentiment  , booktitle = ' book title'):
  warnings.filterwarnings("always")

  l = df_sentiment.iloc[2,:]
  l = pd.DataFrame(l) 
  l['Major'] = l.iloc[:,0].rolling(40).mean()
  l['Minor'] = l.iloc[:,0].rolling(10).mean()

  fig_sentiment, axes = plt.subplots(nrows=2, ncols=1, figsize=(30, 14))
  axes[0].set_title((str(booktitle) + ' : Sentimental Tendency') , fontsize = 30  , loc='left')
  axes[0].plot(l['Major'])
  axes[0].set_xlabel('Sentences' , fontsize=20)
  axes[0].set_ylabel('Sentiment', fontsize=20)
  axes[0].legend(['Major Sentiment(rolling average for 30 sentences)'], fontsize=20)
  axes[0].fill_between(l.index, l['Major'] ,0 , where=l['Major'] >= 0, facecolor='lightgreen',  interpolate=True)
  axes[0].fill_between(l.index, l['Major'] ,0 , where=l['Major'] <= 0, facecolor='lightcoral',  interpolate=True)
  axes[0].set_facecolor('white')
  axes[1].plot(l['Minor'])
  axes[1].set_xlabel('Sentences' , fontsize=15)
  axes[1].set_ylabel('Sentiment' , fontsize=15)
  axes[1].fill_between(l.index, l['Minor'] ,0 , where=l['Minor'] >= 0, facecolor='lightgreen',  interpolate=True)
  axes[1].fill_between(l.index, l['Minor'] ,0 , where=l['Minor'] <= 0, facecolor='lightcoral',  interpolate=True)
  axes[1].set_facecolor('white')
  axes[1].legend(['Minor Sentiment(rolling average for 5 sentences)'] , fontsize=20)
  st.write(fig_sentiment)

# perform calculations on df_emotions and display results 
def Display_emotions(df_emotions , booktitle = 'Book Title'):
    
  df_emotions = pd.DataFrame(df_emotions)
  l = df_emotions.T.value_counts().index
  l = l.tolist()
  v = df_emotions.T.value_counts()
  v = v.tolist()
  fig_emo1 = go.Figure(data=go.Scatterpolar(r=v,theta=l,fill='toself'))
  fig_emo1.update_layout(polar=dict(radialaxis=dict(visible=False),),showlegend=False ,   title={
        'text': (str(booktitle)+' : Composition of Emotion ')})
  fig_emo1.update_layout(width=1100,height=700)
  st.write(fig_emo1)

  df_emotions1 = df_emotions.T
  df_emotions1['Sentences(50)'] =  ((df_emotions1.index/50).astype(int))*50
  df_emotions1['1'] = df_emotions1.index
  data = df_emotions1.groupby(by=['Sentences(50)',0]).count()
  data = pd.DataFrame(data)
  data.reset_index(inplace=True)  
  data.columns = ['Sentences','Emotion','Count']
  fig_emo2 = px.line(data,
                 x=	'Sentences' ,  y='Count', color='Emotion',
                 title=((str(booktitle)+" : Emotional Journey")))
  fig_emo2.update_layout(width=1100,height=700)
  st.write(fig_emo2)



  
#the Below Function draw a Sunburt chart displaying entities mentioned alog with their types 

def Display_entities(df_ents , booktitle = 'Booktitle'):

  # replace label with more understandble labels :
  df_ents = df_ents.replace('NORP', 'Nationalities/groups')
  df_ents = df_ents.replace('PERSON', 'People')
  df_ents = df_ents.replace('GPE', 'Countries/Cities')
  df_ents = df_ents.replace('LOC', 'Locations')
  df_ents = df_ents.replace('EVENT', 'Event')
  df_ents = df_ents.replace('LAW', 'Laws')

  # FIRST FIGURE
  df_ents1 = df_ents
  df_ents1 = df_ents1.groupby(by=['label','entity']).count().sort_values(by='index', ascending=False).head(20)
  df_ents1 = df_ents1.reset_index(level=['label', 'entity'])
  fig = px.sunburst(df_ents1, path=['label', 'entity'], values='index' , title=( str(booktitle)+" : Mentions of Entities (locations, nationalities, events, and characters)"))
  fig.update_layout(width=1100,height=700)
  st.write(fig)

  #SECOND FIGURE
  df_ents2 = df_ents
  df_ents2['sentence'] = ((df_ents2.index/50).astype(int))*50
  df_ents2 = df_ents2.groupby(by=['sentence','label','entity']).count()
  df_ents2 = df_ents2.reset_index(level=['sentence','label','entity'])
  df_ents2 = df_ents2[df_ents2['index'] > 1]
  df_ents2.columns = ['sentence(every 50 senteces)', 'label', 'entity', 'count']
  fig = px.scatter(df_ents2,x='sentence(every 50 senteces)', y="entity",
	         size="count", color="entity")
  fig.update_layout(width=1100,height=700)
  st.write(fig)



