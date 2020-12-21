#!/usr/bin/env python
# coding: utf-8
# In[1]:


import streamlit as st


# In[1]:
import Analysi_Functions as f 
# In[ ]:

st.title("Book Analyzer")
st.header("Transform books into smart charts(Entities,Emotions,Sentiment) ")
booktitle  = st.text_input("Please input book title" ,  'booktitle')
file_path = st.text_input("Please input file path" ,"Please input file path")
                           
if file_path is not "Please input file path":
  st.write('This will take a while...')
  sentences = f.Load_book(file_path)
  st.write("Loading text...")
    
    
  st.write("Analyzing Emotions...")
  df_emo = f.Analyzse_Emotions(sentences)
  st.write("Done...")
  f.Display_emotions(df_emo , booktitle)  

    
  st.write("Analyzing Entities...")   
  df_ents = f.Analyze_entities(sentences)
  st.write("Done...")    
  f.Display_entities(df_ents , booktitle)
    
  #st.write("Analyzing Sentiment...")    
  #df_sent = f.Analyzse_sentiment(sentences)
  #st.write("Done...")   
  #f.Display_sentiment(df_sent , booktitle)
