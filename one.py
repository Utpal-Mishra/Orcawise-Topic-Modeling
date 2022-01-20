import streamlit as st
import time
import pandas as pd

"""data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False);
data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text

print(len(documents))
print(documents[:5])"""

import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)

import collections
import nltk #pip install nltk
#from nltk import stem as stemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('wordnet')

##############################################################################
##############################################################################

def app():
    st.title("ORCAWISE COLLOCATIONS")
    
    st.header("TOPIC MODELING")
    
    st.subheader("Loading the Text Data....")
    
    file = st.file_uploader("Upload file")
    show_file = st.empty()
    
    if not file:
        show_file.info("Please upload a file of type: " + ", ".join([".txt"]))
        return
    
    label = st.empty()
    bar = st.progress(0)
    
    for i in range(100):
        # Update progress bar with iterations
        label.text(f'Loaded {i+1} %')
        bar.progress(i+1)
        time.sleep(0.01)
      
    ".... and now we're done!!!"
    
##############################################################################
 
    #path = st.text_input('CSV file path')
    content = []
    for line in file:
        if str(line) != "b'\r\n'":
            content.append(str(line))
        #content[-1] = content[-1].split('.')
    st.title('Content')
    st.write(content)
    
    data = pd.DataFrame({'headline_text': content})
    #data = df.transpose().reset_index()
    #data.columns = data[['headline_text']]
    data['index'] = data.index
    data = data.dropna(subset=['headline_text'])

    
    """for i in data['headline_text']:
        i.replace("\r\n'b'\r\n'b'", "")"""
            
          
    if st.checkbox("Show DataFrame"):
        st.dataframe(data)
        
    ##############################################################################
    ##############################################################################
        
    ##Select a document to preview after preprocessing
    #doc_sample = documents[documents['index'] == 10]['headline_text']
        
    if st.checkbox('Original Document'):
        st.dataframe(data['headline_text'])
    
    sentence = []
    for word in data['headline_text'].str.split(' '):
        sentence.append(word)
        
    if st.checkbox('Sentence'):
        st.write(sentence)
     
    words = [item for item in sentence if item != []]   
        
    if st.checkbox('Words'):
        #print(words, end="\n")
        st.write(words)
         
    ##Write a function to perform lemmatize and stem preprocessing steps on the data set
        
    for i in range(len(data['headline_text'])):
        data['headline_text'][i] = gensim.utils.simple_preprocess(data['headline_text'][i])
    
    text = []
    for i in range(len(data['headline_text'])):
        temp = []
        for k in range(len(data['headline_text'][i])):
            if data['headline_text'][i][k] not in gensim.parsing.preprocessing.STOPWORDS and len(data['headline_text'][i][k]) > 2:
                temp.append(data['headline_text'][i][k])
        text.append(temp)
        
    text = [item for item in text if item != []]           
        
    if st.checkbox("Lemmatized and Stemmed DataFrame"):
        #print(text)
        st.write(text)
    
    ##############################################################################
    
    ##Preprocess the headline text, saving the results as processed_docs
    """processed_docs = documents['headline_text'].astype(str).map(preprocess)
    
    if st.checkbox('Processed Document'):
        st.dataframe(preprocess(processed_docs))"""
    
    ##############################################################################
    ##############################################################################
    
    #Bag of Words on the Data set
    bagofwords = []
    bagofcounts = []
    for words in text:
        for word in words:
            count = 0
            for loop in text:
                count += loop.count(word)
            #print(word, count)
            bagofwords.append(word)
            bagofcounts.append(count)
                    
        
    bag = pd.DataFrame({'Words': bagofwords, 'Count': bagofcounts})
    wordscount = bag.to_dict('records')
    
    
    """counter = []
    for one in text:
        for two in one:
            count=0
            for three in one:
                count += three.count(two)
            counter.append([three, count])   
    st.write(counter)
    
    dictionary = gensim.corpora.Dictionary(text)
    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 10:
            break
        
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    
    if st.checkbox('Bag of Words'):
        st.write(dictionary)"""
    
    ##############################################################################
    
    ##Gensim doc2bow
    #bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]       
    
    bag['index'] = bag.index
    for i in bag['index']:
        bag['index'][i] += 1    
    
    if st.checkbox('Gensim Doc2bow'):
        st.dataframe(bag)
    
    
    ##Preview Bag Of Words for our sample preprocessed document.
    """bow_doc_10 = bow_corpus[1]
    for i in range(len(bow_doc_10)):
        print("Word {} (\"{}\") appears {} time.".format(bow_doc_10[i][0], 
                                                   dictionary[bow_doc_10[i][0]], bow_doc_10[i][1]))"""
    
    ##############################################################################
    ##############################################################################
    
    """for s in range(len(sentence)):
        for w in range(len(sentence[s])):
            print(s, w, sentence[s][w])
        print("\n")"""
    
    
    #bagofwords = []
    #bagofcounts = []
    TFIDF = []
    
    for words in range(len(text)):
        tfidf = []
        
        for word in range(len(text[words])):
            count = 0
            for loop in text:
                count += loop.count(text[words][word])
            #print(text[words][word], ":: (",words, ",", word, ",", count,")")
            tfidf.append([word, count])
            #Dictionary([text[words][word], str(count)])
            #print([count, text[words][word]])
        #print(list(set(map(tuple,tfidf))), "\n")
        print(len(tfidf))
        TFIDF.append(list(set(map(tuple,tfidf))))
    
    #print(TFIDF)
    #print(list(set(map(tuple,TFIDF))))
    
    if st.checkbox('Initial TF-IDF'):
        st.write(TFIDF)
    
    #TF-IDF
    tfidf = models.TfidfModel(TFIDF)
    corpus_tfidf = tfidf[TFIDF]
    
    #from pprint import pprint
    doc = []
    for d in corpus_tfidf:
        #print(d)
        doc.append(d)
    
    print("\n\n")

    """for i in doc:
        print(len(i))
        
    print("\n")
    
    for i in text:
        print(i, len(i))"""
        
    print("\n\n")
 
    if st.checkbox('Final TF-IDF'):
        st.write(TFIDF)
        #st.write(doc)
    
    """if st.checkbox('Dictionary'):
        st.write(Dictionary)"""
        
    ##############################################################################            
    
    for i in range(len(doc)):
        print(doc[i])
    
    #Running LDA using Bag of Words
    """lda_model = gensim.models.LdaMulticore(doc, num_topics=10, id2word=text, passes=2, workers=2)

    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))
        
    
    ##############################################################################
    ##############################################################################
        
    #Running LDA using TF-IDF
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))
        
    if st.checkbox('LDA using TF-IDF'):
        st.dataframe(processed_docs)
    
    ##############################################################################
    ##############################################################################
    
    #Performance evaluation by classifying sample document using LDA Bag of Words model
    for index, score in sorted(lda_model[bow_corpus[1]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
        
    ##############################################################################
    ##############################################################################
        
    #Performance evaluation by classifying sample document using LDA TF-IDF model
    for index, score in sorted(lda_model_tfidf[bow_corpus[1]], key=lambda tup: -1*tup[1]):
        print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
    
    ##############################################################################
    ##############################################################################
        
    #Testing model on unseen document
    unseen_document = 'How a Pentagon deal became an identity crisis for Google'
    bow_vector = dictionary.doc2bow(preprocess(unseen_document))
    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
        print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))"""