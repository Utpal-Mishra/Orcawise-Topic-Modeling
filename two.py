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

import string

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
    
    def lemmatize_stemming(text):
        stemmer = SnowballStemmer(language='english')
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        return result
    
    st.header("Training on Seen Document")
    
    data = pd.DataFrame({'headline_text': content})
    data_text = data[['headline_text']]
    data_text['index'] = data_text.index
    documents = data_text      
          
    if st.checkbox("Show DataFrame"):
        st.dataframe(documents)
            
    
    ##Select a document to preview after preprocessing
    doc_sample = documents[documents['index'] == 0].values[0][0]
    
    if st.checkbox('Document Sample'):
        st.subheader(doc_sample)
    
    words = []
    for word in doc_sample.split(' '):
        words.append(word)
        
    if st.checkbox('Words'):
        st.dataframe(words)
    
    if st.checkbox('Tokenized and Lemmatized Document'):
        st.write(preprocess(doc_sample))
    
    ##Preprocess the headline text, saving the results as ‘processed_docs
    processed_docs = documents['headline_text'].map(preprocess)
    
    if st.checkbox("Prcessed Document"):
        st.write(processed_docs[:10])
        
    #Bag of Words on the Data set
    dictionary = gensim.corpora.Dictionary(processed_docs)
    
    if st.checkbox("Dictionary"):
        st.write(dictionary)
    
    count = 0
    for k, v in dictionary.iteritems():
        print(k, v)
        count += 1
        if count > 10:
            break
        
    #dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    
    #if st.checkbox("Dictionary"):
    #    st.write(dictionary) #dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000))
        
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    if st.checkbox("Corpus Sample"):
        st.subheader(bow_corpus[10])
    
    print("\n")
    
    ##Preview Bag Of Words for our sample preprocessed document.
    bow_doc_10 = bow_corpus[10]
    for i in range(len(bow_doc_10)):
        print("Word {} (\"{}\") appears {} time.".format(bow_doc_10[i][0], 
                                                   dictionary[bow_doc_10[i][0]],
                                                   bow_doc_10[i][1]))
     
    print("\n")
      
    #TF-IDF
    from gensim import corpora, models
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    
    from pprint import pprint
    for doc in corpus_tfidf:
        pprint(doc)
        break
    
    ##############################################################################
        
    #Running LDA using Bag of Words
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
    
    LDA_BOW = []
    for idx, topic in lda_model.print_topics(-1):
            print('\nTopic: {} \nWords: {}'.format(idx, topic))
            LDA_BOW.append(['\nTopic: {} \nWords: {}'.format(idx, topic)])
    
    if st.checkbox("\nLDA using BAG OF WORDS\n"):
        st.write(LDA_BOW)
        
    ##############################################################################
    ##############################################################################
    
    #Running LDA using TF-IDF
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
    
    LDA_TFIDF = []
    for idx, topic in lda_model_tfidf.print_topics(-1):
            #print('\nTopic: {} \nWord: {}'.format(idx, topic))
            LDA_TFIDF.append(['\nTopic: {} \nWord: {}'.format(idx, topic)])
        
    #processed_docs[10]
    if st.checkbox("\nLDA using TF-IDF\n"):
            st.write(LDA_TFIDF)
            
    ##############################################################################
    ##############################################################################
        
    #Performance evaluation by classifying sample document using LDA Bag of Words model
    
    PE_BOW = []
    for index, score in sorted(lda_model[bow_corpus[10]], key=lambda tup: -1*tup[1]):
            #print("\nScore: {}\nTopic: {}".format(score, lda_model.print_topic(index, 10)))
        PE_BOW.append(["\nScore: {}\nTopic: {}".format(score, lda_model.print_topic(index, 10))])
        
    if st.checkbox("\nPERFORMANCE EVALUATION: LDA using BAG OF WORDS Model\n"):
            st.write(PE_BOW)
            
    ##############################################################################
    ##############################################################################
        
    #Performance evaluation by classifying sample document using LDA TF-IDF model
    
    PE_TFIDF = []
    for index, score in sorted(lda_model_tfidf[bow_corpus[10]], key=lambda tup: -1*tup[1]):
            #print("\nScore: {}\nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
            PE_TFIDF.append(["\nScore: {}\nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10))])
            
    if st.checkbox("\nPERFORMANCE EVALUATION: LDA using TF-IDF Model\n"):
        st.write(PE_TFIDF)
        
    ##############################################################################
    ##############################################################################
        
    st.header("Testing on Unseen Document")
    
    #Testing model on unseen document
    unseen_document = 'How a Pentagon deal became an identity crisis for Google'
    bow_vector = dictionary.doc2bow(preprocess(unseen_document))
    
    st.subheader(unseen_document)
    #st.subheader(bow_vector)
    
    TEST = []
    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
            #print("\nScore: {}\nTopic: {}".format(score, lda_model.print_topic(index, 5)))
            TEST.append(["\nScore: {}\nTopic: {}".format(score, lda_model.print_topic(index, 5))])
    
    if st.checkbox("\nTESTING MODEL ON UNSEEN DOCUMENT\n"):
        st.write(TEST)