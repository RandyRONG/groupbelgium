import json
import os
import re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import string
import operator
from langid import classify
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
from collections import defaultdict,Counter
import math
import gensim
from gensim.corpora import Dictionary
from PIL import Image 
from wordcloud import WordCloud, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pdtext.tm import topic_words
from sklearn.decomposition import LatentDirichletAllocation

def WordCloudDraw(df_tfidf,words_range,picture_path,output_pic_path,mode):
    word_cn_list_2 = []

    for j in range(0,df_tfidf.shape[0]):
        word_cn = df_tfidf.iloc[j,0]
        sub_word_cn_list = [word_cn] * (int(df_tfidf.iloc[j,1]*10000)+1)
        word_cn_list_2.extend(sub_word_cn_list)

    word_counts = Counter(word_cn_list_2)
    print (word_counts)
        
    mask = np.array(Image.open(picture_path))
    wc = WordCloud(
        background_color='white',
        font_path="C:/WINDOWS/Fonts/TIMES.TTF",
        mask=mask, 
        max_words=words_range, 
        max_font_size=64, 
        scale=32,  
        width = 400,
        height = 200,
        color_func = ImageColorGenerator(mask)
    )

    wc.generate_from_frequencies(word_counts)

    

    wc.to_file(output_pic_path) 
    if mode == 'show':
        plt.imshow(wc) 
        plt.axis('off') 
        plt.show() 

def GetTFIDF(list_words,words_range,min_count):
    doc_frequency=defaultdict(int)
    for word_list in list_words:
        for i in word_list:
            doc_frequency[str(i)]+=1
    word_tf={} 
    for i in doc_frequency:
        word_tf[str(i)]=doc_frequency[str(i)]/sum(doc_frequency.values())


    doc_num=len(list_words)
    word_idf={}
    word_doc=defaultdict(int) 
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                word_doc[str(i)]+=1
    for i in doc_frequency:
        word_idf[str(i)]=math.log(doc_num/(word_doc[str(i)]+1))


    word_tf_idf={}
    for i in doc_frequency:
        if doc_frequency[str(i)] <= min_count:
            continue
        word_tf_idf[str(i)]= round(word_tf[str(i)]*word_idf[str(i)],4)

    dict_feature_select=sorted(word_tf_idf.items(),key=operator.itemgetter(1),reverse=True)[:words_range]
    df_tfidf = pd.DataFrame(dict_feature_select, columns=['word', 'TF-IDF']) 
    # df_tfidf.to_csv('output.csv',index=False)
    print (df_tfidf)
    return df_tfidf


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV
    else:
        return ''

def merge(words,lmtzr,rejected_words,words_dict):
    words_list = []
    
    for word in words:
        # if word in rejected_words:
        #     continue
        if word  in words_dict:
            words_list.append(words_dict[word])
            continue
        tag = pos_tag(word_tokenize(word)) # tag is like [('bigger', 'JJR')]

        pos = get_wordnet_pos(tag[0][1])
        if pos:
            lemmatized_word = lmtzr.lemmatize(word, pos)
#                 print ([tag,pos,lemmatized_word])
            if lemmatized_word in rejected_words:
                continue
            words_dict[word] = lemmatized_word
            words_list.append(words_dict[word])
        # else:
        #     words_list.append(word)

    return words_list


def replace_abbreviations(text):
    # patterns that used to find or/and replace particular chars or words
    
    new_text = text
    
    # to find chars that are not a letter, a blank or a quotation
    pat_letter = re.compile(r'[^a-zA-Z \']+')
    new_text = pat_letter.sub(' ', text).strip().lower()
        
    # to find the 's following the pronouns. re.I is refers to ignore case
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    new_text = pat_is.sub(r"\1 is", new_text)
    
    # to find the 's following the letters
    pat_s = re.compile("(?<=[a-zA-Z])\'s")
    new_text = pat_s.sub("", new_text)
    
    # to find the ' following the words ending by s
    pat_s2 = re.compile("(?<=s)\'s?")
    new_text = pat_s2.sub("", new_text)
    
    # to find the abbreviation of not
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")
    new_text = pat_not.sub(" not", new_text)
    
    # to find the abbreviation of would
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    new_text = pat_would.sub(" would", new_text)
    
    # to find the abbreviation of will
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    new_text = pat_will.sub(" will", new_text)
    
    # to find the abbreviation of am
    pat_am = re.compile("(?<=[I|i])\'m")
    new_text = pat_am.sub(" am", new_text)
    
    # to find the abbreviation of are
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    new_text = pat_are.sub(" are", new_text)
    
    # to find the abbreviation of have
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")
    new_text = pat_ve.sub(" have", new_text)
    
    new_text = new_text.replace('\'', ' ')
    
    return new_text


def get_words(text,rejected_words,words_dict):  
    lmtzr = WordNetLemmatizer()
    words_list = (merge(replace_abbreviations(text).split(),lmtzr,rejected_words,words_dict))
    text = ' '.join(words_list)
    return text

def text_prepare(text,rejected_words,not_related_words,words_dict):
    text = text.lower() 
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@#+,;]') 
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    text = REPLACE_BY_SPACE_RE.sub(' ',text) 
    remove = str.maketrans('','',string.punctuation) 
    text = text.translate(remove)
    text = BAD_SYMBOLS_RE.sub(' ',text) 
    for not_related_word in not_related_words:
        if not_related_word in text.split():
            return []
    text = get_words(text,rejected_words,words_dict)   
    STOPWORDS = set(stopwords.words('english'))
    words = [w for w in replace_abbreviations(text).split() if w not in STOPWORDS and len(w)>2]
    return words

def LDAModel(texts,num_topics,min_count):
    vect = CountVectorizer(min_df=min_count, 
                       max_df=0.9,
                      max_features=1000)
    vect.fit(texts)
    tf = vect.transform(texts)
    lda_model = LatentDirichletAllocation(n_components   = num_topics,
                                      max_iter       = 10,
                                    #   evaluate_every = 5,
                                    #   verbose = 2,
                                    #   n_jobs= 2,
                                     )
    lda_model.fit(tf)
    topic_df = topic_words(lda_model, vect)
    return topic_df

if __name__ == '__main__':
    
    words_range = 200
    min_count = 3
    num_topics = 3
    num_words = 5
    root_dir = '../../Data/SocialCulture/'
    picture_path = root_dir+'heatwaves.jpg'
    output_pic_path = root_dir+'output.png'
    df = pd.read_csv(root_dir+'heatwaves_twitter.csv')
    text_list = df['text']
    required_pos = ['CD','FW','JJ','JJR','JJS','LS','NN','NNS','NNP','NNPS','RB','UH','VB','WDT']
    rejected_words = ['heatwave','heatwaves','http','https','isnt','im','tco','dont','amp','ive','thats','didnt','havent','george']
    not_related_words = ['dnf','dream','fcu']

    words_dict = {}
    text_list = [text_prepare(x,rejected_words,not_related_words,words_dict) for x in text_list]
    text_list = [x for x in text_list if len(x)>0 ]
    # print (text_list)
    print (len(text_list))
    df_tfidf = GetTFIDF(text_list,words_range,min_count)
    WordCloudDraw(df_tfidf,words_range,picture_path,output_pic_path,'show')
    dictionary = Dictionary(text_list)
    corpus = [dictionary.doc2bow(text) for text in text_list]
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20) 
    print(ldamodel.print_topics(num_topics=num_topics, num_words=num_words))
    topic_df = LDAModel([' '.join(i) for i in text_list],num_topics,min_count)
    print (topic_df)



    