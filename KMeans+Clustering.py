
# coding: utf-8

# In[40]:

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import re
import pandas as pd
#adicionando linha

# In[22]:

nltk.data.path.append("/Users/thirauj/Documents/Thiago/IBM/Lib/nltk_data/")
stemmer = SnowballStemmer('portuguese')
stopwords = nltk.corpus.stopwords.words('portuguese')


# In[25]:

documents = ['eu gosto de banana','eu quero andar de bicicleta','eu quero uma bicicleta','gosto de comer banana']


# In[17]:

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# In[24]:

tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                                 min_df=0.0, stop_words=stopwords,
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
matrix = tfidf_vectorizer.fit_transform(documents)


# In[71]:

km = KMeans(n_clusters=2,init='random')
km.fit_predict(matrix)
clusters=km.labels_.tolist()


# In[72]:

result = {'question': documents, 'cluster_id': clusters}
pd.DataFrame(result, index=clusters).sort_values(by='cluster_id',ascending=True)

