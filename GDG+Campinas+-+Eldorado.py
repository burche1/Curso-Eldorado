
# coding: utf-8

# In[24]:

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
import nltk
import re
import pandas as pd


# In[34]:

nltk.data.path.append('C:\\Users\Alexandre\Documents\Curso\Cluster\cluster-master\ntlk_data')
stemmer = SnowballStemmer('portuguese')
stopwords = nltk.corpus.stopwords.words('portuguese')

GITHUB

git pull origin master

git status
git add . ou nome do arquivo
git commit -m "mensagem qualquer"
git push origin master
# In[38]:

df = pd.read_csv("C:\\Users\Alexandre\Documents\Curso\Cluster\cluster-master\datasets\documents.csv")
df
documents = df['docs'].tolist()


# In[35]:

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# In[42]:

tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,
                                 min_df=0.0, stop_words=stopwords,
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
matrix = tfidf_vectorizer.fit_transform(documents)


# In[45]:

km = KMeans(n_clusters=5,init='random')
km.fit_predict(matrix)
clusters=km.labels_.tolist()


# In[46]:

result = {'question': documents, 'cluster_id': clusters}
pd.DataFrame(result, index=clusters).sort_values(by='cluster_id',ascending=True)


# In[ ]:



