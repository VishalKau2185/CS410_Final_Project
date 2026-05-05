import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import html
from bs4 import BeautifulSoup
from collections import Counter 
import math

class Engine():
    K = 3
    b = .75
    path = None
    df_orig = None
    df_clean = None
    punctuations = ""
    punct_re = ""
    stop_words=set()
    stemmer = None
    vocab = None
    title_col = ""
    collection_col = ""

    def __init__(self, dataset_path, vocab_len, title_col, collection_col):
        self.path = dataset_path
        self.df_orig = pd.read_csv(self.path).drop_duplicates()
        print(self.df_orig.head())
        
        # create regex to quickly get rid of punctuation and numbers
        self.punctuations = '\'\"\\,<>./?@#$%^&*_~/!()-=[]{};:'
        self.punct_re = "[" + re.escape(self.punctuations) + "0-9" + "]"

        # load stopwords, stemmer
        # initialize vocab, title column, and collection column
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words("english"))
        self.vocab = np.zeros(vocab_len)
        self.title_col = title_col
        self.collection_col = collection_col
        self.stemmer = PorterStemmer()

    def preprocess(self):
        # preprocess
        self.df_clean = self.df_orig.copy()
        self.df_clean[self.collection_col] = \
            self.df_orig[self.collection_col].apply(self.clean)
        
        # build vocab
        allWords = []

        for text in self.df_clean[self.collection_col]:
            if isinstance(text, str):
                allWords.extend(text.split())
        counter = Counter(allWords)
        vocab = [w for w, _ in counter.most_common(self.vocab.shape[0])]

        self.vocab = np.array(vocab)

        # calculate AVDL
        self.avdl = self.df_clean[self.collection_col] \
            .apply(lambda x: len(str(x).split())).mean()

        print(self.vocab)
    
    # function to clean text
    def clean(self, plot):
        if pd.isna(plot):
            return ""
        
        # removes html tags, numbers, punctuation and makes plot lowercase
        plot = html.unescape(plot)
        plot = re.sub(r"\b(quot|apos|amp|lt|gt)\b", " ", plot)
        plot = BeautifulSoup(plot, "html.parser").get_text()
        plot = plot.lower().strip()
        plot = re.sub(self.punct_re, " ", plot)
        plot = re.sub(r"\s+", " ", plot)
        
        # stem words and remove stop words
        words=plot.split()
        stemmed = []
        for word in words:
            if word not in self.stop_words:
                stemmed_word = self.stemmer.stem(word)
                stemmed.append(stemmed_word)
        
        return ' '.join(stemmed)
    
    def show_stats(self):
        # Before and After
        print("Before Preprocessing:\n", self.df_orig[self.collection_col][12])
        print("After Preprocessing:\n", self.df_clean[self.collection_col][12])
        
        # show stats
        print("Before Preprocessing:")
        print(self.df_orig[self.collection_col] \
            .apply(lambda x: len(str(x).split())).describe())
        
        print("After Preprocessing:")
        print(self.df_clean[self.collection_col] \
            .apply(lambda x: len(str(x).split())).describe())

    # Updates the vocabulary to add the words in the query
    def adapt_vocab_query(self, query):   
        vocab = self.vocab.copy()
        queryWords = query.split()
        for w in queryWords:
            if w not in vocab:
                vocab = np.append(vocab, w)
        self.vocab = vocab

    # compute IDF vector
    def compute_IDF(self, M, collection):
        self.IDF  = np.zeros(self.vocab.size)
        allWordsDocFreq = {}
        for text in collection:
            words = set(text.split())
            for w in words:
                allWordsDocFreq[w] = allWordsDocFreq.get(w, 0) + 1

        for idx, w in enumerate(self.vocab):
            df = allWordsDocFreq.get(w, 1)
            self.IDF[idx] = math.log((M+1)/df)

    # implements BM25 with TF-IDF and document length normalization
    def text2TFIDF(self, text, applyBM25_and_IDF=False):
        vocab = self.vocab
        tfidfVector = np.zeros(vocab.size)
        wc = Counter(text.split())
        for idx, word in enumerate(vocab):
            if word in wc:
                # for query and document
                c = wc[word]
                tfidfVector[idx] = c
                if applyBM25_and_IDF:
                    # for document only
                    d = len(text.split(" "))
                    normalizer = 1 - self.b + (self.b * d / self.avdl)
                    tfidfVector[idx] *= (self.K + 1)/((self.K*normalizer) + c)
                    tfidfVector[idx] *= self.IDF[idx]
        return tfidfVector
    
    # compute relevance scores for all docs given a query
    def tfidf_score(self, query, doc, applyBM25_and_IDF=False):
        q = self.text2TFIDF(query)
        d = self.text2TFIDF(doc, applyBM25_and_IDF)

        relevance = np.dot(q, d)
        return relevance

    # function to perform searches
    def search(self, q):
        query = self.clean(q)
        self.adapt_vocab_query(query) 
        self.compute_IDF(self.df_clean.shape[0], self.df_clean[self.collection_col])

        relevances = np.zeros(self.df_clean.shape[0])

        # compute relevance scores for all docs given a query using
        # BM25 with TF-IDF and document length normalization
        for index, row in self.df_clean.iterrows():
            relevances[index] = self.tfidf_score(query, row[self.collection_col], True)

        # sort and print top 5 and bottom 5 most relevant documents
        sorted_indices = np.argsort(relevances)[::-1]

        print("\n\tTop 5 most relevant movies:")
        for idx in sorted_indices[:5]:
            print(f"\t\tDocID: {idx+1}, Score: {relevances[idx]}")
            print(f"\t\t\tTitle: {self.df_clean.iloc[idx][self.title_col]}")
            print(f"\t\t\tPlot: {self.df_orig.iloc[idx][self.collection_col][:250]}...")
        
        print("\n\tBottom 5 least relevant documents:")
        for idx in sorted_indices[-5:]:
            print(f"\t\tDocID: {idx+1}, Score: {relevances[idx]}")
            print(f"\t\t\tTitle: {self.df_clean.iloc[idx][self.title_col]}")
            print(f"\t\t\tPlot: {self.df_orig.iloc[idx][self.collection_col][:250]}...")

        return relevances
    
    # baseline to compare against
    def baseline(self, q):

        q = self.clean(q)
        words = q.split()
        scores = []
        df = self.df_clean.copy()

        # count how many words in the query appear in doc
        for plot in df[self.collection_col]:
            plot = set(plot.split())
            count = 0

            for w in words:
                if w in plot:
                    count = count + 1

            scores.append(count)

        df["base_score"] = scores
        temp = df.sort_values(by="base_score", ascending=False)

        print("Baseline results for:", q)
        print()

        # sort and print 5 most relevant documents
        for i in range(5):
            print(i + 1, "-", temp[self.title_col].iloc[i])
            print("Score:", temp["base_score"].iloc[i])
            print(f"Plot: {temp[self.collection_col].iloc[i][:250]}...")
            print()