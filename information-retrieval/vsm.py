"""
vsm.py

Simple implementation of Vector Space Model

Note: Depend on Numpy, please install it ahead (`pip install numpy`)

Please Run this script using Python3.x 
Tested under Python3.6, Win7 and Python3.5 ubuntu16.04
Author: Richy Zhu
Email: rickyzhu@foxmail.com
"""

from math import log10
from pprint import pprint
import numpy as np

def _tf(tokenized_doc):
    """calculate term frequency for each term in each document"""
    term_tf = {}
    for term in tokenized_doc:
        if term not in term_tf:
            term_tf[term]=1.0
        else:
            term_tf[term]+=1.0

    # pprint(term_tf)
    return term_tf

def _idf(indices, docs_num):
    """calculate inverse document frequency for every term"""
    term_df = {}
    for term in indices:
        # 一个term的df就是倒排索引中这个term的倒排记录表（对应文档列表）的长度 
        term_df.setdefault(term, len(indices[term]))
    
    term_idf = term_df
    for term in term_df:
        term_idf[term] = log10(docs_num /term_df[term])
    # pprint(term_idf)
    return term_idf

def tfidf(tokenized_docs, indices):
    """calcalate tfidf for each term in each document"""
    term_idf = _idf(indices, len(tokenized_docs))
        
    term_tfidf={}
    doc_id=0
    for tokenized_doc in tokenized_docs:
        term_tfidf[doc_id] = {}
        term_tf = _tf(tokenized_doc)
        
        doc_len=len(tokenized_doc)
        for term in tokenized_doc:
            tfidf = term_tf[term]/doc_len * term_idf[term]
            term_tfidf[doc_id][term] =tfidf
        doc_id+=1
    # pprint(term_tfidf)
    return term_tfidf

def build_terms_dictionary(tokenized_docs):
    """assign an ID for each term in the vocabulary"""
    vocabulary = set()
    for doc in tokenized_docs:
        for term in doc:
            vocabulary.add(term)
    vocabulary = list(vocabulary)
    dictionary = {}
    for i in range(len(vocabulary)):
        dictionary.setdefault(i, vocabulary[i])
    return dictionary

def vectorize_docs(docs_dict, terms_dict, tf_idf):
    """ transform documents to vectors
        using bag-of-words model and if-idf
    """
    docs_vectors = np.zeros([len(docs_dict), len(terms_dict)])

    for doc_id in docs_dict:
        for term_id in terms_dict:
            if terms_dict[term_id] in tf_idf[doc_id]:
                docs_vectors[doc_id][term_id] = tf_idf[doc_id][terms_dict[term_id]]
    return docs_vectors

def vectorize_query(tokenized_query, terms_dict):
    """ transform user query to vectors 
        using bag-of-words model and vector normalization
    """
    query_vector = np.zeros(len(terms_dict))
    for term_id in terms_dict:
        if terms_dict[term_id] in tokenized_query:
            query_vector[term_id] += 1
    return query_vector / np.linalg.norm(query_vector)

def cos_similarity(vector1, vector2):
    """compute cosine similarity of two vectors"""
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2))) 

def compute_simmilarity(docs_vectors, query_vector, docs_dict):
    """compute all similarites between user query and all documents"""
    similarities = {}
    for doc_id in docs_dict:
        similarities[doc_id] = cos_similarity(docs_vectors[doc_id], query_vector)
    return similarities


if __name__ == '__main__':
    tokenized_docs = [
        ['hello', 'world'],
        ['hello', 'python'],
        ['i', 'love', 'c', 'java', 'python', 'typescript', 'and', 'php'],
        ['use', 'python', 'to', 'build', 'inverted', 'indices'],
        ['you', 'and', 'me', 'are', 'in', 'one', 'world']
                    ]
    tokenized_query = ["python", "indices"]
    docs_dict = {
        0: "docs[0]",
        1: "docs[1]",
        2: "docs[2]",
        3: "docs[3]",
        4: "docs[4]"
    }
    indices = {'and': [2, 4], 'are': [4], 'build': [3], 'c': [2], 'hello': [0, 1], 'i': [2], 
            'in': [4], 'indices': [3], 'inverted': [3], 'java': [2], 'love': [2], 'me': [4],
            'one': [4], 'php': [2], 'python': [1, 2, 3], 'to': [3], 'typescript': [2], 'use'
            : [3], 'world': [0, 4], 'you': [4]}
    tf_idf = tfidf(tokenized_docs, indices)
    terms_dict = build_terms_dictionary(tokenized_docs);
    docs_vectors = vectorize_docs(docs_dict, terms_dict, tf_idf)
    query_vector = vectorize_query(tokenized_query, terms_dict)
    # pprint(docs_vectors)
    pprint(compute_simmilarity(docs_vectors, query_vector, docs_dict))

