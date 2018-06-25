
# coding: utf-8

# In[2]:


import numpy as np
import six
import sys
import os
import traceback
import re
import pickle


from chainer import cuda
from context2vec.common.context_models import Toks
from context2vec.common.model_reader import ModelReader
import sklearn
import pandas as pd
import logging
from scipy.stats import spearmanr
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim


# In[10]:


# def space_punc(sent):
#     new_sent=sent
#     add=0
#     for m in re.finditer(r'[\d\-\/\"\';,.!?\)\(]+',sent):
#         if m.end(0)+add>=len(new_sent):
#             return new_sent[:m.start(0)+add]+' '+new_sent[m.start(0)+add:m.end(0)+add]
#         else:
#             new_sent=new_sent[:m.start(0)+add]+' '+new_sent[m.start(0)+add:m.end(0)+add]+' '+new_sent[m.end(0)+add:]
#             add+=2
#     return new_sent
        
# def mult_sim(w, target_v, context_v):
#     target_similarity = w.dot(target_v)
#     target_similarity[target_similarity<0] = 0.0
#     context_similarity = w.dot(context_v)
#     context_similarity[context_similarity<0] = 0.0
#     return (target_similarity * context_similarity)


# def find_unk_pos(words,test_w):
#     try:
#         pos=words.index(test_w.upper())
#     except ValueError:
#         try:
#             pos=words.index(test_w.upper()+'S')
#         except ValueError as e:
#             try:
#                 pos=words.index(test_w.upper()+'ES')
#             except ValueError as e:
#                 try:
#                     pos=words.index(test_w.upper()[:-1]+'IES')
#                 except ValueError as e:
#                     try:
#                         pos=words.index(test_w.upper()+test_w.upper()[-1]+'ES')
#                     except ValueError as e:
#                         print (e)
#     return pos

def produce_top_n_simwords(w_filter,context_embed,n_result,index2word):
        # compute top n_result similarity weights

        print('producing top {0} simwords'.format(n_result))
        similarity = (w_filter.dot(context_embed)+1.0)/2
        top_words_i=[]
        count = 0
        for i in (-similarity).argsort():
                    if xp.isnan(similarity[i]):
                        continue
                    print('{0}: {1}'.format(str(index2word[i]), str(similarity[i])))
                    count += 1
                    top_words_i.append(i)
                    if count == n_result:
                        break

        top_vec=w_filter[top_words_i,:]
        return top_vec
def skipgram_context(model,words,pos):
    context_wvs=[]
    for i,word in enumerate(words):
        if i != pos:
            try:
                context_wvs.append(model[word])
                
            except KeyError as e:
                print ('==warning==: key error in context {0}'.format(e))
    context_embed=sum(np.array(context_wvs))/len(context_wvs)
    return len(context_wvs),context_embed
    
def context_inform(test_s,test_w, model,model_type,n_result,w_filter,index2word,weight):
    test_s=test_s.replace(test_w, ' '+test_w+' ')

    #test_s=space_punc(test_s)
    print(test_s)
    words=test_s.split()
    pos=words.index(test_w)
    
    score=1.0
    #decide on model
    if model_type=='context2vec':
        context_embed= model.context2vec(words, pos)
        context_embed = context_embed / xp.sqrt((context_embed * context_embed).sum())
    
    elif model_type=='skipgram':
        score,context_embed=skipgram_context(model,words,pos)
        context_embed = context_embed / xp.sqrt((context_embed * context_embed).sum())
        
    else:
        print ('model type {0} not recognized'.format(model_type))
        sys.exit(1)
        
        
    #decide on weight 
    if weight=='top_mutual_sim':
        top_vec=produce_top_n_simwords(w_filter,context_embed,n_result,index2word)
        score=sum(sum(top_vec.dot(top_vec.T)))/(n_result**2)
    elif weight=='learned':
        print ('learned not implemented')
    elif weight=='gaussian':
        print ('gaussian not implemented')
    elif weight ==False:
        score=score
    else:
        print ('weight mode {0} not recognized'.format(weight))
    return score,context_embed

def additive_model(test_ss,test_w, model_type,model,n_result,w_filter,index2word,weight=False):
    context_out=[]
    context_weights=[]
    for test_s in test_ss.split('@@'):
        test_s=test_s.strip()
        score,context_embed=context_inform(test_s,test_w, model,model_type,n_result,w_filter,index2word,weight)
        print ('weight is {0}'.format(score))
        context_out.append(context_embed)
        context_weights.append(score)
    context_out=np.array(context_out)
    norm_weights=np.array(context_weights).reshape(len(context_weights),1)/float(sum(context_weights))
    print ('normalized weight: \n  {0}'.format(norm_weights))
    context_avg=sum(norm_weights*context_out)
    print('producing top {0} words for new embedding'.format(n_result))
    top_vec=produce_top_n_simwords(w_filter,context_avg,n_result,index2word)
     
    return context_avg

def filter_w(w,word2index,index2word,word_freq_f):
    #filter out words with freq less than 200, words with no letters in, and stopwords
    stopw=stopwords.words('english')
    stopw=[word.encode('utf-8') for word in stopw]
    index2word_filter={}
    word2index_filter={}
    index_filter2index=[]
    counter=0
    with open(word_freq_f) as f:
        for line in f:
            f_w=line.split()
            if len(f_w)>1:
                if f_w[1] in word2index and re.search('[a-zA-Z]',f_w[1])!=None and f_w[1] not in stopw :
                    #word2freq_nostop[f_w[1]]=f_w[0]
                    index_filter2index.append(word2index[f_w[1]])
                    word2index_filter[f_w[1]]=counter
                    index2word_filter[counter]=f_w[1]
                    counter+=1
    w_filter= w[index_filter2index,:]
    return w_filter,word2index_filter,index2word_filter

def rm_stopw_context(model):
    stopw=stopwords.words('english')
    stopw=[word.encode('utf-8') for word in stopw]
    
    model={word:model.wv.__getitem__(word) for word in model.wv.vocab if word not in stopw}
    return model


def eval_chimera(chimeras_data_dir,num_sent,context_model,model_type,n_result,w,index2word,weight=False):
    golds=[]
    model_predict=[]
    num_indic='l'+str(num_sent)
    for root, subdir, fname in os.walk(chimeras_data_dir):
        for fn in fname:
            if fn.endswith('fixed.test.txt')and 'l'+str(num_sent)==fn.split('.')[1]: #read in the test file
                print (fn)
                data=pd.read_csv(os.path.join(chimeras_data_dir,fn),delimiter='\t')
                for index, row in data.iterrows():
                    
                    #compute context representation
                    if weight!='learned':
                        context_avg=additive_model(row[1],'___', model_type,context_model,n_result,w,index2word,weight)
                   
                    
                    #cosine similarity with probe embedding
                    for gold,probe in zip(row[3].split(','),row[2].split(',')):
                        try:
                            cos=w[word2index[probe]].dot(context_avg)
                            if xp.isnan(cos):
                                continue
                            else:
                                model_predict.append(cos)
                                golds.append(gold)
                        except KeyError as e:
                            print ("====warning key error for probe=====: {0}".format(e))
    if len(golds)==len(model_predict):
        print ('spearman correlation is {0}'.format(spearmanr(golds,model_predict)))
    else:
        print ('unequal length: gold {0}, model {1}'.format( len(golds),len(model_predict)))


    


# In[25]:


import gensim



# In[14]:


{key: model.wv.vocab[key].index for key in model.wv.vocab}


# In[29]:


model = gensim.models.Word2Vec.load('../models/wiki_all.model/wiki_all.sent.split.model')
stopw=stopwords.words('english')
stopw=[word.encode('utf-8') for word in stopw]
model={word:model.wv.__getitem__(word) for word in model.wv.vocab if word not in stopw}
model['it']


# In[34]:



if __name__=="__main__":
    
    #params read in
    if sys.argv[0]=='/usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py':
#         model_param_file='../models/context2vec/model_dir/context2vec.ukwac.model.params'
#         model_type='context2vec'
        
        model_param_file='../models/wiki_all.model/wiki_all.sent.split.model'
        model_type='skipgram'
        weight='top_mutual_sim'
        context_rm_stopw=1
        data='../data/data-chimeras'
    else:
        if len(sys.argv) < 3:
            print >> sys.stderr, "Usage: %s <model_param_file> <model_type> <weight> <context_rm_stop>"  % (sys.argv[0])
            sys.exit(1)

        model_param_file = sys.argv[1]
        model_type=sys.argv[2]
        weight=int(sys.argv[3])
        context_rm_stop=int.argv[4]
        data =argv[5]
    
    
    #gpu setup 
    gpu = -1 # todo: make this work with gpu

    if gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()    
    xp = cuda.cupy if gpu >= 0 else np
    
    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    
    #choose model type
    print ('read model....')
    if model_type=='context2vec':
        #read in model
        
        model_reader = ModelReader(model_param_file)
        w = model_reader.w
        index2word = model_reader.index2word
        word2index=model_reader.word2index
        model = model_reader.model
    elif model_type=='skipgram':
        model = gensim.models.Word2Vec.load(model_param_file)
        w=model.wv.vectors
        #vector normalize for probe w embedding
        s = np.sqrt((w * w).sum(1))
        s[s==0.] = 1.
        w /= s.reshape((s.shape[0], 1))
        
        index2word=model.wv.index2word
        word2index={key: model.wv.vocab[key].index for key in model.wv.vocab}
        
        
    #weight
    if weight=='top_mutual_sim':
        #filter w : remove stop words and low freq words 
        print ('filter words for target....')
        w,word2index,index2word=filter_w(w,word2index,index2word,'word_freq')
        n_result = 20
        
    # remove context stop words
    if int(context_rm_stopw)==1:
        print ('filter words for context....')

        model=rm_stopw_context(model)
        
    


# In[33]:


model['it']


# In[38]:


#read in data
if data == '../data/data-chimeras':
        eval_chimera(data,6,model,model_type,n_result,w,index2word)

