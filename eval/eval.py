
# coding: utf-8

# In[1]:


import numpy as np
import six
import sys
import os
import traceback
import re
import pickle
from copy import deepcopy

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
import math




# In[5]:


def produce_top_n_simwords(w_filter,context_embed,n_result,index2word):
        #assume that w_filter is already normalized
        context_embed = context_embed / xp.sqrt((context_embed * context_embed).sum())
        similarity_scores=[]
        print('producing top {0} simwords'.format(n_result))
        similarity = (w_filter.dot(context_embed)+1.0)/2
        top_words_i=[]
        top_words=[]
        count = 0
        for i in (-similarity).argsort():
                    if xp.isnan(similarity[i]):
                        continue
                    print('{0}: {1}'.format(str(index2word[i]), str(similarity[i])))
                    count += 1
                    top_words_i.append(i)
                    top_words.append(index2word[i])
                    similarity_scores.append(similarity[i])
                    if count == n_result:
                        break

        top_vec=w_filter[top_words_i,:]
        
        return top_vec,np.array(similarity_scores),top_words
    
def top_mutual_sim(top_vec,similarity_scores):

    #normalize the top_vec
    s = np.sqrt((top_vec * top_vec).sum(1))
    s[s==0.] = 1.
    top_vec /= s.reshape((s.shape[0], 1))
    
    # substitutes' similarity to sentence (similarity_scores) as weight matrix to mutual similarity
    max_score=similarity_scores[0]
    similarity_scores=np.array(similarity_scores)
    sim_weights=(similarity_scores+similarity_scores.reshape(len(similarity_scores),1))/2.0
    #weighted by the maximum score in the substitutes (highre max score means the context is more certain about the substitutes)
    sim_weights=(sim_weights/float(sum(sum(sim_weights))))*max_score
    # dot product weighted by substitute probability (sim_weights)
    inf_score=sum(sum(top_vec.dot(top_vec.T)*sim_weights))
    return inf_score


# In[43]:


def load_w2salience(w2salience_f,weight_type):
    w2salience={}
    with open(w2salience_f) as f:
        for line in f:
            if line.strip()=='':
                continue
            w,w_count,s_count=line.strip().split('\t')
            if weight_type==INVERSE_W_FREQ:
                w2salience[w]=1/float(w_count)
            elif weight_type==INVERSE_S_FREQ:
                w2salience[w]=math.log(1+84755431/float(s_count))
    return w2salience

def skipgram_context(model,words,pos,weight=None,w2entropy=None):
    context_wvs=[]
    weights=[]
    for i,word in enumerate(words):
        if i != pos: #surroudn context words
            try:
                if weight ==LDA:
                    if word in w2entropy and word in model:
                        print (word,w2entropy[word])
                        weights.append(1/(w2entropy[word]+1.0))
                        context_wvs.append(model[word])
                elif weight in [INVERSE_W_FREQ,INVERSE_S_FREQ]:
                    if word in w2entropy and word in model:
                        print (word,w2entropy[word])
                        weights.append(w2entropy[word])
                        context_wvs.append(model[word])
                else:
                    #equal weights per word
                    context_wvs.append(model[word])
                    weights.append(1.0)
            except KeyError as e:
                print ('==warning==: key error in context {0}'.format(e))
    context_embed=sum(np.array(context_wvs)*np.array(weights).reshape(len(weights),1))#/sum(weights)
    return sum(weights),context_embed #  will be normalized later

def lg_model_out_w2v(top_words,w_target,word2index_target):
        # lg model substitutes in skipgram embedding
        top_vec=[]
        index_list=[]
        for i,word in enumerate(top_words):
            try :
                top_vec.append(w_target[word2index_target[word]])
                index_list.append(i)
            except KeyError as e:
                print (e)
        return np.array(top_vec),index_list
    
def context_inform(test_s,test_w, model,model_type,n_result,w_filter,index2word,weight,w2entropy=None,w_target=None,word2index_target=None,index2word_target=None):
    #produce context representation and infromative score for each context
    test_s=test_s.replace(test_w, ' '+test_w+' ')
    print(test_s)
    words=test_s.split()
    pos=words.index(test_w)
    
    score=1.0 #default score
    
    # Decide on the model
    if model_type=='context2vec':
        context_embed= model.context2vec(words, pos)
        context_embed_out=context_embed
    
    elif model_type=='skipgram':
        score,context_embed=skipgram_context(model,words,pos,weight,w2entropy)
        context_embed_out=context_embed
        
    elif model_type=='context2vec-skipgram':
        # context2vec substitutes in skipgram space
        context_embed= model.context2vec(words, pos)
        top_vec,sim_scores,top_words=produce_top_n_simwords(w_filter,context_embed,n_result,index2word)
        top_vec,index_list=lg_model_out_w2v(top_words,w_target,word2index_target) 
        sim_scores=sim_scores[index_list] #weighted by substitute probability
        context_embed_out=sum(top_vec*((sim_scores/sum(sim_scores)).reshape(len(sim_scores),1)))
    else:
        print ('model type {0} not recognized'.format(model_type))
        sys.exit(1)
        
        
    #decide on weight per sentence
    if weight==TOP_MUTUAL_SIM:
        print (weight)
#         if word2index_target==None:
            #context2vec word embedding space neighbours
        top_vec,sim_scores,top_words=produce_top_n_simwords(w_filter,context_embed,n_result,index2word)
        #skipgram word embedding space neighbours when context2vec-skipgram
        score=top_mutual_sim(top_vec,sim_scores)
        print (score)

    elif weight=='learned':
        print ('learned not implemented')
    elif weight=='gaussian':
        print ('gaussian not implemented')
    elif weight ==False or weight in [LDA,INVERSE_S_FREQ,INVERSE_W_FREQ]:
        score=score
    else:
        print ('weight mode {0} not recognized'.format(weight))
    return score,context_embed_out

def additive_model(f_w,test_ss,test_w, model_type,model,n_result,w_filter,index2word,weight=False,w2entropy=None,w_target=None,word2index_target=None,index2word_target=None):
    #produce context representation across contexts using weighted average
    
    context_out=[]
    context_weights=[]
    for test_s in test_ss.split('@@'):
        test_s=test_s.strip()
        #produce context representation with scores
        score,context_embed=context_inform(test_s,test_w, model,model_type,n_result,w_filter,index2word,weight,w2entropy,w_target,word2index_target,index2word_target)
        print ('weight is {0}'.format(score))
        context_out.append(context_embed)
        context_weights.append(score)
    
    
    #sum representation across contexts
    context_out=np.array(context_out)
    norm_weights=np.array(context_weights).reshape(len(context_weights),1)/float(sum(context_weights))
    f_w.write(','.join([str(i[0]) for i in norm_weights])+'\n')
    print ('normalized weight: \n  {0}'.format(norm_weights))
    
    if model_type=='skipgram':
        # context representation by weighted sum of all context words in all contexts
        context_avg=sum(context_out)/sum(context_weights)
    else:
        # context represenatation by weighted sum of contexts
        context_avg=sum(norm_weights*context_out)
    
    
    # check new embedding neighbours

    print('producing top {0} words for new embedding'.format(n_result))
    if index2word_target==None:
        top_vec,scores,top_words=produce_top_n_simwords(w,context_avg,n_result,index2word)
    else:
        #print the target space neighbours for context2vec-skipgram
        print (w_target.shape)
        top_vec,scores,top_words=produce_top_n_simwords(w_target,context_avg,n_result,index2word_target)
    
    return context_avg




# In[6]:


def filter_w(w,word2index,index2word):
    #filter out words with no letters in, and stopwords
    stopw=stopwords.words('english')
    stopw=[word.encode('utf-8') for word in stopw]
    index2word_filter={}
    word2index_filter={}
    index_filter2index=[]
    counter=0
    for word in word2index:
            if word not in stopw:
                    index_filter2index.append(word2index[word])
                    word2index_filter[word]=counter
                    index2word_filter[counter]=word
                    counter+=1
    w_filter= w[index_filter2index,:]
    return w_filter,word2index_filter,index2word_filter

def rm_stopw_context(model):
    stopw=stopwords.words('english')
    stopw=[word.encode('utf-8') for word in stopw]
    
    model={word:model.wv.__getitem__(word) for word in model.wv.vocab if word not in stopw}
    return model




# In[7]:


def eval_chimera(chimeras_data_f,context_model,model_type,n_result,w,index2word,weight=False,w2entropy=None,w_target=None,word2index_target=None,index2word_target=None):
    chimeras_data_dir='/'.join(chimeras_data_f.split('/')[:-1])
    num_sent=chimeras_data_f.split('/')[-1].split('.')[1][1]
    print (chimeras_data_dir)
    print (num_sent)
    with open(chimeras_data_dir+'/weights_{0}_{1}_{2}'.format(num_sent,model_type,str(weight)),'w') as f_w:
        spearmans=[]
        data=pd.read_csv(os.path.join(chimeras_data_f),delimiter='\t',header=None)

        for index, row in data.iterrows():
            golds=[]
            model_predict=[]
            probes=[]
            #compute context representation
            if weight!='learned':
                context_avg=additive_model(f_w,row[1].lower(),'___', model_type,context_model,n_result,w,index2word,weight,w2entropy,w_target,word2index_target,index2word_target)
            context_avg = context_avg / xp.sqrt((context_avg * context_avg).sum())

            #cosine similarity with probe embedding
            for gold,probe in zip(row[3].split(','),row[2].split(',')):
                try:
                    if index2word_target==None:
                        probe_w_vec=xp.array(w[word2index[probe]])
                    else:
                        probe_w_vec=xp.array(w_target[word2index_target[probe]])
                    probe_w_vec=probe_w_vec/xp.sqrt((probe_w_vec*probe_w_vec).sum())
                    cos=probe_w_vec.dot(context_avg)
                    if xp.isnan(cos):
                        continue
                    else:
                        model_predict.append(cos)
                        golds.append(gold)
                        probes.append(probe)
                except KeyError as e:
                    print ("====warning key error for probe=====: {0}".format(e))
            print ('probes',probes)
            print ('gold',golds)
            print ('model_predict',model_predict)
            sp=spearmanr(golds,model_predict)[0]
            print ('spearman correlation is {0}'.format(sp))
            if not math.isnan(sp):
                spearmans.append(sp)
        print ("AVERAGE RHO:",float(sum(spearmans))/float(len(spearmans)))


# In[8]:


# a=np.array([1,2,3])
# b=a.reshape(len(a),1)
# c=(a+b)/2.0
# c=c/(sum(sum(c)))
# # print(sum(sum(c)))
# d=np.array([[1,1,1],[2,2,2],[3,3,3]])
# e=d.dot(d.T)
# print ('e',e)
# print ('c',c)
# e*c
# a
# context_embed= model.context2vec(['cats',',','dogs',',','snakes','are','animals'],2 )
# context_embed = context_embed / xp.sqrt((context_embed * context_embed).sum())
# produce_top_n_simwords(context_embed=context_embed,index2word=index2word,w_filter=w,n_result=20)


# In[47]:


TOP_MUTUAL_SIM='top_mutual_sim'
LDA='lda'
INVERSE_S_FREQ='inverse_s_freq'
INVERSE_W_FREQ='inverse_w_q'
WEIGHT_DICT={0:False,1:TOP_MUTUAL_SIM,2:LDA,3:INVERSE_S_FREQ,4:INVERSE_W_FREQ}


if __name__=="__main__":
    
    #params read in
    if sys.argv[0]=='/usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py':
        
        data='./eval_data/data-chimeras/dataset.l4.fixed.test.txt.punct'

        weight=WEIGHT_DICT[1]
        
#         ##context2vec
##         model_param_file='../models/context2vec/model_dir/context2vec.ukwac.model.params'
#         model_param_file='../models/context2vec/model_dir/MODEL-wiki.params.14'
        
#         model_type='context2vec'
#         context_rm_stopw=0

# ####skipgram
#         model_param_file='../models/wiki_all.model/wiki_all.sent.split.model'
#         model_type='skipgram'
#         context_rm_stopw=1
#         weight='inverse_w_freq'
#         w2salience_f='../corpora/corpora/wiki.all.utf8.sent.split.tokenized.vocab'
#         w2salience_f='../models/lda/w2entropy'


####context2vec-skipgram
#         model_param_file='../models/context2vec/model_dir/MODEL-wiki.params.14,../models/wiki_all.model/wiki_all.sent.split.model'
        model_param_file='../models/context2vec/model_dir/context2vec.ukwac.model.params,../models/wiki_all.model/wiki_all.sent.split.model'
        model_type='context2vec-skipgram'
        context_rm_stopw=0
    
    else:
        if len(sys.argv) < 6:
            print >> sys.stderr, "Usage: %s <model_param_file> <model_type> <weight:0:False,1:'top_mutual_sim',2:'lda',3:'inverse_s_freq',4:'inverse_w_freq'> <context_rm_stop> <eval_data> <w2salience>"  % (sys.argv[0])
            sys.exit(1)
        
        model_param_file = sys.argv[1]
        model_type=sys.argv[2]
        
        weight=WEIGHT_DICT[int(sys.argv[3])]
        context_rm_stopw=int(sys.argv[4])
        data =sys.argv[5]
        if len(sys.argv)>6:
            w2salience_f=argv[6]
        else:
            w2salience_f=None
    
    
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
        w_target=None
        word2index_target=None
        index2word_target=None
    elif model_type=='skipgram':
        model = gensim.models.Word2Vec.load(model_param_file)
        w=deepcopy(model.wv.vectors)
        #vector normalize for probe w embedding
        s = np.sqrt((w * w).sum(1))
        s[s==0.] = 1.
        w /= s.reshape((s.shape[0], 1))
        
        index2word=model.wv.index2word
        word2index={key: model.wv.vocab[key].index for key in model.wv.vocab}
        w_target=None
        word2index_target=None
        index2word_target=None
        
        
    elif model_type=='context2vec-skipgram':
        model_param_context,model_param_w2v=model_param_file.split(',')
        model_reader = ModelReader(model_param_context)
        w = model_reader.w
        index2word = model_reader.index2word
        word2index=model_reader.word2index
        model = model_reader.model
        
        model_w2v = gensim.models.Word2Vec.load(model_param_w2v)
        w_target=model_w2v.wv.vectors
        index2word_target=model_w2v.wv.index2word
        word2index_target={key: model_w2v.wv.vocab[key].index for key in model_w2v.wv.vocab}
    
    
    w2salience=None
    
    #remove stop words in target word space
    print ('filter words for target....')
    w,word2index,index2word=filter_w(w,word2index,index2word)
    if  index2word_target!=None:
        w_target,word2index_target,index2word_target=filter_w(w_target,word2index_target,index2word_target)
    
    #weight

    if weight==TOP_MUTUAL_SIM:
        n_result = 20
    elif weight==LDA:
        print ('load vectors and entropy')
        w2salience=pickle.load(open(w2salience_f))
    elif weight==INVERSE_W_FREQ:
        print ('load w2freq')
        w2salience=load_w2salience(w2salience_f,weight)
    elif weight==INVERSE_S_FREQ:
        print ('load w2freq')
        w2salience=load_w2salience(w2salience_f,weight)


    # remove context stop words
    if int(context_rm_stopw)==1:
        print ('filter words for context....')

        model=rm_stopw_context(model)
        
    


# In[51]:


#read in data
# data='./eval_data/data-chimeras/dataset.l6.fixed.test.txt.punct'
if data.split('/')[-2]== 'data-chimeras':
#         weight=None
        eval_chimera(data,model,model_type,20,w,index2word,weight,w2salience,w_target,word2index_target,index2word_target)
        

