
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
import collections
import argparse
import h5py
from collections import defaultdict
from gensim.models import KeyedVectors


# In[2]:


# def load_vectors(vectorfile, dim=300, skip=False):
#   '''loads word vectors from a text file
#   Args:
#     vectorfile: string; vector file name
#     dim: int; dimensions of the vectors
#     skip: boolean; whether or not to skip the first line (for word2vec)
#   Returns:
#     generator of (string, numpy array); word and its embedding
#   '''
#   with open(vectorfile, 'r') as f:
#     for line in f:
#       if skip:
#         skip = False
#       else:
#         index = line.index(' ')
#         word = line[:index]
#         yield word, np.array([FLOAT(entry) for entry in line[index+1:].split()[:dim]])

def produce_top_n_simwords(w_filter,context_embed,n_result,index2word,debug=False):
        #assume that w_filter is already normalized
        context_embed = context_embed / xp.sqrt((context_embed * context_embed).sum())
        similarity_scores=[]
#         print('producing top {0} simwords'.format(n_result))
        similarity = (w_filter.dot(context_embed)+1.0)/2
        top_words_i=[]
        top_words=[]
        count = 0
        for i in (-similarity).argsort():
                    if xp.isnan(similarity[i]):
                        continue
                    if debug==True:
                        print('{0}: {1}'.format(str(index2word[int(i)]), str(similarity[int(i)])))
                    count += 1
                    top_words_i.append(int(i))
                    top_words.append(index2word[int(i)])
                    similarity_scores.append(float(similarity[int(i)]))
                    if count == n_result:
                        break

        top_vec=w_filter[top_words_i,:]
        return top_vec,xp.array(similarity_scores),top_words

def top_cluster_density(top_vec,similarity_scores):
    #normalize the top_vec
    s = xp.sqrt((top_vec * top_vec).sum(1))
    s[s==0.] = 1.
    top_vec = top_vec/ s.reshape((s.shape[0], 1))
    
    #perform the centroid
    max_score=similarity_scores[0]
    similarity_scores=xp.array(similarity_scores).reshape(len(similarity_scores),1)/sum(similarity_scores)
    centroid_vector=sum(top_vec*similarity_scores)
    # average of cosine distance to the centroid,weighted by max scores
    inf_score=float(sum(top_vec.dot(centroid_vector))/len(top_vec)*max_score)
    return inf_score


# In[25]:


def load_w2salience(model_w2v,w2salience_f,weight_type):
    w2salience={}
    with open(w2salience_f) as f:
        for line in f:
            line=line.strip()
            if line=='':
                continue
            if line.startswith('sentence total'):
                sent_total=int(line.split(':')[1])
                continue
            w,w_count,s_count=line.split('\t')
            if model_w2v.wv.__contains__(w):
                if weight_type==INVERSE_W_FREQ:
                    w2salience[w]=1/float(w_count)
                elif weight_type==INVERSE_S_FREQ:
                    w2salience[w]=math.log(1+sent_total/float(s_count))
    #                 w2salience[w]=math.log(1+84755431/float(s_count))
    return w2salience

def skipgram_context(model,words,pos,weight=None,w2entropy=None):
    context_wvs=[]
    weights=[]
    for i,word in enumerate(words):
        if i != pos: #surroudn context words
            if word in model:
                if weight ==LDA:
                    if word in w2entropy:
#                         print (word,w2entropy[word])
                        weights.append(1/(w2entropy[word]+1.0))
                        context_wvs.append(model[word])
        
                elif weight in [INVERSE_W_FREQ,INVERSE_S_FREQ]:
                    if word in w2entropy:
                        model[word]
                        weights.append(w2entropy[word])
                        context_wvs.append(model[word])
                else:
                    #equal weights per word
                    context_wvs.append(model[word])
                    weights.append(1.0)
            else:
                print ('==warning==: key error in context {0}'.format(word))
#     print ('per word weights',weights)
    context_embed=sum(np.array(context_wvs)*np.array(weights).reshape(len(weights),1))#/sum(weights)
#     print ('skipgram context sum:', context_embed[:10])
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
                print ('lg subs {0} not in w2v'.format(e))
        if top_vec==[]:
            print ('no lg subs in w2v space')
            return xp.array([]),[]
        else:
            return xp.stack(top_vec),index_list
    
def context_inform(test_s,test_w, model,model_type,n_result,w_filter,index2word,weight,w2entropy=None,w_target=None,word2index_target=None,index2word_target=None):
    #produce context representation and infromative score for each context
    test_s=test_s.replace(test_w, ' '+test_w+' ')
#     print(test_s)
    words=test_s.split()
    pos=words.index(test_w)

    score=1.0 #default score
    
    # Decide on the model
    if model_type=='context2vec':
            context_embed= model.context2vec(words, pos)

    elif model_type=='skipgram':
        score,context_embed=skipgram_context(model,words,pos,weight,w2entropy)
        context_embed_out=xp.array(context_embed)
        
    elif model_type=='context2vec-skipgram':
        context_embed= model.context2vec(words, pos)

        top_vec,sim_scores,top_words=produce_top_n_simwords(w_filter,context_embed,n_result,index2word)
        top_vec,index_list=lg_model_out_w2v(top_words,w_target,word2index_target) 
        sim_scores=sim_scores[index_list] #weighted by substitute probability
        if weight==SUBSTITUTE_PROB:
            context_embed_out=xp.array(sum(top_vec*sim_scores.reshape(len(sim_scores),1)))
        else:
            context_embed_out=xp.array(sum(top_vec*((sim_scores/sum(sim_scores)).reshape(len(sim_scores),1))))
        
    else:
        print ('model type {0} not recognized'.format(model_type))
        sys.exit(1)
        
    #decide on weight per sentence
    if weight==TOP_MUTUAL_SIM:
#         if word2index_target==None: #not context2vec-skipgram
#             context2vec word embedding space neighbours
        top_vec,sim_scores,top_words=produce_top_n_simwords(w_filter,context_embed,n_result,index2word)
        #skipgram word embedding space neighbours when context2vec-skipgram
        score=top_mutual_sim(top_vec,sim_scores)
    elif weight==TOP_CLUSTER_DENSITY:
        top_vec,sim_scores,top_words=produce_top_n_simwords(w_filter,context_embed,n_result,index2word)
        score=top_cluster_density(top_vec,sim_scores)
    elif weight==SUBSTITUTE_PROB:
        score=sum(sim_scores)
        print ('substitute prob score',score)
    elif weight=='learned':
        print ('learned not implemented')
    elif weight=='gaussian':
        print ('gaussian not implemented')
    elif weight ==False or weight in [LDA,INVERSE_S_FREQ,INVERSE_W_FREQ]:
        score=score
    else:
        print ('weight mode {0} not recognized'.format(weight))
    return float(score),context_embed_out

def additive_model(test_ss,test_w, model_type,model,n_result,w_filter,index2word,weight=False,w2entropy=None,w_target=None,word2index_target=None,index2word_target=None,f_w=None,context2vec_preembeds=None,scores=None):
    #produce context representation across contexts using weighted average
    print ('model type is :{0}'.format(model_type))
    context_out=[]
    context_weights=[]
    for test_id in range(len(test_ss)):
        test_s=test_ss[test_id]
        test_s=test_s.lower().strip()
        #produce context representation with scores
        if type(context2vec_preembeds)!=type(None):
            context_embed=xp.array(context2vec_preembeds[test_id])
            score=float(scores[test_id])
        else:
            
            score,context_embed=context_inform(test_s,test_w, model,model_type,n_result,w_filter,index2word,weight,w2entropy,w_target,word2index_target,index2word_target)
        
        if score==0 or context_embed.all()==0:
            print ('empty context vector')
           
        else:
            context_out.append(context_embed)
            context_weights.append(score)
    
#     print ('context_weights',context_weights)
    #sum representation across contexts
    if context_out==[]:
        return None
    else:
        context_out=xp.stack(context_out)
    
    if model_type=='skipgram' or weight==SUBSTITUTE_PROB:
        # context representation by weighted sum of all context words in all contexts
#         print ('context out: ', context_out[:10])
#         print ('context_weights',context_weights)
        context_avg=sum(context_out)/sum(context_weights)
    else:
        norm_weights=xp.array(context_weights).reshape(len(context_weights),1)/float(sum(context_weights))
        if f_w!=None:
            f_w.write(','.join([str(i[0]) for i in norm_weights])+'\n')
        # context represenatation by weighted sum of contexts
        
        context_avg=sum(norm_weights*context_out)
    
    
    # check new embedding neighbours

#     print('producing top {0} words for new embedding'.format(n_result))
#     if index2word_target==None:
#         top_vec,scores,top_words=produce_top_n_simwords(w_filter,context_avg,n_result,index2word,debug=False)
#     else:
#         #print the target space neighbours for context2vec-skipgram
#         top_vec,scores,top_words=produce_top_n_simwords(w_target,context_avg,n_result,index2word_target,debug=False)
    return context_avg


def contexts_per_tgw(sents,model_type,context_model,n_result,w,index2word,weight,w2entropy,w_target,word2index_target,index2word_target,context2vec_preembeds=None,scores=None):
    
    if model_type=='context2vec-skipgram?skipgram':
#             
            #context2vevc      
            context_avg_1=additive_model(sents,'___', model_type.split('?')[0],context_model[0],n_result,w[0],index2word[0],weight[0],w2entropy[0],w_target[0],word2index_target[0],index2word_target[0],context2vec_preembeds=context2vec_preembeds,scores=scores)
            #skipgram

            context_avg_2=additive_model(sents,'___', model_type.split('?')[1],context_model[1],n_result,w[1],index2word[1],weight[1],w2entropy[1],w_target[1],word2index_target[1],index2word_target[1])

            if type(context_avg_1)!=type(None) and type(context_avg_2)!=type(None):
                context_avg=(context_avg_1+context_avg_2)/2
#                 print ('context2vec avg embed',context_avg_1[:10])
#                 print ('skipgram avg embed', context_avg_2[:10])
#                 print ('context avg out', context_avg[:10])

            elif type(context_avg_1)!=type(None):
                context_avg=context_avg_1

            elif type(context_avg_2)!=type(None):
                context_avg=context_avg_2
            else:
                context_avg=None
            
    else:

            context_avg=additive_model(sents,'___', model_type,context_model,n_result,w,index2word,weight[0],w2entropy,w_target,word2index_target,index2word_target,context2vec_preembeds=context2vec_preembeds,scores=scores)
#             print ('context avg out', context_avg[:10])
    return context_avg
  
def output_embedding(w,w_target,word2index,word2index_target):
    if model_type=='context2vec-skipgram?skipgram':
        #compute probe embeddings in skipgram space
            w_out=w[1]
            w_target_out=w_target[1]
            word2index_out=word2index[1]
            word2index_target_out=word2index_target[1]
    else:
            w_out=w
            w_target_out=w_target
            word2index_out=word2index
            word2index_target_out=word2index_target
    if word2index_target_out==None:
        return w_out,word2index_out
    else:
        return w_target_out,word2index_target_out


# In[4]:


def filter_w(w,word2index,index2word):
    #filter out words with no letters in, and stopwords
    stopw=stopwords.words('english')
    stopw=[word.encode('utf-8') for word in stopw]
    index2word_filter={}
    word2index_filter={}
    index_filter2index=[]
    counter=0
    for word in word2index:
            if word not in stopw: #and re.search(r'[^a-zA-Z]',word)==None:
                    index_filter2index.append(word2index[word])
                    word2index_filter[word]=counter
                    index2word_filter[counter]=word
                    counter+=1
    w_filter= w[index_filter2index,:]
    return w_filter,word2index_filter,index2word_filter

def rm_stopw_context(model):
    stopw=stopwords.words('english')
    stopw=[word.encode('utf-8') for word in stopw]
    
    model2={word:model.wv.__getitem__(word) for word in model.wv.vocab if word not in stopw}
    return model2




# In[31]:


def cosine(context_avg,probe_w_vec):
    context_avg = context_avg / xp.sqrt((context_avg * context_avg).sum())
    probe_w_vec=probe_w_vec/xp.sqrt((probe_w_vec*probe_w_vec).sum())
    cos=float(probe_w_vec.dot(context_avg))
    print (cos)
    if np.isnan(cos):
        print ('Warning: cos is nan')
        sys.exit(1)
    return cos

def preprocess_nonce(sent,contexts):
    
    sents_out=[]
    
    sent=sent.lower()
    results=re.finditer('___ ',sent)
    matches=[m for m in results]
    for i in range(len(matches)):
        sent_masked=sent
        matches_mask=[(m2.start(0),m2.end(0)) for i2,m2 in enumerate(matches) if i2!=i]
        matches_mask=sorted(matches_mask, key=lambda x:x[0],reverse=True)
        for m in matches_mask:
            sent_masked=sent_masked[:m[0]]+sent_masked[m[1]:]
        sents_out.append(sent_masked+' .')
    return sents_out

def update_mrr(nns,nonce,mrr,ranks):
    rr = 0
    n = 1
    for nn in nns:
        word = nn[0]
        if word == nonce:
            print (word)
            rr = n
            ranks.append(rr)
        else:
            n+=1

    if rr != 0:
        mrr+=float(1)/float(rr)	
    print rr,mrr
    return mrr,ranks

def eval_nonce(nonce_data_f,context_model,model_w2v,model_type,n_result,w,index2word,word2index,weight=False,w2entropy=None,w_target=None,word2index_target=None,index2word_target=None,contexts=None):
        #read in contexts
        ranks = []
        mrr = 0.0
        data=pd.read_csv(os.path.join(nonce_data_f),delimiter='\t',header=None,comment='#')
        
        for index, row in data.iterrows():
            if index>100 and index%100==0:
                print (index)
            sents=preprocess_nonce(row[1],contexts)
            nonce=row[0]
            if nonce not in model_w2v:
                print ('{0} not known'.format(nonce))
                continue
            context_avg=contexts_per_tgw(sents,model_type,context_model,n_result,w,index2word,weight,w2entropy,w_target,word2index_target,index2word_target)
            if xp==cuda.cupy:
                context_avg=xp.asnumpy(context_avg)
                
            # MRR Rank calculation
            nns=model_w2v.similar_by_vector(context_avg,topn=len(model_w2v.wv.vocab))
            mrr,ranks=update_mrr(nns,nonce,mrr,ranks)

        print ("Final MRR: ",mrr,len(ranks),float(mrr)/float(len(ranks)))
        print ('mean: ', np.mean(ranks))
        print ('mediam : {0}'.format(np.median(ranks)))
        return ranks
            
def eval_nonce_withcontexts(nonce_data_f,context_model,model_w2v,model_type,n_result,w,index2word,word2index,weight=False,w2entropy=None,w_target=None,word2index_target=None,index2word_target=None,trials=100):

    data=pd.read_csv(os.path.join(nonce_data_f),delimiter='\t',header=None,comment='#')
    ranks=defaultdict(lambda: defaultdict(list))
    mrrs=defaultdict(lambda: defaultdict(int))
    context2vec_preembeds_all=None
    context2vec_preembeds=None
    scores_all=None
    orders_inf=None
    scores=None
#     start evaluation
    contexts_f=os.path.join(os.path.dirname(nonce_data_f),'contexts')
    
    for index, row in data.iterrows():
        rw=row[0]
        if rw not in model_w2v:
                print ('{0} not known'.format(rw))
                continue
        if not os.path.isfile(os.path.join(contexts_f,rw+'.txt')):
            print ('{0} does not have contexts'.format(rw))
            continue
            
        print ('\n==========processing rareword {0}'.format(rw))

        #definition:
#         sents_def=preprocess_nonce(row[1],contexts)
#         context_avg_def=contexts_per_tgw(sents,model_type,context_model,n_result,w,index2word,weight,w2entropy,w_target,word2index_target,index2word_target)

        #contexts:
        #load sentences
        sents_all=load_sents(contexts_f,rw)
        #load contexts
        if 'context2vec' in model_type.split('?')[0]:
                if model_type=='context2vec-skipgram?skipgram':
                    context2vec_preembeds_all,scores_all=load_contexts(rw,sents_all,context_model[0],model_type.split('?')[0],n_result,w[0],index2word[0],weight[0],w2entropy[0],w_target[0],word2index_target[0],index2word_target[0])
                elif model_type=='context2vec-skipgram':
                    context2vec_preembeds_all,scores_all=load_contexts(rw,sents_all,context_model,model_type.split('?')[0],n_result,w,index2word,weight[0],w2entropy,w_target,word2index_target,index2word_target)
                orders_inf=(-scores_all).argsort()
        #do trials
        for trial in range(trials):
            print ('\n=====Trial no. {0}'.format(trial))
            perm = np.random.permutation(255)
            for logfreq in range(8):
                freq = 2**logfreq
                print ('\ncontext num is {0}'.format(freq))
                context2vec_preembeds, scores,sents=produce_contexts_per_trial(trials,freq,perm,scores_all,context2vec_preembeds_all,orders_inf,sents_all)
                context_avg=contexts_per_tgw(sents,model_type,context_model,n_result,w,index2word,weight,w2entropy,w_target,word2index_target,index2word_target,context2vec_preembeds,scores)
                
                if type(context_avg)!=type(None):
                    if xp==cuda.cupy:
                        context_avg=xp.asnumpy(context_avg)
                
                    # MRR Rank calculation
                    nns=model_w2v.similar_by_vector(context_avg,topn=len(model_w2v.wv.vocab))
                    mrrs[trial][freq],ranks[trial][freq]=update_mrr(nns,rw,mrrs[trial][freq],ranks[trial][freq])
    mrr_res=defaultdict(list)
    median_res=defaultdict(list)
    for trial in mrrs:
        for freq in mrrs[trial]:
            mrr_res[freq].append(float(mrrs[trial][freq])/float(len(ranks[trial][freq])))
            median_res[freq].append(np.median(ranks[trial][freq]))
    print ('{0}\t{1}\t{2}\t{3}\t{4}'.format('freq','MRR','MRR_STD','MEDIAN','MEDIAN_STD'))
    for freq in sorted(mrr_res.keys()):
        print ('{0}\t{1}\t{2}\t{3}\t{4} '.format(freq,np.mean(np.array(mrr_res[freq])),np.std(np.array(mrr_res[freq])),np.mean(np.array(median_res[freq])),np.std(np.array(median_res[freq]))))
    return mrr_res,median_res

def eval_chimera(chimeras_data_f,context_model,model_type,n_result,w,index2word,word2index,weight=False,w2entropy=None,w_target=None,word2index_target=None,index2word_target=None):
    chimeras_data_dir=os.path.dirname(chimeras_data_f)
    num_sent=os.path.basename(chimeras_data_f).split('.')[1][1]
    print (chimeras_data_dir)
    print (num_sent)
    with open(chimeras_data_dir+'/weights_{0}_{1}_{2}'.format(num_sent,model_type,str(weight)),'w') as f_w:
        spearmans=[]
        data=pd.read_csv(os.path.join(chimeras_data_f),delimiter='\t',header=None)
        w_target_out,word2index_target_out=output_embedding(w,w_target,word2index,word2index_target)
        for index, row in data.iterrows():
            if index>100 and index%100==0:
                print (index)
            golds=[]
            model_predict=[]
            probes=[]
            sents=row[1].lower().split('@@')
            #compute context representation
            context_avg=contexts_per_tgw(sents,model_type,context_model,n_result,w,index2word,weight,w2entropy,w_target,word2index_target,index2word_target)
            #cosine similarity with probe embedding
            for gold,probe in zip(row[3].split(','),row[2].split(',')):
                try:
                    probe_w_vec=w_target_out[word2index_target_out[probe]]
                    cos=cosine(context_avg,probe_w_vec)
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


def load_contexts(rw,sents,model,model_type,n_result,w_filter,index2word,weight,w2entropy,w_target,word2index_target,index2word_target):          
    print ('loading contexts..')
    contexts=[]
    scores=[]
    for i,test_s in enumerate(sents):
        test_s=sents[i]
        print (test_s)
        test_s=test_s.replace(TARGET_W, ' '+TARGET_W+' ')
        print(i),
        words=test_s.split()
        pos=words.index(TARGET_W)
        score,context_embed=context_inform(test_s,TARGET_W, model,model_type,n_result,w_filter,index2word,weight,w2entropy,w_target,word2index_target,index2word_target)
        if context_embed.all()==0:
            context_embed=xp.zeros(w_target.shape[-1])
            score=0
        contexts.append(context_embed)
        scores.append(score)
    contexts=xp.stack(contexts,axis=0)
    scores=np.array(scores)
    print ('\ncontexts for {0} completed'.format(rw))
    return contexts,scores

def load_sents(contexts_f,rw):
    sents_all=[]
    with open (os.path.join(contexts_f,rw+'.txt')) as f:
        for line in f:
            line=line.replace(rw,TARGET_W).strip().lower()
            if not line.endswith('.'):
                line=line+' .'
            sents_all.append(line)
    sents_all=np.array(sents_all)
    return sents_all

def produce_contexts_per_trial(trials,freq,perm,scores_all,context2vec_preembeds_all,orders_inf,sents_all):
    if trials==1 and type(scores_all)!=type(None):
        context_inds=orders_inf[:freq]
    else:
        context_inds=perm[np.array(range(freq-1, 2*freq-1)),]

    sents=sents_all[np.array(context_inds),]
    print (context_inds)
    if type(context2vec_preembeds_all)!=type(None):
        context2vec_preembeds=context2vec_preembeds_all[context_inds,]
        scores=scores_all[context_inds,]
        return context2vec_preembeds, scores,sents
    else:
        return None,None,sents
    

def eval_crw_stf(crw_stf_f,model_param_f,context_model,model_type,n_result,w,index2word,word2index,weight=False,w2entropy=None,w_target=None,word2index_target=None,index2word_target=None,trials=100):
    data=pd.read_csv(os.path.join(crw_stf_f),delimiter='\t',header=None,comment='#')
    model_predicts=defaultdict(lambda: defaultdict(list))
    golds=defaultdict(lambda: defaultdict(list))
    context2vec_preembeds_all=None
    context2vec_preembeds=None
    scores_all=None
    orders_inf=None
    scores=None
    rw_prev=''
#     start evaluation
    w_target_out,word2index_target_out=output_embedding(w,w_target,word2index,word2index_target)
    contexts_f=os.path.join(os.path.dirname(crw_stf_f),'context')
    
    for index, row in data.iterrows():
        probe_w=row[0]
        rw=row[1]
        if probe_w not in word2index_target_out: 
            continue
        gold=row[2]
        print ('\n==========processing rareword {0}'.format(rw))

        #load sentences
        sents_all=load_sents(contexts_f,rw)
        #load contexts
        if 'context2vec' in model_type.split('?')[0]:
            if rw_prev!=rw:
                rw_prev=rw
                if model_type=='context2vec-skipgram?skipgram':
                    context2vec_preembeds_all,scores_all=load_contexts(rw,sents_all,context_model[0],model_type.split('?')[0],n_result,w[0],index2word[0],weight[0],w2entropy[0],w_target[0],word2index_target[0],index2word_target[0])
                elif model_type=='context2vec-skipgram':
                    context2vec_preembeds_all,scores_all=load_contexts(rw,sents_all,context_model,model_type.split('?')[0],n_result,w,index2word,weight[0],w2entropy,w_target,word2index_target,index2word_target)
                orders_inf=(-scores_all).argsort()
        #do trials
        for trial in range(trials):
            print ('\n=====Trial no. {0}'.format(trial))
            perm = np.random.permutation(255)
            for logfreq in range(8):
                freq = 2**logfreq
                print ('\ncontext num is {0}'.format(freq))
                context2vec_preembeds, scores,sents=produce_contexts_per_trial(trials,freq,perm,scores_all,context2vec_preembeds_all,orders_inf,sents_all)
                context_avg=contexts_per_tgw(sents,model_type,context_model,n_result,w,index2word,weight,w2entropy,w_target,word2index_target,index2word_target,context2vec_preembeds,scores)
                
                if type(context_avg)!=type(None):
                    #cosine similarity
                    probe_w_vec=w_target_out[word2index_target_out[probe_w]]
                    cos=cosine(context_avg,probe_w_vec)
                    model_predicts[trial][freq].append(cos)
                    golds[trial][freq].append(gold)
     
    #
    sps=defaultdict(list)
    for trial in model_predicts:
        for freq in model_predicts[trial]:
            sp=spearmanr(golds[trial][freq],model_predicts[trial][freq])[0]
            sps[freq].append(sp)
    print ('{0}\t{1}\t{2}'.format('freq','SPEARMAN RANKS MEAN','SPEARMAN RANKS STD'))
    for freq in sorted(sps.keys()):
        print ('{0}\t{1}\t{2} '.format(freq,np.mean(np.array(sps[freq])),np.std(np.array(sps[freq]))))
    return model_predicts,sps

            
            
            


# In[6]:


if __name__=="__main__":
    
    TOP_MUTUAL_SIM='top_mutual_sim'
    TOP_CLUSTER_DENSITY='top_cluster_density'
    LDA='lda'
    TARGET_W='___'
    INVERSE_S_FREQ='inverse_s_freq'
    INVERSE_W_FREQ='inverse_w_q'
    SUBSTITUTE_PROB='substitute_prob'
    WEIGHT_DICT={0:False,1:TOP_MUTUAL_SIM,2:LDA,3:INVERSE_S_FREQ,4:INVERSE_W_FREQ,5:TOP_CLUSTER_DENSITY, 6:SUBSTITUTE_PROB}

    ##### 1. params read in
    if sys.argv[0]=='/usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py':
        
        ###data:
#         data='./eval_data/data-chimeras/dataset_alacarte.l2.fixed.test.txt.punct'
        data='./eval_data/data-nonces/n2v.definitional.dataset.test.txt'
#         data='./eval_data/CRW/CRW-562.txt'
#         weights=[WEIGHT_DICT[5],WEIGHT_DICT[3]]
#         weights=[WEIGHT_DICT[0]]
        gpu=-1
        model_type='skipgram'
        w2salience_f=None
        n_result=20
        trials=100
#         skipgram_model_f='./eval_data/CRW/vectors.txt'
        skipgram_model_f='../models/wiki_all.model/wiki_all.sent.split.model'
        context2vec_model_f='../models/context2vec/model_dir/MODEL-wiki.params.12'
        ######w2salience_f
        w2salience_f='../corpora/corpora/wiki.all.utf8.sent.split.tokenized.vocab'
#         w2salience_f='../corpora/corpora/WWC_norarew.txt.tokenized.vocab'
#         w2salience_f='../models/lda/w2entropy'


        if model_type=='skipgram':
            model_param_file=skipgram_model_f
        elif model_type=='context2vec-skipgram':
            model_param_file=context2vec_model_f+'?'+skipgram_model_f
        elif model_type=='context2vec-skipgram?skipgram':
            model_param_file=context2vec_model_f+'?'+skipgram_model_f

    
    else:
        

        parser = argparse.ArgumentParser(description='Evauate on rare words.')
        parser.add_argument('--f',  type=str,
                            help='model_param_file',dest='model_param_file')
        parser.add_argument('--m', dest='model_type', type=str,
                            help='<model_type: context2vec; context2vec-skipgram (context2vec substitutes in skipgram space); context2vec-skipgram?skipgram (context2vec substitutes in skipgram space plus skipgram context words)>')
        parser.add_argument('--w', dest='weights', type=int, nargs='+',help='<weight:{0}>'.format (sys.argv[0],WEIGHT_DICT.items()))       
        parser.add_argument('--d', dest='data', type=str, help='data file')
        parser.add_argument('--g', dest='gpu',type=int, default=-1,help='gpu, default is -1')
        parser.add_argument('--ws', dest='w2salience_f',type=str, default=None,help='word2salience file, optional')
        parser.add_argument('--n_result',default=20,dest='n_result',type=int,help='top n result for language model substitutes')
        parser.add_argument('--t', dest='trials',type=int, default=100, help='trial number. When trial==1, only test on the most infortive contexts')

        args = parser.parse_args()
        model_param_file = args.model_param_file
        model_type=args.model_type
        n_result=args.n_result
        weights=[WEIGHT_DICT[w_i] for w_i in args.weights]
        trials=args.trials 
        data =args.data
        gpu=args.gpu
        w2salience_f=args.w2salience_f
           
    
    #### 2. gpu setup 
   
    if gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(gpu).use()    
    xp = cuda.cupy if gpu >= 0 else np
    
    
    #### 3. initialize according to model types
    print ('read model....')
    if model_type=='context2vec':
        #read in model
        model_reader = ModelReader(model_param_file,gpu)
        w = xp.array(model_reader.w)
        index2word = model_reader.index2word
        word2index=model_reader.word2index
        model = model_reader.model
        w_target=None
        word2index_target=None
        index2word_target=None
        
    elif model_type=='skipgram':
        if not model_param_file.endswith('txt'):
            model_w2v = gensim.models.Word2Vec.load(model_param_file)
        else:
            model_w2v = KeyedVectors.load_word2vec_format(model_param_file)
            
        w=xp.array(deepcopy(model_w2v.wv.vectors))
        #vector normalize for target w embedding, consistent with context2vec w and convenient for cosine computation among substitutes
        s = xp.sqrt((w * w).sum(1))
        s[s==0.] = 1.
        w /= s.reshape((s.shape[0], 1))
        
        index2word=model_w2v.wv.index2word
        word2index={key: model_w2v.wv.vocab[key].index for key in model_w2v.wv.vocab}
        w_target=None
        word2index_target=None
        index2word_target=None
        print ('filter words for context....')
        model=rm_stopw_context(model_w2v)
        
    elif model_type=='context2vec-skipgram':
        model_param_context,model_param_w2v=model_param_file.split('?')
        model_reader = ModelReader(model_param_context,gpu)
        w = xp.array(model_reader.w)
        index2word = model_reader.index2word
        word2index=model_reader.word2index
        model = model_reader.model
        
        if not model_param_file.endswith('txt'):
            model_w2v = gensim.models.Word2Vec.load(model_param_w2v)
        else:
            model_w2v = KeyedVectors.load_word2vec_format(model_param_w2v)
        w_target=xp.array(model_w2v.wv.vectors)
        index2word_target=model_w2v.wv.index2word
        word2index_target={key: model_w2v.wv.vocab[key].index for key in model_w2v.wv.vocab}
    
    elif model_type=='context2vec-skipgram?skipgram':
        model_param_context,model_param_w2v=model_param_file.split('?')
        #context2vec-skipgram
        model_reader = ModelReader(model_param_context,gpu)
        w = xp.array(model_reader.w)
        index2word = model_reader.index2word
        word2index =model_reader.word2index
        model = model_reader.model
        
        if not model_param_file.endswith('txt'):
            model_w2v = gensim.models.Word2Vec.load(model_param_w2v)
        else:
            model_w2v = KeyedVectors.load_word2vec_format(model_param_w2v)
        w_target=xp.array(model_w2v.wv.vectors)
        index2word_target=model_w2v.wv.index2word
        word2index_target={key: model_w2v.wv.vocab[key].index for key in model_w2v.wv.vocab}
    
        # skigpram
        model_skipgram = model_w2v
        w_skipgram=xp.array(deepcopy(model_skipgram.wv.vectors))
        #vector normalize for probe w embedding for calculating top vector similarity
        s = xp.sqrt((w_skipgram * w_skipgram).sum(1))
        s[s==0.] = 1.
        w_skipgram /= s.reshape((s.shape[0], 1))
        
        index2word_skipgram=model_skipgram.wv.index2word
        word2index_skipgram={key: model_skipgram.wv.vocab[key].index for key in model_skipgram.wv.vocab}
        w_target_skipgram=None
        word2index_target_skipgram=None
        index2word_target_skipgram=None
        
        print ('filter words for context....')
        model_skipgram=rm_stopw_context(model_skipgram)
        
    
    #remove stop words in target word space and asarray for computing CI
    print ('filter words for target....')
    w,word2index,index2word=filter_w(w,word2index,index2word)
    if  index2word_target!=None:
        w_target,word2index_target,index2word_target=filter_w(w_target,word2index_target,index2word_target)
    if model_type=='context2vec-skipgram?skipgram':
        w_skipgram,word2index_skipgram,index2word_skipgram=filter_w(w_skipgram,word2index_skipgram,index2word_skipgram)
    
    #### 4. per word weight
    
    w2salience=None
    for wt in weights:
        if wt==LDA:
            print ('load vectors and entropy')
            w2salience=pickle.load(open(w2salience_f))
        elif wt in [INVERSE_W_FREQ, INVERSE_S_FREQ]:
            print ('load w2freq')
            w2salience=load_w2salience(model_w2v,w2salience_f,wt)
    


    ##### 5. combine parameters for skipgram?context2vec-skipgram
    if model_type=='context2vec-skipgram?skipgram':
        model=(model,model_skipgram)
        w=(w,w_skipgram)
        index2word=(index2word,index2word_skipgram)
        word2index=(word2index,word2index_skipgram)
        w2salience=(w2salience,w2salience)
        w_target=(w_target,w_target_skipgram)
        word2index_target=(word2index_target,word2index_target_skipgram)
        index2word_target=(index2word_target,index2word_target_skipgram)
    
    print (model_param_file,model_type,weights,data,w2salience_f)


# In[32]:


##### 6. read in data and perform evaluation
import time
start_time = time.time()

print (os.path.basename(os.path.split(data)[0]))
if os.path.basename(os.path.split(data)[0])== 'data-chimeras':

        eval_chimera(data,model,model_type,n_result,w,index2word,word2index,weights,w2salience,w_target,word2index_target,index2word_target)

elif os.path.basename(os.path.split(data)[0])== 'data-nonces':
#             ranks=eval_nonce(data,model,model_w2v,model_type,n_result,w,index2word,word2index,weights,w2salience,w_target,word2index_target,index2word_target)
        eval_nonce_withcontexts(data,model,model_w2v,model_type,n_result,w,index2word,word2index,weights,w2salience,w_target,word2index_target,index2word_target,trials)

elif os.path.basename(os.path.split(data)[0])=='CRW':
    
        model_predicts,sps=eval_crw_stf(data,model_param_file,model,model_type,n_result,w,index2word,word2index,weights,w2salience,w_target,word2index_target,index2word_target,trials)
print("--- %s seconds ---" % (time.time() - start_time))



