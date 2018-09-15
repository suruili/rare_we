
# coding: utf-8

# In[138]:


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


# In[105]:


# from sklearn.preprocessing import normalize
# a=np.array([1.23,2.232,3.2])
# print(a/np.sqrt((a*a).sum()))
# print (normalize([a]))


# In[41]:


def produce_top_n_simwords(w_filter,context_embed,n_result,index2word,debug=False):
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
    
def top_mutual_sim(top_vec,similarity_scores):

    #normalize the top_vec
    s = xp.sqrt((top_vec * top_vec).sum(1))
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

def top_cluster_density(top_vec,similarity_scores):
    #normalize the top_vec
    s = xp.sqrt((top_vec * top_vec).sum(1))
    s[s==0.] = 1.
    top_vec = top_vec/ s.reshape((s.shape[0], 1))
    
    #perform the centroid
    max_score=similarity_scores[0]
    similarity_scores=np.array(similarity_scores).reshape(len(similarity_scores),1)/sum(similarity_scores)
    centroid_vector=sum(top_vec*similarity_scores)
    # average of cosine distance to the centroid,weighted by max scores
    inf_score=sum(top_vec.dot(centroid_vector))/len(top_vec)*max_score
    return inf_score


# In[133]:


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
    print ('per word weights',weights)
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
        return xp.stack(top_vec),index_list
    
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
        
    elif model_type=='skipgram':
        score,context_embed=skipgram_context(model,words,pos,weight,w2entropy)
        context_embed_out=xp.array(context_embed)
        
    elif model_type=='context2vec-skipgram':
        # context2vec substitutes in skipgram space
        context_embed= model.context2vec(words, pos)
        top_vec,sim_scores,top_words=produce_top_n_simwords(w_filter,context_embed,n_result,index2word)
        top_vec,index_list=lg_model_out_w2v(top_words,w_target,word2index_target) 
        sim_scores=sim_scores[index_list] #weighted by substitute probability
        print (type(top_vec),type(sim_scores))
        if weight==SUBSTITUTE_PROB:
            context_embed_out=xp.array(sum(top_vec*sim_scores.reshape(len(sim_scores),1)))
        else:
            context_embed_out=xp.array(sum(top_vec*((sim_scores/sum(sim_scores)).reshape(len(sim_scores),1))))

    else:
        print ('model type {0} not recognized'.format(model_type))
        sys.exit(1)
        
    print('context_embed original', context_embed[:10])
    print ('context_embed_out',context_embed_out[:10])
    print ('3',type(context_embed_out))

    #decide on weight per sentence
    print ('weight mode',weight)
    if weight==TOP_MUTUAL_SIM:
#         if word2index_target==None: #not context2vec-skipgram
#             context2vec word embedding space neighbours
        top_vec,sim_scores,top_words=produce_top_n_simwords(w_filter,context_embed,n_result,index2word)
        #skipgram word embedding space neighbours when context2vec-skipgram
        score=top_mutual_sim(top_vec,sim_scores)
    elif weight==TOP_CLUSTER_DENSITY:
#         if word2index_target==None: #not context2vec-skipgram
#             context2vec word embedding space neighbours
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

    return score,context_embed_out

def additive_model(test_ss,test_w, model_type,model,n_result,w_filter,index2word,weight=False,w2entropy=None,w_target=None,word2index_target=None,index2word_target=None,f_w=None):
    #produce context representation across contexts using weighted average
    context_out=[]
    context_weights=[]
    for test_s in test_ss:
        test_s=test_s.strip()
        #produce context representation with scores
        score,context_embed=context_inform(test_s,test_w, model,model_type,n_result,w_filter,index2word,weight,w2entropy,w_target,word2index_target,index2word_target)
        context_out.append(context_embed)
        print ('context_embedtype',type(context_embed))
        context_weights.append(score)
    
    
    print ('context_weights',context_weights)
    #sum representation across contexts
    context_out=xp.stack(context_out)
    
    
    if model_type=='skipgram' or weight==SUBSTITUTE_PROB:
        # context representation by weighted sum of all context words in all contexts
        context_avg=sum(context_out)/sum(context_weights)
    else:
        norm_weights=xp.array(context_weights).reshape(len(context_weights),1)/float(sum(context_weights))
        if f_w!=None:
            f_w.write(','.join([str(i[0]) for i in norm_weights])+'\n')
        print ('normalized weight: \n  {0}'.format(norm_weights))
        # context represenatation by weighted sum of contexts
        context_avg=sum(norm_weights*context_out)
    
    
    # check new embedding neighbours

    print('producing top {0} words for new embedding'.format(n_result))
    if index2word_target==None:
        top_vec,scores,top_words=produce_top_n_simwords(w_filter,context_avg,n_result,index2word,debug=True)
    else:
        #print the target space neighbours for context2vec-skipgram
        top_vec,scores,top_words=produce_top_n_simwords(w_target,context_avg,n_result,index2word_target,debug=True)
    
    return context_avg




# In[134]:


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
    
    model2={word:model.wv.__getitem__(word) for word in model.wv.vocab if word not in stopw}
    return model2




# In[135]:


def preprocess_nonce(sent):
    
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

def eval_nonce(nonce_data_f,context_model,model_w2v,model_type,n_result,w,index2word,word2index,weight=False,w2entropy=None,w_target=None,word2index_target=None,index2word_target=None):
        ranks = []
        mrr = 0.0
        data=pd.read_csv(os.path.join(nonce_data_f),delimiter='\t',header=None,comment='#')
        c = 0
        for index, row in data.iterrows():
            if index>100 and index%100==0:
                print (index)
            sents=preprocess_nonce(row[1])
            nonce=row[0]
            if nonce not in model_w2v:
                print ('{0} not known'.format(nonce))
                continue
            #compute context representation
            if model_type=='context2vec-skipgram?skipgram':
                    #context2vevc                
                    context_avg_1=additive_model(sents,'___', model_type.split('?')[0],context_model[0],n_result,w[0],index2word[0],weight[0],w2entropy[0],w_target[0],word2index_target[0],index2word_target[0])
                    print ('context2vec avg embed',context_avg_1[:10])
                    context_avg_2=additive_model(sents,'___', model_type.split('?')[1],context_model[1],n_result,w[1],index2word[1],weight[1],w2entropy[1],w_target[1],word2index_target[1],index2word_target[1])
                    print ('skipgram avg embed', context_avg_2[:10])
                    context_avg=(context_avg_1+context_avg_2)/2
                    print ('context2vec avg out', context_avg[:10])
                    #compute probe embeddings in skipgram space
                    w_out=w[1]
                    w_target_out=w_target[1]
                    word2index_out=word2index[1]
                    word2index_target_out=word2index_target[1]
                    
            else:
                    
                    context_avg=additive_model(sents,'___', model_type,context_model,n_result,w,index2word,weight,w2entropy,w_target,word2index_target,index2word_target)
                    print ('context avg out', context_avg[:10])
                    w_out=w
                    w_target_out=w_target
                    word2index_out=word2index
                    word2index_target_out=word2index_target
            
            if xp==cuda.cupy:
                context_avg=xp.asnumpy(context_avg)
            print ('vector norm: {0}'.format(np.linalg.norm(context_avg)))
            
            
            # MRR Rank calculation
            nns=model_w2v.similar_by_vector(context_avg,topn=len(model_w2v.wv.vocab))

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
            c+=1
        print ("Final MRR: ",mrr,c,float(mrr)/float(c))

        print ('mediam : {0}'.format(np.median(ranks)))
        return ranks
            


def eval_chimera(chimeras_data_f,context_model,model_type,n_result,w,index2word,word2index,weight=False,w2entropy=None,w_target=None,word2index_target=None,index2word_target=None):
    chimeras_data_dir='/'.join(chimeras_data_f.split('/')[:-1])
    num_sent=chimeras_data_f.split('/')[-1].split('.')[1][1]
    print (chimeras_data_dir)
    print (num_sent)
    with open(chimeras_data_dir+'/weights_{0}_{1}_{2}'.format(num_sent,model_type,str(weight)),'w') as f_w:
        spearmans=[]
        data=pd.read_csv(os.path.join(chimeras_data_f),delimiter='\t',header=None)

        for index, row in data.iterrows():
            if index>100 and index%100==0:
                print (index)
            golds=[]
            model_predict=[]
            probes=[]
            #compute context representation
            sents=row[1].lower().split('@@')
            if model_type=='context2vec-skipgram?skipgram':
                    #context2vevc
                    context_avg_1=additive_model(sents,'___', model_type.split('?')[0],context_model[0],n_result,w[0],index2word[0],weight[0],w2entropy[0],w_target[0],word2index_target[0],index2word_target[0],f_w)
                    print ('context2vec avg embed',context_avg_1[:10])
                    context_avg_2=additive_model(sents,'___', model_type.split('?')[1],context_model[1],n_result,w[1],index2word[1],weight[1],w2entropy[1],w_target[1],word2index_target[1],index2word_target[1],f_w)
                    print ('skipgram avg embed', context_avg_2[:10])
                    context_avg=(context_avg_1+context_avg_2)/2
                    print ('context2vec avg out', context_avg[:10])
                    #compute probe embeddings in skipgram space
                    w_out=w[1]
                    w_target_out=w_target[1]
                    word2index_out=word2index[1]
                    word2index_target_out=word2index_target[1]
                    
            else:

                    context_avg=additive_model(sents,'___', model_type,context_model,n_result,w,index2word,weight,w2entropy,w_target,word2index_target,index2word_target,f_w)
                    print ('context avg out', context_avg[:10])
                    w_out=w
                    w_target_out=w_target
                    word2index_out=word2index
                    word2index_target_out=word2index_target
            
            context_avg = context_avg / xp.sqrt((context_avg * context_avg).sum())

           
            
            #cosine similarity with probe embedding
            for gold,probe in zip(row[3].split(','),row[2].split(',')):
                try:
                    if word2index_target_out==None:
                        probe_w_vec=w_out[word2index_out[probe]]
                    else:
                        probe_w_vec=w_target_out[word2index_target_out[probe]]
                    probe_w_vec=probe_w_vec/xp.sqrt((probe_w_vec*probe_w_vec).sum())
                    cos=float(probe_w_vec.dot(context_avg))
                    if np.isnan(cos):
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


# In[139]:


TOP_MUTUAL_SIM='top_mutual_sim'
TOP_CLUSTER_DENSITY='top_cluster_density'
LDA='lda'
INVERSE_S_FREQ='inverse_s_freq'
INVERSE_W_FREQ='inverse_w_q'
SUBSTITUTE_PROB='substitute_prob'
WEIGHT_DICT={0:False,1:TOP_MUTUAL_SIM,2:LDA,3:INVERSE_S_FREQ,4:INVERSE_W_FREQ,5:TOP_CLUSTER_DENSITY, 6:SUBSTITUTE_PROB}


if __name__=="__main__":
    
    #params read in
    if sys.argv[0]=='/usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py':
        
        data='./eval_data/data-chimeras/dataset.l2.fixed.test.txt.punct'
#         data='./eval_data/data-nonces/n2v.definitional.dataset.train.txt'
        weight=WEIGHT_DICT[0]
        gpu=1
#         ##context2vec
##         model_param_file='../models/context2vec/model_dir/context2vec.ukwac.model.params'
#         model_param_file='../models/context2vec/model_dir/MODEL-wiki.params.14'
        
        model_type='context2vec-skipgram'

####skipgram
        if model_type=='skipgram':
            model_param_file='../models/wiki_all.model/wiki_all.sent.split.model'
            model_type='skipgram'
    #         weight='inverse_w_freq'
    #         w2salience_f='../corpora/corpora/wiki.all.utf8.sent.split.tokenized.vocab'
    #         w2salience_f='../models/lda/w2entropy'
            n_result=20
            w2salience_f=None
        elif model_type=='context2vec-skipgram':
            model_param_file='../models/context2vec/model_dir/MODEL-wiki.params.14?../models/wiki_all.model/wiki_all.sent.split.model'
    #         model_param_file='../models/context2vec/model_dir/context2vec.ukwac.model.params?../models/wiki_all.model/wiki_all.sent.split.model'
            n_result=20
            w2salience_f=None
        elif model_type=='skipgram?context2vec-skipgram':
        #####skipgram?context2vec-skipgram
            model_param_file='../models/context2vec/model_dir/MODEL-wiki.params.14?../models/wiki_all.model/wiki_all.sent.split.model'
    #         weight='inverse_w_freq'
    #         w2salience_f='../corpora/corpora/wiki.all.utf8.sent.split.tokenized.vocab'
    #         w2salience_f='../models/lda/w2entropy'
            n_result=20
            w2salience_f=None
    
    else:
        

        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--f',  type=str,
                            help='model_param_file',dest='model_param_file')
        parser.add_argument('--m', dest='model_type', type=str,
                            help='<model_type: context2vec; context2vec-skipgram (context2vec substitutes in skipgram space); context2vec-skipgram?skipgram (context2vec substitutes in skipgram space plus skipgram context words)>')
        parser.add_argument('--w', dest='weight', type=str, help='<weight:{0}>'.format (sys.argv[0],WEIGHT_DICT.items()))       
        parser.add_argument('--d', dest='data', type=str, help='data file')
        parser.add_argument('--g', dest='gpu',type=int, default=-1,help='gpu, default is -1')
        parser.add_argument('--ws', dest='w2salience_f',type=str, default=None,help='word2salience file, optional')

        args = parser.parse_args()
        
#         if len(sys.argv) < 6:
#             print >> sys.stderr, "Usage: {0} <model_param_file> <model_type: context2vec; context2vec-skipgram (context2vec substitutes in skipgram space); context2vec-skipgram?skipgram (context2vec substitutes in skipgram space plus skipgram context words)> <weight:{1}> <eval_data> <gpu> <w2salience> "  .format (sys.argv[0],WEIGHT_DICT.items())
#             sys.exit(1)
        
        model_param_file = args.model_param_file
        model_type=args.model_type
        
        if '-' in args.weight:
            weight,n_result=args.weight.split('-')
            weight=WEIGHT_DICT[int(weight)]
            n_result=int(n_result)
        else:
            weight=WEIGHT_DICT[int(args.weight)]
            n_result=20 #default is 20 top
            
        data =args.data
        gpu=args.gpu
        w2salience_f=args.w2salience_f
       
    
    #gpu setup 
   

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
        
        model_reader = ModelReader(model_param_file,gpu)
        w = xp.array(model_reader.w)
        index2word = model_reader.index2word
        word2index=model_reader.word2index
        model = model_reader.model
        w_target=None
        word2index_target=None
        index2word_target=None
        
    elif model_type=='skipgram':
        model_w2v = gensim.models.Word2Vec.load(model_param_file)
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
        
        model_w2v = gensim.models.Word2Vec.load(model_param_w2v)
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
        
        model_w2v = gensim.models.Word2Vec.load(model_param_w2v)
        w_target=xp.array(model_w2v.wv.vectors)
        index2word_target=model_w2v.wv.index2word
        word2index_target={key: model_w2v.wv.vocab[key].index for key in model_w2v.wv.vocab}
    
        # skigpram
        model_skipgram = model_w2v
        w_skipgram=xp.array(deepcopy(model_skipgram.wv.vectors))
        #vector normalize for probe w embedding
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
        
                
        
    
    #remove stop words in target word space and asarray
    print ('filter words for target....')
    w,word2index,index2word=filter_w(w,word2index,index2word)
    if  index2word_target!=None:
        w_target,word2index_target,index2word_target=filter_w(w_target,word2index_target,index2word_target)
    if model_type=='context2vec-skipgram?skipgram':
        w_skipgram,word2index_skipgram,index2word_skipgram=filter_w(w_skipgram,word2index_skipgram,index2word_skipgram)
    
    #per word weight
    
    w2salience=None
    if weight==LDA:
        print ('load vectors and entropy')
        w2salience=pickle.load(open(w2salience_f))
    elif weight==INVERSE_W_FREQ:
        print ('load w2freq')
        w2salience=load_w2salience(w2salience_f,weight)
    elif weight==INVERSE_S_FREQ:
        print ('load w2freq')
        w2salience=load_w2salience(w2salience_f,weight)


    #combine parameters for skipgram?context2vec-skipgram
    if model_type=='context2vec-skipgram?skipgram':
        model=(model,model_skipgram)
        w=(w,w_skipgram)
        index2word=(index2word,index2word_skipgram)
        word2index=(word2index,word2index_skipgram)
        weight=(weight,WEIGHT_DICT[0])#assume that skipgram has no weight
        w2salience=(w2salience,w2salience)
        w_target=(w_target,w_target_skipgram)
        word2index_target=(word2index_target,word2index_target_skipgram)
        index2word_target=(index2word_target,index2word_target_skipgram)
    
    print (model_param_file,model_type,weight,data,w2salience_f)


# In[136]:


#read in data
import time
start_time = time.time()


if data.split('/')[-2]== 'data-chimeras':

        eval_chimera(data,model,model_type,n_result,w,index2word,word2index,weight,w2salience,w_target,word2index_target,index2word_target)

elif data.split('/')[-2]== 'data-nonces':
        ranks=eval_nonce(data,model,model_w2v,model_type,n_result,w,index2word,word2index,weight,w2salience,w_target,word2index_target,index2word_target)
print("--- %s seconds ---" % (time.time() - start_time))

