
# coding: utf-8

# In[1]:


import sys
import h5py
import argparse
import os
from context2vec.common.model_reader import ModelReader
import cupy as cp
from chainer import cuda

import numpy as np

import math
import re
from collections import defaultdict


# In[2]:


def increment_write_h5py(hf,chunk,data_name='data'):
    if data_name not in hf:
        maxshape = (None,) + chunk.shape[1:]
        data=hf.create_dataset(data_name,data=chunk,chunks=chunk.shape,maxshape=maxshape,compression="gzip",compression_opts=9)

    else:
        data=hf[data_name]
        data.resize((chunk.shape[0]+data.shape[0],)+data.shape[1:])
        data[-chunk.shape[0]:]=chunk



# In[11]:




class context2vec_batch_generator(object):
    
    def __init__(self,batchsize,model,word2index,output_f):
        self.same_len_dict=defaultdict(list)
        self.same_len_2_index=defaultdict(lambda:-1)
        self.batchsize=batchsize
        self.model=model
        self.word2index=word2index
        self.output_f=output_f
        
    def process_batch(self,w_lst):
        sent_len=len(w_lst)
        w_ind_lst=self.sent2wordid(w_lst)
        self.same_len_dict[sent_len].append(w_ind_lst)
        self.same_len_2_index[sent_len]+=1
        if len(self.same_len_dict[sent_len])>=self.batchsize: #process batches
            #run model for a batch
            print ('run model for sent len {0}'.format(str(sent_len)))
            self.model.reset_state()
            sent_ys = self.model._contexts_rep(xp.array(self.same_len_dict[sent_len]))
            sent_ys=xp.array([arr.data for arr in sent_ys]).swapaxes(0,1)
            #write to h5py
            increment_write_h5py(self.output_f,sent_ys,data_name=str(len(w_lst)))
            self.same_len_dict[sent_len]=[]
    
    def process_remainder(self):
        for sent_len in self.same_len_dict:
            if self.same_len_dict[sent_len]!=[]:
                print ('remainder run_model for sent leng {0}'.format(str(sent_len)))
                self.model.reset_state()
                sent_ys = self.model._contexts_rep(xp.array(self.same_len_dict[sent_len]))
                sent_ys=xp.array([arr.data for arr in sent_ys]).swapaxes(0,1)
                #write to h5py
                increment_write_h5py(self.output_f,sent_ys,data_name=str(sent_len))
                self.same_len_dict[sent_len]=[]
                
    #helper functions
    def sent2wordid(self,w_lst):
        sent_inds = []
        for word in w_lst:
            word= word.decode('utf-8')
            if word in self.word2index:
                ind = self.word2index[word]
            else:
                print ('unknown word: {0}'.format(word.encode('utf-8')))
                ind = self.word2index['<UNK>']
            sent_inds.append(ind)
        return sent_inds


# In[5]:


if __name__ == '__main__':
    
    
    
    if sys.argv[0]=='/usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py':
        model_param_file='../models/context2vec/model_dir/MODEL-wiki.params.14'
        gpu=-1
        text_f='eval_data/CRW/context'
        batchsize=100

    else:
        parser = argparse.ArgumentParser(description='Write context2vec embeddings to file.')
        parser.add_argument('--f',  type=str,
                            help='model_param_file',dest='model_param_file')
        parser.add_argument('--g', dest='gpu',type=int, default=-1,help='gpu, default is -1')
        parser.add_argument('--t', dest='text_f', type=str, help='data text file or folder')
        parser.add_argument('--b',dest='batchsize',type=int,help='batch size')
        args = parser.parse_args()
        model_param_file=args.model_param_file
        text_f=args.text_f
        gpu=args.gpu
        batchsize=args.batchsize

    #1.setup
    if gpu >= 0:
            cuda.check_cuda_available()
            cuda.get_device(gpu).use()    
    xp = cuda.cupy if gpu >= 0 else np
    
    #2. initialize context2vec model 
    model_reader = ModelReader(model_param_file,gpu)
    word2index=model_reader.word2index
    model = model_reader.model
   


# In[16]:


#3. process context2vec text into vectors and store in h5py
output_f = h5py.File(text_f+'_'+model_param_file.split('/')[-1]+'.'+'vec.h5', 'w')
bg=context2vec_batch_generator(batchsize,model,word2index,output_f)
index_out_f=open(text_f+'_'+model_param_file.split('/')[-1]+'.'+'index','w')

if os.path.isdir(text_f):
    index_out=defaultdict(list)
    for root, subdir, files in os.walk(text_f):
        for f in files:
            name=f.split('.')[0]
            for line in open(os.path.join(root,f)):
                #read in one line at once and batch sentences
                w_lst=line.strip().split()
                bg.process_batch(w_lst) # process batch and write to h5py          
                index_in_h5py=bg.same_len_2_index[len(w_lst)]
                index_out[name].append(str(len(w_lst))+','+str(index_in_h5py))         
    bg.process_remainder()
            
    #write indexes
    for name in index_out:
        index_out_f.write('{0}:::{1}'.format(name,'\t'.join(index_out[name])))

index_out_f.close()
output_f.close()

