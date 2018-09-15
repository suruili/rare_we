
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


# class SentGenerator(defaultdict):
#     def __init__ (self,item_type,batchsize):
#         super(SentGenerator, self).__init__(item_type)
#         self.batchsize=batchsize
#         self.sent_ls=[]
#     def sent_generator(self,a_list):
        
#             for i in a_list:
#                 yield i
                
#     def convert_to_generator(self,batchsize):
#         self.sent_ls=[]
#         for sent_l in self:
#             repeat=int(math.ceil(float(len(self[sent_l]))/self.batchsize))
#             self.sent_ls+=[sent_l]*repeat
#             self[sent_l]=self.sent_generator(self[sent_l])
#         np.random.seed(1034)
#         np.random.shuffle(self.sent_ls) 
        
# def batch_data(line_list,batchsize):
#     sent_l2line_is=SentGenerator(list,batchsize)
#     for line_i,line in enumerate(line_list):
# #         print (list(line))
#         sent_l=len(list(line))
#         sent_l2line_is[sent_l].append(line_i)
# #     print (sent_l2line_is.keys())
#     sent_l2line_is.convert_to_generator(batchsize)
#     return sent_l2line_is

#  def read_batch(f1,line_i,batchsize, word2index1):        
#     batch = []
#     line_inds=[]
#     while len(batch) < batchsize:
#         try:
#             line1=next(f1)
            
#         except StopIteration:
#             print ('{0} completed'.format(f1))
#             return batch,line_inds
#         if not line1: break
#         sent_words1 = list(line1)
# #         print (sent_words1)
#         sent_inds1 = []
#         for word in sent_words1:
# #             print (word)
#             word= word.encode('utf-8')
#             if word in word2index1:
                
#                 ind = word2index1[word]
#             else:
#                 ind = word2index1['<UNK>']
#             sent_inds1.append(ind)
        
#         batch.append(sent_inds1)
#         line_inds.append(line_i)
#     return batch,line_inds

#  def next_batch(sent_l2line_is,simp_lines,batchsize):
    
#     for sent_l in sent_l2line_is.sent_ls:
#         sent_arr,line_inds=read_batch(sent_l2line_is[sent_l],simp_lines,batchsize,mr.word2index1)
#         if sent_arr==[]:
#             continue
#         else:
#             yield xp.array(sent_arr),line_inds
# # sent_ys = self._contexts_rep(sent_arr)


# In[23]:




class context2vec_batch_generator(object):
    
    def __init__(self,batchsize,model,word2index):
        self.same_len_dict=defaultdict(list)
        self.batchsize=batchsize
        self.model=model
        self.word2index=word2index

        
    def process_batch(self,line):
        w_lst=line.strip().split()
        sent_len=len(w_lst)
        w_ind_lst=self.sent2wordid(w_lst)
        self.same_len_dict[sent_len].append(w_ind_lst)
        if len(self.same_len_dict[sent_len])>=self.batchsize: #process batches
            #run model
            print ('run model')
            self.same_len_dict[sent_len]=[]
    def process_remainder(self):
        for sent_len in self.same_len_dict:
            if self.same_len_dict[sent_len]!=[]:
                print ('remainder run_model')
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


# In[13]:


if __name__ == '__main__':
    
    
    
    if sys.argv[0]=='/usr/local/lib/python2.7/dist-packages/ipykernel_launcher.py':
        model_param_file='../models/context2vec/model_dir/MODEL-wiki.params.14'
        gpu=-1
        
    #1.setup
    if gpu >= 0:
            cuda.check_cuda_available()
            cuda.get_device(gpu).use()    
    xp = cuda.cupy if gpu >= 0 else np
    #2. initialize the batchgenerator with context2vec model 
    model_reader = ModelReader(model_param_file,gpu)
#     index2word = model_reader.index2word
    word2index=model_reader.word2index
    model = model_reader.model
   


# In[24]:


batchsize=252
bg=context2vec_batch_generator(batchsize,model,word2index)
for root, subdir, files in os.walk('eval_data/CRW/context/'):
    for f in files:
        for line in open(os.path.join(root,f)):
            bg.process_batch(line)
bg.process_remainder()

