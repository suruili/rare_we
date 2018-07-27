
# coding: utf-8

# In[18]:


# from collections import Counter,defaultdict
# word_counts=defaultdict(lambda: [0, 0])
# word_counts['a'][0]+=1
# word_counts['b'][0]=5
# sorted(word_counts.items(), key=lambda x:x[1][0],reverse=True)


# In[13]:


import sys
import nltk
# from nltk.tokenize import sent_tokenize
import codecs
from collections import Counter,defaultdict

def tokenize_vocab(corpus_out):
    word_counts=defaultdict(lambda: [0, 0])
    with codecs.open (corpus_out, 'w',encoding='utf-8') as f_out:
        #first pass
        counter=0
        for line in f:
            
            line=line.strip().lower()
            if line=='':
                    continue
            bgw_per_sent={}
            ws=line.split()
            for w in ws:
                word_counts[w][0]+=1
                if w not in bgw_per_sent:
                    bgw_per_sent[w]=1
                    word_counts[w][1]+=1

            f_out.write(line+'\n')
                
            if counter%10000==0 and counter>=10000:
                print ('{0} '.format(counter)),
            counter+=1
    return word_counts
       
def write_to_vocab(vocab_fn,word_count):
    with codecs.open(vocab_fn,encoding='utf-8',mode='w') as vocab_f:
            w_id=0
            word_counts_most_common_w=sorted(word_count.items(), key=lambda x:x[1][0],reverse=True)
            for w, counts in word_counts_most_common_w:
                w_count,s_count=counts
                vocab_f.write(w+'\t'+str(w_count)+'\t'+str(s_count)+'\n')
                word2id[w]=w_id
                id2word[w_id]=w
                w_id+=1
           
    return word2id,id2word

if __name__=="__main__":
#     nltk.download('punkt')
    word_context_matrix=defaultdict(lambda: Counter())
    context_max=int(sys.argv[2])
    target_max=int(sys.argv[3])
    word2id={}
    id2word={}
    
    
    corpus_dir=sys.argv[1]
#     corpus_dir='./corpora/wiki.all.utf8.sent.split.mini'
    corpus_out=corpus_dir+'.tokenized'
    vocab_fn=corpus_dir+'.tokenized.vocab'
    w_c_fn=corpus_dir+'.tokenized.context'
    with codecs.open (corpus_dir,encoding='utf-8') as f:
        print ('===first pass====')
        word_count=tokenize_vocab(corpus_out)
        word2id,id2word=write_to_vocab(vocab_fn,word_count)
        word_count=''
        
#                     for c_w in ws:
#                         if c_w !=w:
#                             word_context_matrix[w][c_w]+=1
                
                                
        #filter target and contexg words
#         target_w_freq=word_counts.most_common()[:20000]
#         target_w=zip(*target_w_freq)[0]
#         context_w=zip(*target_w_freq[:5000])[0]
                
        #second pass
    if context_max!=0 and target_max!=0:
        with codecs.open (corpus_dir,encoding='utf-8') as f:
            print ('\n===second pass=====')
            counter=0
            for line in f:

                line=line.strip().lower()
                if line=='':
                        continue

                if counter%10000==0 and counter>=10000:
                    print ('{0} '.format(counter)),
                ws=line.split()
                counter+=1
                for w in ws:
                    if word2id[w] <target_max:
                        for c_w in ws:
                            if word2id[c_w]<context_max and c_w !=w:
                                word_context_matrix[w][c_w]+=1




        with codecs.open(w_c_fn,encoding='utf-8',mode='w') as w_c_f:
            for i in range(min(target_max,len(id2word))):
                w=id2word[i]
                w_c_pairs= [str(word2id[c_w])+':'+str(word_context_matrix[w][c_w]) for c_w in word_context_matrix[w] if word2id[c_w]<context_max]
                w_c_f.write(str(len(w_c_pairs))+' '+' '.join(w_c_pairs)+'\n')


            
        


# In[3]:


# from collections import Counter
# a=Counter()
# a['a']+=1
# a['b']+=1
# zip(*a.most_common())

