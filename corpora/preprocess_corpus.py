
# coding: utf-8

# In[31]:


import sys
import nltk
from nltk.tokenize import sent_tokenize
import codecs

if __name__=="__main__":
    nltk.download('punkt')
    counter=0
    #corpus_dir=sys.argv[1]
    corpus_dir='enwiki-20161001_corpus.txt'
    corpus_out=corpus_dir+'.tokenized'
    with codecs.open (corpus_dir,encoding='utf-8') as f:
        with codecs.open (corpus_out, 'w',encoding='utf-8') as f_out:
            for line in f:
                line=line.strip()
                
                if line=='':
                    continue
                f_out.write(('\n'.join(sent_tokenize(line))+'\n'))
                counter+=1
                if counter%10000==0 and counter>=10000:
                    print ('.'),

