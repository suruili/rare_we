
# coding: utf-8

# In[33]:


import argparse
import os
from collections import defaultdict
import pandas as pd
import sys
import numpy as np


# In[40]:


def rareword_lst(rarewf):
    if os.path.basename(os.path.dirname(rarewf))=='data-nonces':
        data=pd.read_csv(os.path.join(rarewf),delimiter='\t',header=None,comment='#',skip_blank_lines=True)
        rarewords=list(data[0])
    if os.path.basename(os.path.dirname(rarewf))=='card-660':
        data=pd.read_csv(os.path.join(rarewf),delimiter='\t',header=None,comment='#',skip_blank_lines=True)
      
        rarewords=list(set(list(data[0])+list(data[1])))
    rarewords=[str(w).lower() for w in rarewords if w==w]
    return rarewords


# In[41]:


rareword_lst('../eval/eval_data/card-660/dataset.tsv')


# In[2]:


if __name__=='__main__':
    parser=argparse.ArgumentParser('Extract additional contexts')
    parser.add_argument('--sf',type=str,help='source filename from which contexts are extracted', dest='sourcef')
    parser.add_argument('--cf',default='nodir',type=str,help='file directory in which sentences are stored in directories according to sentence length, to compare with  extracted contexts', dest='compf')
    parser.add_argument('--rwf',type=str,help='a file that contains rarewords each on a line', dest='rarewf')
    args = parser.parse_args()
    
    sourcef=args.sourcef
    compf=args.compf
    rarewf=args.rarewf
    
    outf=os.path.join(os.path.dirname(rarewf),'contexts')
    if os.path.isdir(outf):
        print ('directory exists: {0}'.format(outf))
        sys.exit(1)
    os.makedirs(outf)
    
    rarewords=rareword_lst(rarewf)
    print (rarewords)
    rareword2contexts=defaultdict(list)

    index=0
    with open (sourcef) as source_f:
        for line in source_f:
            line=line.strip().lower()
            if index>=1000 and index%1000==0:
                print (index)
            index+=1
            if rarewords==[]:
                print ('all done')
                break
            for rareword in rarewords:
                rareword_match=str(rareword).lower()
                for rareword_match in list(set([rareword_match,rareword_match.replace('_',' '),rareword_match.replace('-',' ')])):
                    if ' '+rareword_match+' ' in line or line.startswith(rareword_match+' '):
                        if os.path.isfile(os.path.join(compf,'sents.'+str(len(line.split())))):
                            with open(os.path.join(compf,'sents.'+str(len(line.split())))) as f:
                                if any(line == x.strip() for x in f) :
                                    continue
                        if line not in rareword2contexts[rareword]:
                            rareword2contexts[rareword].append(line+'\n')
                            if len(rareword2contexts[rareword])>=255:
                                print ('{0} done'.format(rareword))
                                rarewords.remove(rareword)
                                print (outf,rareword)
                                with open(os.path.join(outf,rareword+'.txt'),'w') as out_f:
                                    out_f.writelines(rareword2contexts[rareword])
                            break
    for rareword in rarewords:
        print ('{0} rareword has {1} number of contexts'.format(rareword,str(len(rareword2contexts[rareword]))))
                                    


