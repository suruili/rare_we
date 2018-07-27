
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import re
# from collections import defaultdict

def space_punc(sent):
    new_sent=sent
    add=0
    for m in re.finditer(r'[\d\-\/\"\';,.!?\)\(]+',sent):
        if m.end(0)+add>=len(new_sent):
            return new_sent[:m.start(0)+add]+' '+new_sent[m.start(0)+add:m.end(0)+add]
        else:
            new_sent=new_sent[:m.start(0)+add]+' '+new_sent[m.start(0)+add:m.end(0)+add]+' '+new_sent[m.end(0)+add:]
            add+=2
    return new_sent

def replace_unk(words,test_w):
    try:
        pos=words.index(test_w.upper())
    except ValueError:
        try:
            pos=words.index(test_w.upper()+'S')
        except ValueError as e:
            try:
                pos=words.index(test_w.upper()+'ES')
            except ValueError as e:
                try:
                    pos=words.index(test_w.upper()[:-1]+'IES')
                except ValueError as e:
                    try:
                        pos=words.index(test_w.upper()+test_w.upper()[-1]+'ES')
                    except ValueError as e:
                        print (words,test_w,e)
    words[pos]='___'
    


if __name__=="__main__":
    trial2sent={}
    chimeras_data_dir='./eval_data/data-chimeras'
    
    data=pd.read_csv(os.path.join(chimeras_data_dir,'dataset_fixed.csv'))

    for index, row in data.iterrows():
        
        trial=str(row['TRIAL'])[:-2]
        if len(trial)<2:
            continue
        if '@@' not in str(row['PASSAGE']):
            continue
        
        if trial in trial2sent:
            continue
        
        test_ss=row['PASSAGE'].split('@@')
        test_ss_out=[]
        for test_s in test_ss:
            test_s.strip()
#             print (test_s)
            test_s=space_punc(test_s)
            words=test_s.split()
            replace_unk(words,row['NONCE'])
            test_ss_out.append(' '.join(words))
        
        trial2sent[trial]=(' @@ '.join(test_ss_out))

    
    for root, subdir, fnames in os.walk(chimeras_data_dir):
        for fn in fnames:
            if fn.endswith('fixed.train.txt'):
                print (fn)
                infos=fn.split('.')
                sent_no=infos[1].upper()
                data_sub=pd.read_csv(os.path.join(chimeras_data_dir,fn),delimiter='\t',header=None)
                for line_num in range(len(data_sub)):
                    data_sub.at[line_num,1]=trial2sent[str(data_sub.at[line_num,0])+'_'+sent_no].strip()
                data_sub.to_csv(header=False,index=False,sep='\t',path_or_buf=os.path.join(chimeras_data_dir,fn+'.punct'))
                        
                        
    

