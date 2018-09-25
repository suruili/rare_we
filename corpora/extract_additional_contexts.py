
# coding: utf-8

# In[1]:


import argparse
import os
from collections import defaultdict
import pandas as pd
import sys


# In[2]:


if __name__=='__main__':
    parser=argparse.ArgumentParser('Extract additional contexts')
    parser.add_argument('--sf',type=str,help='source filename from which contexts are extracted', dest='sourcef')
    parser.add_argument('--cf',type=str,help='file directory in which sentences are stored in directories according to sentence length, to compare with  extracted contexts', dest='compf')
    parser.add_argument('--rwf',type=str,help='a file that contains rarewords each on a line', dest='rarewf')
    args = parser.parse_args()
    
    sourcef=args.sourcef
    compf=args.compf
    rarewf=args.rarewf
    
    outf=os.path.join(os.path.dirname(sourcef),'contexts')
    if os.path.isdir(outf):
        print ('directory exists: {0}'.format(os.path.join(os.path.dirname(sourcef),'contexts')))
        sys.exit(1)
    os.makedirs(os.path.join(os.path.dirname(sourcef),'contexts'))
    data=pd.read_csv(os.path.join(rarewf),delimiter='\t',header=None,comment='#')
    rarewords=list(data[0])
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
                if rareword in line:
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
                                    


