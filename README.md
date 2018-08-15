# rare_we

1. requirements
    nvidia-docker
    nvidia-docker run  --name context2vec-rw -it -p 8887:8887 -v /home/ql261/rare_we/:/home/rare_we/ chainer/chainer:v4.0.0-python2-lq /bin/bash

2. chimeras evaluation
    Usage:python eval.py <model_param_file> <model_type> <weight> <eval_data> <w2salience>

    example usage: 	nice python -u eval.py ../models/context2vec/model_dir/MODEL-wiki.params.6 context2vec 0 0 ./eval_data/data-chimeras/dataset.l2.fixed.test.txt.punct &> eval_context2vec_6_0_0_l2_punct.log &

    weight:
        TOP_MUTUAL_SIM='top_mutual_sim'
            measure the top n substitutes' mutual similarity (weighted by the top 1 substitutes' compatibility to context)
        TOP_CLUSTER_DENSITY='top_cluster_density'
            measure the top n substitutes' distance towards the cluster centroid (weighted by the top 1 substitutes' compatibility to context)

        LDA='lda'
            word salience measured by lda topic entropy 

        INVERSE_S_FREQ='inverse_s_freq'
            word salience measured by inverse sentence frequency

        INVERSE_W_FREQ='inverse_w_q'
            word salience measuerd by inverse word frequency

        SUBSTITUTE_PROB='substitute_prob'
            substitutes weighted by their compatibility to context

        WEIGHT_DICT={0:False,1:TOP_MUTUAL_SIM,2:LDA,3:INVERSE_S_FREQ,4:INVERSE_W_FREQ,5:TOP_CLUSTER_DENSITY, 6:SUBSTITUTE_PROB}


    model_type:
        context2vec (word embedding in context2vec space)
        
        context2vec-skipgram(context2vec substitutes in skipgram space)
        
        context2vec-skipgram?skipgram (context2vec substitutes in skipgram space plus skipgram context words)
