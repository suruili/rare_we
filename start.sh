#run inside the docker nvidia-docker run  --name context2vec-gpu -it -p 8888:8888 -v /home/ql261/simp2trad/:/home/simp2trad/ chainer/chainer:v4.0.0-python2-lqc /bin/bash 

#install optional
# sudo apt-get update
# sudo apt-get upgrade
# sudo apt-get install git
git config --global user.email "hey_flora@126.com"
git config --global user.name "qianchu"

# set up bash shell
# pack python project
cd /home/rare_we/models/context2vec
sudo python setup.py install
sudo pip install -U scipy
sudo pip install -U scikit-learn
sudo pip install matplotlib
sudo pip install nltk
sudo pip install --upgrade gensim
# run jupyter
cd /home/
sudo chmod -R 777 ./*
cd /home/rare_we/
# sudo python -m pip install --upgrade pip
# sudo python -m pip install jupyter
# sudo python -m pip install pandas

if [ ! -d '/home/ql261/.jupyter/' ]; then
    sudo mkdir /home/ql261/.jupyter/
fi
sudo cp /home/rare_we/jupyter_notebook_config.py /home/ql261/.jupyter/

for pid in $(ps -def | grep jupyter | awk '{print $2}'); do sudo kill -9 $pid; done

export SHELL=/bin/bash
jupyter notebook --ip '*'  --port=8887 --allow-root &
