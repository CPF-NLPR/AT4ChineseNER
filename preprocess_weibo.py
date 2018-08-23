import numpy as np
import Weibo_model
import codecs
import re

rNUM = '(-|\+)?\d+((\.)\d+)?%?'
rENG = '[A-Za-z_.]+'
vector = []
word2id = {}
id2word = {}
tag_id = {}
id_tag={}
word_dim=100
num_steps=80

def load_embedding(setting):
    print 'reading chinese word embedding.....'
    f = open('./data/embed.txt','r')
    f.readline()
    while True:
        content=f.readline()
        if content=='':
            break
        else:
            content=content.strip().split()
            word2id[content[0]]=len(word2id)
            id2word[len(id2word)]=content[0]
            content=content[1:]
            content=[float(i) for i in content]
            vector.append(content)
    f.close()
    word2id['padding']=len(word2id)
    word2id['unk']=len(word2id)
    vector.append(np.zeros(shape=setting.word_dim,dtype=np.float32))
    vector.append(np.random.normal(loc=0.0,scale=0.1,size=setting.word_dim))
    id2word[len(id2word)]='padding'
    id2word[len(id2word)]='unk'

def process_train_data(setting):
    print 'reading train data.....'
    train_word=[]
    train_label=[]
    train_length=[]
    f=open('./data/weiboNER.conll.train','r')
    train_word.append([])
    train_label.append([])
    train_max_len=0
    while True:
        content=f.readline()
        if content=='':
            break
        elif content=='\n':
            length=len(train_word[len(train_word)-1])
            train_length.append(min(length,num_steps))
            if length>train_max_len:
                train_max_len=length
            train_word.append([])
            train_label.append([])
        else:
            content=content.replace('\n','').replace('\r','').strip().split()
            if content[1]!='O':
                label1=content[1].split('.')[0]
                label2=content[1].split('.')[1]
                content[1]=label1
                if label2=='NOM':
                    content[1]='O'
            if content[0] not in word2id:
                word2id[content[0]]=len(word2id)
                vector.append(np.random.normal(loc=0.0,scale=0.1,size=setting.word_dim))
                id2word[len(id2word)]=content[0]
            if content[1] not in tag_id:
                tag_id[content[1]]=len(tag_id)
                id_tag[len(id_tag)]=content[1]
            train_word[len(train_word)-1].append(word2id[content[0]])
            train_label[len(train_label)-1].append(tag_id[content[1]])

    if len(train_word[len(train_word)-1])!=0:
        train_length.append(min(len(train_word[len(train_word)-1]),num_steps))
    if [] in train_word:
        train_word.remove([])
    if [] in train_label:
        train_label.remove([])

    assert len(train_word)==len(train_label)
    assert len(train_word)==len(train_length)
    for i in range(len(train_word)):
        if len(train_word[i])<num_steps:
            for j in range(num_steps-train_length[i]):
                train_word[i].append(word2id['padding'])
                train_label[i].append(tag_id['O'])
        else:
            train_word[i]=train_word[i][:num_steps]
            train_label[i]=train_label[i][:num_steps]

    train_word = np.asarray(train_word)
    train_label = np.asarray(train_label)
    train_length = np.asarray(train_length)
    np.save('./data/weibo_train_word.npy',train_word)
    np.save('./data/weibo_train_label.npy',train_label)
    np.save('./data/weibo_train_length.npy', train_length)

def process_test_data(setting):
    print 'reading test data.....'
    test_word=[]
    test_label=[]
    test_length=[]
    f=open('./data/weiboNER.conll.test','r')
    test_word.append([])
    test_label.append([])
    test_max_len=0
    while True:
        content=f.readline()
        if content=='':
            break
        elif content=='\n':
            test_length.append(min(len(test_word[len(test_word)-1]),num_steps))
            if len(test_word[len(test_word)-1])>test_max_len:
                test_max_len=len(test_word[len(test_word)-1])
            test_word.append([])
            test_label.append([])
        else:
            content = content.replace('\n', '').replace('\r', '').strip().split()
            if content[1]!='O':
                label1=content[1].split('.')[0]
                label2=content[1].split('.')[1]
                content[1]=label1
                if label2=='NOM':
                    content[1]='O'
            if content[0] not in word2id:
                word2id[content[0]]=len(word2id)
                vector.append(np.random.normal(loc=0.0,scale=0.1,size=setting.word_dim))
                id2word[len(id2word)]=content[0]
            if content[1] not in tag_id:
                tag_id[content[1]]=len(tag_id)
                id_tag[len(id_tag)]=content[1]
            test_word[len(test_word)-1].append(word2id[content[0]])
            test_label[len(test_label)-1].append(tag_id[content[1]])
    if len(test_word[len(test_word)-1])!=0:
        test_length.append(len(test_word[len(test_word)-1]))
    if [] in test_word:
        test_word.remove([])
    if [] in test_label:
        test_label.remove([])
    assert len(test_word) == len(test_label)
    assert len(test_word) == len(test_length)
    for i in range(len(test_word)):
        if len(test_word[i]) < num_steps:
            for j in range(num_steps - test_length[i]):
                test_word[i].append(word2id['padding'])
                test_label[i].append(tag_id['O'])
        else:
            test_word[i]=test_word[i][:num_steps]
            test_label[i]=test_label[i][:num_steps]
    test_word = np.asarray(test_word)
    test_label = np.asarray(test_label)
    test_length = np.asarray(test_length)
    np.save('./data/weibo_test_word.npy',test_word)
    np.save('./data/weibo_test_label.npy',test_label)
    np.save('./data/weibo_test_length.npy', test_length)

def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += unichr(inside_code)
    return rstring

def preprocess(filename):
    sentence=[]
    length=[]
    label=[]
    max_len=0
    with codecs.open(filename,'r','utf-8') as f:
        print 'reading cws data.....'
        for line in f:
            sent=strQ2B(line).split()
            new_sent=[]
            sent_label=[]
            for word in sent:
                word=re.sub(rNUM,'0',word)
                word=re.sub(rENG,'X',word)
                for i in range(len(word)):
                    if word[i] not in word2id:
                        word2id[word[i]]=len(word2id)
                        vector.append(np.random.normal(loc=0.0, scale=0.1, size=word_dim))
                        id2word[len(id2word)]=word[i]
                    new_sent.append(word2id[word[i]])
                if len(word)==1:
                    sent_label.append(0)
                elif len(word)==2:
                    sent_label.extend([1,3])
                else:
                    sent_label.append(1)
                    for i in range(1,len(word)-1):
                        sent_label.append(2)
                    sent_label.append(3)
            length.append(min(len(new_sent),num_steps))
            if len(new_sent)>max_len:
                max_len=len(new_sent)
            sentence.append(new_sent)
            label.append(sent_label)
    for i in range(len(sentence)):
        if len(sentence[i])<num_steps:
            sent_len=len(sentence[i])
            for j in range(num_steps-sent_len):
                sentence[i].append(word2id['padding'])
                label[i].append(0)
        else:
            sentence[i]=sentence[i][:num_steps]
            label[i]=label[i][:num_steps]
    assert len(sentence)==len(label)
    assert len(sentence)==len(length)
    sentence=np.asarray(sentence,dtype=np.int32)
    label=np.asarray(label,dtype=np.int32)
    length=np.asarray(length,np.int32)
    np.save('./data/weibo_cws_word.npy', sentence)
    np.save('./data/weibo_cws_label.npy', label)
    np.save('./data/weibo_cws_length.npy', length)

def id_to_tag(x):
    return id_tag[x]

def id_to_word(x):
    return id2word[x]

setting=Weibo_model.Setting()
load_embedding(setting)
process_train_data(setting)
process_test_data(setting)

filename='./data/msr_training.utf8'
preprocess(filename)
vector=np.asarray(vector)
np.save('./data/weibo_vector.npy',vector)
print 'The number of word is:'
print len(word2id)
print 'The number of tag is:'
print len(tag_id)


