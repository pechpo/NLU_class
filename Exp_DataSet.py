import os
import json
import numpy as np
import torch
import jieba
import pickle
from tqdm import tqdm
from torch.utils.data import TensorDataset

class Dictionary(object):
    def __init__(self, path):

        #词和对应的token数字
        self.word2tkn = {"[PAD]": 0}
        self.tkn2word = ["[PAD]"]

        #话题label和对应的label编号
        self.label2idx = {}
        self.idx2label = []

        # 获取 label 的 映射
        with open(os.path.join(path, 'labels.json'), 'r', encoding='utf-8') as f:
            for line in f:
                one_data = json.loads(line)
                label, label_desc = one_data['label'], one_data['label_desc']
                self.idx2label.append([label, label_desc])
                self.label2idx[label] = len(self.idx2label) - 1

    def add_word(self, word):  #给字典加一个词，维护词的编号(也就是token)
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]

class Corpus(object):
    '''
    完成对数据集的读取和预处理，处理后得到所有文本数据的对应的 token 表示及相应的标签。
    
    该类适用于任务一、任务二，若要完成任务三，需对整个类进行调整，例如，可直接调用预训练模型提供的 tokenizer 将文本转为对应的 token 序列。
    '''
    def __init__(self, path, max_token_per_sent, embedding_dim):
        self.dictionary = Dictionary(path)

        self.max_token_per_sent = max_token_per_sent

        print("Loading data...")

        self.train = self.tokenize(os.path.join(path, 'train.json'))
        self.valid = self.tokenize(os.path.join(path, 'dev.json'))
        self.test = self.tokenize(os.path.join(path, 'test.json'), True)

        #-----------------------------------------------------begin-----------------------------------------------------#
        # 若要采用预训练的 embedding, 需处理得到 token->embedding 的映射矩阵 embedding_weight。矩阵的格式参考 nn.Embedding() 中的参数 _weight
        # 注意，需考虑 [PAD] 和 [UNK] 两个特殊词向量的设置

        if os.path.exists("embedding.pkl"):  #避免重复读取
            f = open("embedding.pkl", "rb")
            self.embedding = pickle.load(f)
            f.close()
            return

        print("Loading pre-trained embedding...")
        f = open("sgns.baidubaike.bigram-char", "r", encoding="utf-8")
        a = f.read().split("\n")
        dic = {}  #从单词到词向量的映射
        first = True
        for line in tqdm(a):
            if first:
                first = False
                continue
            line = line.strip()
            b = line.split(" ")
            dic[b[0]] = b[1:]
        f.close()

        print("Loading word embedding...")
        self.embedding = torch.zeros(len(self.dictionary.word2tkn), embedding_dim)
        for word, token in self.dictionary.word2tkn.items():
            if word == "[PAD]":
                continue
            if word not in dic.keys():
                print(word)  #[UNK]
                continue
            #print(dic[word])
            c = list(map(float, dic[word]))
            #print(c)
            self.embedding[token] = torch.tensor(c)
        
        f = open("embedding.pkl", "wb")
        pickle.dump(self.embedding, f)
        f.close()

        #------------------------------------------------------end------------------------------------------------------#

    def pad(self, origin_token_seq):
        '''
        padding: 将原始的 token 序列补 0 至预设的最大长度 self.max_token_per_sent
        '''
        if len(origin_token_seq) > self.max_token_per_sent:
            return origin_token_seq[:self.max_token_per_sent]
        else:
            return origin_token_seq + [0 for _ in range(self.max_token_per_sent-len(origin_token_seq))]

    def tokenize(self, path, test_mode=False):
        '''
        处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
        '''
        idss = []
        labels = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                one_data = json.loads(line)  # 读取一条数据
                sent = one_data['sentence']
                #-----------------------------------------------------begin-----------------------------------------------------#
                # 若要采用预训练的 embedding, 需在此处对 sent 进行分词

                sent = jieba.lcut(sent)

                #------------------------------------------------------end------------------------------------------------------#
                # 向词典中添加词
                for word in sent:
                    self.dictionary.add_word(word)

                ids = []  #当前句子对应的token序列
                for word in sent:
                    ids.append(self.dictionary.word2tkn[word])
                idss.append(self.pad(ids))
                
                # 测试集无标签，在 label 中存测试数据的 id，便于最终预测文件的打印
                if test_mode:
                    label = json.loads(line)['id']      
                    labels.append(label)
                else:
                    label = json.loads(line)['label']
                    labels.append(self.dictionary.label2idx[label])

            idss = torch.tensor(np.array(idss))
            labels = torch.tensor(np.array(labels)).long()
            
        return TensorDataset(idss, labels)  #一个token序列对应一个label数字