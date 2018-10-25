import numpy as np
import pandas as pd
import os
import jieba


class DataLoader:
    def __init__(self):
        word_path = "/Users/chenzheng"
        data_path = "/Users/chenzheng/DataFoutainRace/vehicle_user_comment"
        self.vocab = {}
        word_file = os.path.join(word_path, "sgns.zhihu.word")
        self._initEmbedding(word_file)
        data_file = os.path.join(data_path, "train.csv")
        stop_file = os.path.join(data_path, "stop_words.txt")
        self._load_data(data_file)
        self._get_classes()
        self._map_word2vec()
        self._load_stop_words(stop_file)

    def _initEmbedding(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            line = f.readline()
            self.embedding_dim = int(line.strip().split(" ")[1])
            while line:
                line = f.readline()
                strs = line.strip().split(" ")
                word = strs[0]
                embedding = np.asarray(strs[1:], dtype="float32")
                self.vocab[word] = embedding

    # 加载中文停用词
    def _load_stop_words(self, stop_path):
        self.stop_words = set()
        with open(stop_path, "rb") as f:
            line = f.readline()
            while line:
                self.stop_words.add(line.strip())
                line = f.readline()

    def _load_data(self, data_path):
        self.data = pd.read_csv(data_path).drop_duplicates(["content_id"], keep="first")

    def _get_classes(self):
        self.class_mapping = { label:index for index, label
                               in enumerate(np.unique(self.data["subject"]))}
        self.data["subject"] = self.data["subject"].map(self.class_mapping)

        self.senti_dict = { label:index for index, label
                                     in enumerate(np.unique(self.data["sentiment_value"]))}
        self.data["sentiment_class"] = self.data["sentiment_value"].map(self.senti_dict)

    def _map_word2vec(self):
        words = self.vocab.keys()
        # 所有词的个数
        self.words_num = len(words)
        self.words_index = range(1, len(words)+1)
        # 由词建立索引
        self.word2Index = dict(zip(words, self.words_index))
        self.pre_embedding_matrix = np.zeros((len(words)+1, self.embedding_dim))
        for word, i in self.word2Index.items():
            embedding_vector = self.vocab.get(word,None)
            if embedding_vector is not None and word is not "":
                self.pre_embedding_matrix[i] = embedding_vector


    def getDataFrame(self):
        def cut_func(str):
            words = jieba.lcut(str)
            words = [word for word in words if word not in self.stop_words]
            words_index = [self.word2Index.get(word,0) for word in words]
            return words_index

        self.data["words"] = self.data["content"].apply(cut_func)
        max_length = self.data["words"].map(lambda x:len(x)).max()
        return self.data,max_length




if __name__=="__main__":
    data = DataLoader()



