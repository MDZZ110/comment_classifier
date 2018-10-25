from data_loader import DataLoader
from model import LstmModel
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
import numpy as np

data = DataLoader()
words_index = data.words_index
pre_embedding_matrix = data.vocab

df, max_length = data.getDataFrame()
x = list(df["words"])
# 将数据进行填充保证长度一致 用0填充
x = pad_sequences(x, max_length)

sentiment_y = list(df["sentiment_class"])
sentiment_y= np_utils.to_categorical(sentiment_y)

subject_y = list(df["subject"])
subject_y = np_utils.to_categorical(subject_y)

tmp_y = sentiment_y[:10]
print(tmp_y)
for i in range(10):
    print(np.argmax(tmp_y[i,:]))

embedding_matrix = data.pre_embedding_matrix

myModel = LstmModel(data.embedding_dim)
myModel.build_model(embedding_matrix, data.words_num, max_length)

batch_size = 256
epochs = 50
myModel.fit(x, sentiment_y, 128, 1)

x_test = x[:10]
y_test = myModel.predict(x_test)
y_real = sentiment_y[:10]

for i in range(10):
    print(np.argmax(y_test[i,:]))
    print("---------------分割线------------")
    print(np.argmax(y_real[i,:]))









