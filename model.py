from keras.layers import Embedding,Dense,LSTM
from keras.models import Sequential

class LstmModel:
    def __init__(self, vector_dim):
        self.pre_vector_dim = vector_dim
        self.latent_dim = 256

    def build_model(self, pre_embedding_matrix, word_num, max_length):
        self.model = Sequential()

        self.model.add(Embedding(word_num+1,
                                 self.pre_vector_dim,
                                 weights=[pre_embedding_matrix],
                                 input_length=max_length,
                                 trainable=False))
        self.model.add(LSTM(self.latent_dim,dropout=0.5))
        self.model.add(Dense(3, activation="softmax"))

        self.model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=['accuracy'])

    def fit(self, x, y, batch_size, epochs):
        file_path = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5 "
        self.model.fit(x, y ,batch_size=batch_size, epochs=epochs, validation_split=0.2)

    def evaluate(self,x,y,batch_size):
        self.model.evaluate(x,batch_size)

    def predict(self, x):
        return self.model.predict(x)












