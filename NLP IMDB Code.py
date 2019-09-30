import tensorflow as tf
tf.__version__
tf.enable_eager_execution()

import tensorflow_datasets as tfds
imdb,info = tfds.load('imdb_reviews',with_info=True,as_supervised=True)
import pandas as pd
import numpy as np


train_data ,test_data = imdb['train'],imdb['test']

training_sentences=[]
training_labels=[]

test_sentences=[]
test_labels=[]

for s,l in train_data:
    training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())

for s,l in test_data:
    test_sentences.append(str(s.numpy()))
    test_labels.append(l.numpy())

training_labels=np.array(training_labels)

test_labels=np.array(test_labels)

from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences



tokenizer = Tokenizer(num_words=10000,oov_token='<OOV>')
tokenizer.fit_on_texts(training_sentences)
word_index=tokenizer.word_index

sequences=tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,padding='post',maxlen=120,truncating='post')

test_sequences=tokenizer.texts_to_sequences(test_sentences)
test_padding=pad_sequences(test_sequences,padding='post',maxlen=120,truncating='post')


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[1]))
print(training_sentences[1])



import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Embedding

model=Sequential()
model.add(Embedding(10000, 16, input_length=120))
model.add(Flatten())
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=10,validation_data=(test_padding,test_labels))

y_pred = model.predict()

y_test = (y_pred>0.5)

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)


import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()




































