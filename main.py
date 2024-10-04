import nltk 
from nltk.corpus import gutenberg
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

nltk.download('gutenberg')

import pandas as pd 

data = gutenberg.raw('shakespeare-hamlet.txt')

with open('hamlet.txt', 'w') as f:
    f.write(data)

with open('hamlet.txt','r') as file:
    text = file.read().lower()
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

tokenizer.word_index
total_words=len(tokenizer.word_index)+1

print(total_words)
input_sequences = []

for line in text.split('\n'):
    tokenized_line = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokenized_line)):
        n_gram_sequence = tokenized_line[:i+1]
        input_sequences.append(n_gram_sequence)

import numpy as np
max_sequences_len = max([len(x) for x in input_sequences])

padded_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequences_len, padding='pre'))

import tensorflow as tf 
from tensorflow.keras.utils import to_categorical
X = padded_sequences[:,:-1]
labels = padded_sequences[:,-1]
y= to_categorical(labels, num_classes=len(tokenizer.word_index)+1)
labels.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow import keras 
from tensorflow.keras.layers import Embedding, LSTM, Dense,Dropout
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequences_len-1))
model.add(LSTM(128,activation='relu',return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64,activation='relu'))

model.add(Dense(len(tokenizer.word_index)+1, activation='softmax')) 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))


def predict_next_word(seed_text, model, tokenizer, max_sequence_len):
    tokenized_seed = tokenizer.texts_to_sequences([seed_text])[0]
    tokenized_seed = pad_sequences([tokenized_seed], maxlen=max_sequence_len-1, padding='pre')  
    predicted_probs = model.predict(tokenized_seed, verbose=0)[0]
    predicted_index = np.argmax(predicted_probs)
    predicted_word = tokenizer.index_word[predicted_index]
    return predicted_word


input_text="I am the"

predicted_word = predict_next_word(input_text, model, tokenizer, max_sequences_len)
print(f"Predicted word for '{input_text}': {predicted_word}")