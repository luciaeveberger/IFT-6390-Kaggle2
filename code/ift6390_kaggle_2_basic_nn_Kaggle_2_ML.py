#!/usr/bin/env python
# coding: utf-8

# # Importing and data reading
# 
# 

# In[1]:


import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print(tf.__version__)


# In[2]:


from google.colab import drive

drive.mount("/content/drive")


# In[61]:


def get_train_data(path="/content/drive/My Drive/Mila/ML/kaggle_2/data/"):
  x_train = pd.read_csv(path + 'train_data.csv')
  y_train = pd.read_csv(path + 'train_results.csv')
  return x_train["text"].tolist(), y_train["target"].tolist()

x_train, y_train = get_train_data()


# # Preprocesing

# In[62]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

nltk.download('stopwords')
english_stopwords = stopwords.words("english")
english_stopwords.extend(["..", "...", "...."])
negative_words = ('no', 'not')
english_stopwords = [e for e in english_stopwords if e not in negative_words]

tweet_tokenizer = TweetTokenizer(
    preserve_case=False,
    strip_handles=True,
    reduce_len=True
)

tokens = []

# for tweet in x_train:
for i, tweet in enumerate(x_train):
    token = tweet_tokenizer.tokenize(tweet)
    token_clened = [word for word in token if (word not in english_stopwords and word not in string.punctuation)]
    if y_train[i]=="positive":
      label = 1
    elif y_train[i]=="negative":
      label = 0
    else:
      label = 2
    tokens.append([" ".join(token_clened), label])
    # tokens.append([tweet, label])


# In[63]:


df = pd.DataFrame(tokens, columns=["x_train", "y_train"])


# In[74]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(df["x_train"], df["y_train"], test_size=0.2, random_state=42)

train = pd.concat([X_train, y_train], axis = 1, join = 'outer', ignore_index=False, sort=False)
val = pd.concat([X_val, y_val], axis = 1, join = 'outer', ignore_index=False, sort=False)

train.to_csv('/content/drive/My Drive/Mila/ML/kaggle_2/data_clean/train_data.csv', index=False)
val.to_csv('/content/drive/My Drive/Mila/ML/kaggle_2/data_clean/val_data.csv', index=False)


# In[3]:


sequence_length = 280

vectorize_layer = layers.TextVectorization(
    # max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)


# In[10]:


EPOCHS = 30
BATCH_SIZE = 64
STEPS_PER_EPOCH = int(832259 / BATCH_SIZE)

train_dataset = tf.data.experimental.make_csv_dataset("/content/drive/My Drive/Mila/ML/kaggle_2/data_clean/train_data.csv",
  batch_size=BATCH_SIZE,
  num_epochs=EPOCHS,
  label_name="y_train",
  header=True,
)

val_dataset = tf.data.experimental.make_csv_dataset("/content/drive/My Drive/Mila/ML/kaggle_2/data_clean/val_data.csv",
  batch_size=BATCH_SIZE,
  num_epochs=EPOCHS,
  label_name="y_train",
  header=True,
)


# In[5]:


train_text = train_dataset.map(lambda x, y: x["x_train"])
vectorize_layer.adapt(train_text)


# In[6]:


max_features = len(vectorize_layer.get_vocabulary())
vectorize_layer(["hola hhola dsads bye adios hi hello how are you the a"])


# # Train

# In[11]:


embedding_dim = 16
model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.3),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.3),
  layers.Dense(3)])

model.summary()


# In[12]:


from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

early_stopping = callbacks.EarlyStopping(
    monitor='accuracy', 
    verbose=1,
    patience=4,
    mode='max',
    restore_best_weights=True)

model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=optimizers.Adam(learning_rate=1e-3),
              metrics=['accuracy'])


# In[13]:


def vectorize_text(text, label):
  text = tf.expand_dims(text["x_train"], -1)
  return vectorize_layer(text), label


# In[14]:


train_ds = train_dataset.map(vectorize_text)
val_ds = val_dataset.map(vectorize_text)


# ### Train the model
# 
# You will train the model by passing the `dataset` object to the fit method.

# In[15]:


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks = [early_stopping]
)


# In[16]:


model.save("/content/drive/My Drive/Mila/ML/kaggle_2/data/best_nn_.h5")


# In[17]:


history_dict = history.history
history_dict.keys()


# In[18]:


acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[19]:


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()


# # Test

# In[28]:


def get_test_data(path="/content/drive/My Drive/Mila/ML/kaggle_2/data/"):
  x_train = pd.read_csv(path + 'test_data.csv')
  return x_train["text"].tolist()

x_test = get_test_data()


# In[32]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

nltk.download('stopwords')
english_stopwords = stopwords.words("english")
english_stopwords.extend(["..", "...", "...."])
negative_words = ('no', 'not')
english_stopwords = [e for e in english_stopwords if e not in negative_words]

tweet_tokenizer = TweetTokenizer(
    preserve_case=False,
    strip_handles=True,
    reduce_len=True
)

tokens = []



# for tweet in x_train:
for i, tweet in enumerate(x_test):
    token = tweet_tokenizer.tokenize(tweet)
    token_clened = [word for word in token if (word not in english_stopwords and word not in string.punctuation)]
 
    tokens.append(" ".join(token_clened))

df = pd.DataFrame(tokens, columns=["x_test"])
df.to_csv('/content/drive/My Drive/Mila/ML/kaggle_2/data_clean/test_data.csv', index=False)


# In[33]:


test_dataset = tf.data.experimental.make_csv_dataset("/content/drive/My Drive/Mila/ML/kaggle_2/data_clean/test_data.csv",
  batch_size=64,
  num_epochs=1,
  # label_name="y_train",
  header=True,
)

def vectorize_test_text(text):
  text = tf.expand_dims(text["x_test"], -1)
  return vectorize_layer(text)

x_test = test_dataset.map(vectorize_test_text)


# In[34]:


y_pred = model.predict(x_test, batch_size=BATCH_SIZE)


# In[45]:


print(y_pred[0])
print(y_pred[1])
print(y_pred[2])


# In[63]:


df = pd.DataFrame(y_pred.argmax(axis=1), columns=["target"])
df["target"] = df["target"].replace(1, 2)

df.index.name = 'id'


# In[64]:


print(df)


# In[66]:


df.to_csv('/content/drive/My Drive/Mila/ML/kaggle_2/data_clean/nn.csv')

