
# coding: utf-8

# In[46]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
sentences = []
tweets = []
f = open("C://Users/bhavana/Desktop/code/dataset/train-broncos.csv","r")
for line in f:
    if(len(line.strip())>0):
        print(line)
        text = ' '.join(re.sub("([^@#0-9A-Za-z\t])|(\w+:\/\/\S+)"," ",line.split(";")[0]).split())
        sentences.append(text)
        tweets.append(text+";"+line.split(";")[1].strip()+";"+line.split(";")[2].strip())
sid = SentimentIntensityAnalyzer()
for i in range(len(sentences)):
    sentence = sentences[i]
    ss = sid.polarity_scores(sentence)
    tweet = ";" +str(ss['compound'])
    tweets[i] = tweets[i] + tweet + os.linesep
out = open("C://Users/bhavana/Desktop/code/dataset/train-broncos2.csv","w")
for tweet in tweets:
    out.write(tweet)
out.close()


# In[41]:


out.close()


# In[39]:


import re
f = open("C://Users/bhavana/Desktop/code/dataset/broncos-1-6-process.csv","r")
count = 0
for line in f:
    print(line.split(",")[0])
    text = line.split(",")[0]
    text = ' '.join(re.sub("([^@#0-9A-Za-z\'\"\t])|(\w+:\/\/\S+)"," ",text).split())
    print(text)
    count+=1
    if(count > 500):
        break


# In[31]:


import nltk
nltk.download()


# In[13]:


out.close()


# In[120]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout
import re
data = pd.read_csv('C://Users/bhavana/Desktop/code/dataset/train-broncos2.csv',sep=';')
data = data[['text','retweets','favorites','compound']]
data['text'] = data['text'].apply(lambda x: str(x).lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
max_fatures = 10000
tokenizer = Tokenizer(nb_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)
print(len(X[0]))



# In[121]:


data['compound'] = data['compound'].apply(lambda x: 1 if x > 0 else 0 )
Y = data['compound'].values
print(Y)


# In[122]:


embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, 128, input_length= X.shape[1]))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

'''
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(1,activation='softmax'))
model.compile( loss = 'mean_squared_error',optimizer='adam',metrics = ['accuracy'])
'''




# In[126]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
batch_size = 32
model.fit(X_train, Y_train, epochs = 5, batch_size=batch_size, verbose = 1)


# In[127]:


print(X_test)


# In[128]:


validation_size = 150

X_validate = X_test[validation_size:]
Y_validate = Y_test[validation_size:]
X_test = X_test[:validation_size]
Y_test = Y_test[:validation_size]
print(X_test[0],Y_test[0])
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))


# In[129]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
sentences = []
tweets = []
f = open("C://Users/bhavana/Desktop/code/dataset/test-broncos.csv","r")
for line in f:
    if(len(line.strip())>0):
        text = ' '.join(re.sub("([^@#0-9A-Za-z\t])|(\w+:\/\/\S+)"," ",line.split(";")[0]).split())
        sentences.append(text+os.linesep)
print(len(sentences))
    
out = open("C://Users/bhavana/Desktop/code/dataset/test-broncos2.csv","w")
for sentence in sentences:
    out.write(sentence)
out.close()


# In[130]:


test_data = pd.read_csv("C://Users/bhavana/Desktop/code/dataset/test-broncos2.csv")
test_data['text'] = test_data['text'].apply(lambda x: str(x).lower())
test_data['text'] = test_data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
max_fatures = 10000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(test_data['text'].values)
X_denver_test = tokenizer.texts_to_sequences(test_data['text'].values)
X_denver_test = pad_sequences(X_denver_test,maxlen=59)
print(len(X_denver_test[1]))


# In[136]:


output = open("C://Users/bhavana/Desktop/code/dataset/sentiments.csv","w")
count_pos = 0
count_neg = 0
for instance in range(len(X_denver_test)):
    result_Y = model.predict(X_denver_test[instance].reshape(1,59),batch_size=1,verbose = 2)[0]
    #print(test_data['text'][instance])
    #print(result_Y)
    if(result_Y[0]>0.5): count_pos += 1
    else: count_neg += 1
    output.write(str(result_Y[0]) +os.linesep)
print("-"+ str(count_neg) +","+ str(count_pos))
output.close()
    

