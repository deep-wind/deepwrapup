import numpy as np
import os
import pandas as pd
import re
bbc_df = pd.read_csv(r"C:\Users\PRAMILA\Downloads\Text_Summarization_using_Bidirectional_LSTM-master\Text_Summarization_using_Bidirectional_LSTM-master\cleaned_bbc_news.csv")
len(bbc_df)

def cleantext(text):
    text = str(text)
    text=text.split()
    words=[]
    for t in text:
        if t.isalpha():
            words.append(t)
    text=" ".join(words)
    text=text.lower()
    text=re.sub(r"what's","what is ",text)
    text=re.sub(r"it's","it is ",text)
    text=re.sub(r"\'ve"," have ",text)
    text=re.sub(r"i'm","i am ",text)
    text=re.sub(r"\'re"," are ",text)
    text=re.sub(r"n't"," not ",text)
    text=re.sub(r"\'d"," would ",text)
    text=re.sub(r"\'s","s",text)
    text=re.sub(r"\'ll"," will ",text)
    text=re.sub(r"can't"," cannot ",text)
    text=re.sub(r" e g "," eg ",text)
    text=re.sub(r"e-mail","email",text)
    text=re.sub(r"9\\/11"," 911 ",text)
    text=re.sub(r" u.s"," american ",text)
    text=re.sub(r" u.n"," united nations ",text)
    text=re.sub(r"\n"," ",text)
    text=re.sub(r":"," ",text)
    text=re.sub(r"-"," ",text)
    text=re.sub(r"\_"," ",text)
    text=re.sub(r"\d+"," ",text)
    text=re.sub(r"[$#@%&*!~?%{}()]"," ",text)
    
    return text



import numpy as np
import os
import pandas as pd
import re

bbc_art_sum = pd.read_csv("cleaned_bbc_news.csv")
bbc_art_sum.drop("Unnamed: 0", axis=1, inplace=True)
bbc_art_sum.head()

articles = list(bbc_art_sum.row_article)
summaries = list(bbc_art_sum.summary)

from keras.preprocessing.text import Tokenizer
VOCAB_SIZE = 14999
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(articles)
article_sequences = tokenizer.texts_to_sequences(articles)
art_word_index = tokenizer.word_index
len(art_word_index)

print(article_sequences[0][:20])
print(article_sequences[1][:20])
print(article_sequences[2][:20])

art_word_index_1500 = {}
counter = 0
for word in art_word_index.keys():
    if art_word_index[word] == 0:
        print("found 0!")
        break
    if art_word_index[word] > VOCAB_SIZE:
        continue
    else:
        art_word_index_1500[word] = art_word_index[word]
        counter += 1

counter

tokenizer.fit_on_texts(summaries)
summary_sequences = tokenizer.texts_to_sequences(summaries)
sum_word_index = tokenizer.word_index
len(sum_word_index)

sum_word_index_1500 = {}
counter = 0
for word in sum_word_index.keys():
    if sum_word_index[word] == 0:
        print("found 0!")
        break
    if sum_word_index[word] > VOCAB_SIZE:
        continue
    else:
        sum_word_index_1500[word] = sum_word_index[word]
        counter += 1

counter

from keras.preprocessing.sequence import pad_sequences
MAX_LEN = 1000
pad_art_sequences = pad_sequences(article_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

print(len(article_sequences[1]), len(pad_art_sequences[1]))

pad_sum_sequences = pad_sequences(summary_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

print(len(summary_sequences[1]), len(pad_sum_sequences[1]))

pad_art_sequences.shape

pad_art_sequences

decoder_outputs = np.zeros((2225, 1000, 15000), dtype='float32')
decoder_outputs.shape

for i, seqs in enumerate(pad_sum_sequences):
    for j, seq in enumerate(seqs):
        decoder_outputs[i, j, seq] = 1.

decoder_outputs.shape

### 2-2-5. Pre-trained word2vec and word2vec Matrix

embeddings_index = {}
with open('glove.6B.200d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

print('Found %s word vectors.' % len(embeddings_index))

def embedding_matrix_creater(embedding_dimention, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimention))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

art_embedding_matrix = embedding_matrix_creater(200, word_index=art_word_index_1500)
art_embedding_matrix.shape

sum_embedding_matrix = embedding_matrix_creater(200, word_index=sum_word_index_1500)
sum_embedding_matrix.shape

# encoder_embedding_layer = Embedding(input_dim = 15000, 
#                                     output_dim = 200,
#                                     input_length = MAX_LEN,
#                                     weights = [art_embedding_matrix],
#                                     trainable = False)

# decoder_embedding_layer = Embedding(input_dim = 15000, 
#                                     output_dim = 200,
#                                     input_length = MAX_LEN,
#                                     weights = [sum_embedding_matrix],
#                                     trainable = False)

sum_embedding_matrix.shape