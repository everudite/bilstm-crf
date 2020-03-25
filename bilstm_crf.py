import pandas as pd
import numpy as np
import SentenceGetter as sg
import matplotlib.pyplot as plt
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

#process input data
def process_input(filename):
	data = pd.read_csv(filename, encoding="latin1")
	data = data.fillna(method="ffill")

	words = list(set(data["Word"].values))
	#words.append("ENDPAD")

	tags = list(set(data["Tag"].values))

	getter = sg.SentenceGetter(data)
	#sent = getter.get_next()
	sentences = getter.sentences
	#print(sent)
	return (words, tags, sentences)

words, tags, sentences = process_input("ner_dataset.csv")
n_words = len(words);
n_tags = len(tags);
n_sentences = len(sentences);
print("# of words: {}, # of tags: {}, # of sentences: {}".format(n_words, n_tags, n_sentences))

#one-hot
max_pad_len = 50
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t:i for i, t in enumerate(tags)}

X = [[word2idx[w[0]] for w in s] for s in sentences]
y = [[tag2idx[w[2]] for w in s] for s in sentences]

from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(maxlen=max_pad_len, sequences=X, padding="post", value=0)
y = pad_sequences(maxlen=max_pad_len, sequences=y, padding="post", value=tag2idx["O"])

from keras.utils import to_categorical
y = [to_categorical(i, num_classes=n_tags) for i in y]

from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

#print("X_tr has shape " + X_tr.shape)
#print("Y_tr has shape " + Y_tr.shape)

#word2vec
# w2v_dim = 100
# wv_from_text = KeyedVectors.load_word2vec_format(datapath('/Users/yongbo/Downloads/word2vec/trunk/output/vectors.txt'), binary=False)
# X_tr_vectors = [[wv_from_text[w[0]]] for w in words]
# print("X_tr_vectors has shape " + X_tr_vectors.shape)


#model
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF

input = Input(shape=(max_pad_len,w2v_dim))

input = Input(shape=(max_pad_len,))
model = Embedding(input_dim=n_words + 1, output_dim=20,
                  input_length=max_pad_len)(input)  # 20-dim embedding

model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(n_tags, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output

model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()

history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=1,
                    validation_split=0.1, verbose=1)
hist = pd.DataFrame(history.history)

# plt.style.use("ggplot")
# plt.figure(figsize=(12,12))
# plt.plot(hist["acc"])
# plt.plot(hist["val_acc"])
# plt.show()

test_pred = model.predict(X_te, verbose=1)

idx2tag = {i: w for w, i in tag2idx.items()}

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out


pred_labels = pred2label(test_pred)
test_labels = pred2label(y_te)

print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))
print(classification_report(test_labels, pred_labels))

i = 227
p = model.predict(np.array([X_te[i]]))
p_index = np.argmax(p, axis=-1)
true_index = np.argmax(y_te[i], -1)
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(X_te[i], true_index, p_index[0]):
    if w != 0:
        print("{:15}: {:5} {}".format(words[w-1], tags[t], tags[pred]))