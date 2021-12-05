import sys, random, math
from collections import Counter
import numpy as np

from layers.crossentropyloss import CrossEntropyLoss
from layers.embedding import Embedding
from layers.rnncell import RNNCell
from optimization.sgd import SGD
from tensor.tensor import Tensor
#Подготовка входных данных
f = open('qa1_single-supporting-fact_train.txt', 'r')
raw = f.readlines()
f.close()

tokens = list()
for line in raw[0:1000]:
    tokens.append(line.lower().replace("\n", "").split(" ")[1:])

new_tokens = list()
for line in tokens:
    new_tokens.append(['-'] * (6 - len(line)) + line)

tokens = new_tokens

vocab = set()
for sent in tokens:
    for word in sent:
        vocab.add(word)

vocab = list(vocab)

word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i


def words2indices(sentence):
    idx = list()
    for word in sentence:
        idx.append(word2index[word])
    return idx


indices = list()
for line in tokens:
    idx = list()
    for w in line:
        idx.append(word2index[w])
    indices.append(idx)

data = np.array(indices)
#Архитектура
embed = Embedding(vocab_size=len(vocab),dim=16)
model = RNNCell(n_inputs=16, n_hidden=16, n_output=len(vocab))

criterion = CrossEntropyLoss()
optim = SGD(parameters=model.get_parameters() + embed.get_parameters(), alpha=0.05)

#Обучение
for iter in range(1000):
    batch_size = 100
    total_loss = 0

    hidden = model.init_hidden(batch_size=batch_size)

    for t in range(5):
        input = Tensor(data[0:batch_size, t], autograd=True)
        rnn_input = embed.forward(input=input)
        output, hidden = model.forward(input=rnn_input, hidden=hidden)

    target = Tensor(data[0:batch_size, t + 1], autograd=True)
    loss = criterion.forward(output, target)
    loss.backward()
    optim.step()
    total_loss += loss.data
    if (iter % 200 == 0):
        p_correct = (target.data == np.argmax(output.data, axis=1)).mean()
        print("Loss:", total_loss / (len(data) / batch_size), "% Correct:", p_correct)

#Тест обученной сети
batch_size = 1
hidden = model.init_hidden(batch_size=batch_size)
for t in range(5):
    input = Tensor(data[0:batch_size,t], autograd=True)
    rnn_input = embed.forward(input=input)
    output, hidden = model.forward(input=rnn_input, hidden=hidden)

target = Tensor(data[0:batch_size,t+1], autograd=True)
loss = criterion.forward(output, target)

ctx = ""
for idx in data[0:batch_size][0][0:-1]:
    ctx += vocab[idx] + " "
print("Context:",ctx)
print("True:",vocab[target.data[0]])
print("Pred:", vocab[output.data.argmax()])