# balance_data.py

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('training_data.npy', allow_pickle=True)

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

lefts = []
rights = []
forwards = []

shuffle(train_data)

# Seguindo a lógica de adicionar as amostras para cada classe
for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1, 0, 0]:
        lefts.append([img, choice])
    elif choice == [0, 1, 0]:
        forwards.append([img, choice])
    elif choice == [0, 0, 1]:
        rights.append([img, choice])
    else:
        print('no matches')

# Encontrando o número mínimo de amostras entre as classes
min_len = min(len(lefts), len(rights), len(forwards))

# Balanceando as classes para terem o mesmo número de amostras
forwards = forwards[:min_len]
lefts = lefts[:min_len]
rights = rights[:min_len]


print(len(forwards), len(lefts), len(rights))

# Unindo as classes balanceada
final_data = forwards + lefts + rights

# Embaralhando o conjunto de dados final
shuffle(final_data)

# Salvando os dados balanceados
np.save('training_data.npy', final_data)
