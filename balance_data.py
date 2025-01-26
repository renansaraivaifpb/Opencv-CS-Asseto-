# balance_data.py

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data = np.load('training_data.npy', allow_pickle=True)

df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

# Atualizando para reconhecer a classe 'A' (caso [0, 0, 1])
wa = []
rights = []
w = []
a_key = []  # Lista para a nova classe 'A'
d_key = []
s_key = []

shuffle(train_data)

# Seguindo a lógica de adicionar as amostras para cada classe
for data in train_data:
    img = data[0]
    choice = data[1]
    # wa
    if choice == [0,1,0,0,0,0]:
        wa.append([img, choice])
    # w
    elif choice == [1,0,0,0,0,0]:
        w.append([img, choice])
    # a
    elif choice == [0,0,0,1,0,0]:
        a_key.append([img, choice])  # Adicionando à nova lista para 'A'
    # wd
    elif choice == [0,0,1,0,0,0]:
        rights.append([img, choice])
    # d
    elif choice == [0,0,0,0,1,0]:
        d_key.append([img, choice])
    # s
    elif choice == [0,0,0,0,0,1]:
        s_key.append([img, choice])
    else:
        print('no matches')

# Encontrando o número mínimo de amostras entre as classes
min_len = min(len(wa), len(rights), len(w), len(a_key), len(d_key), len(s_key))

# Balanceando as classes para terem o mesmo número de amostras
w = w[:min_len]
wa = wa[:min_len]
rights = rights[:min_len]
a_key = a_key[:min_len]  # Balanceando também a nova classe
d_key = d_key[:min_len]
s_key = s_key[:min_len]

print(len(w), len(wa), len(rights), len(a_key), len(d_key), len(s_key))

# Unindo as classes balanceadas
final_data = w + wa + rights + a_key + d_key + s_key# Incluindo 'A' na unificação

# Embaralhando o conjunto de dados final
shuffle(final_data)

# Salvando os dados balanceados
np.save('training_data.npy', final_data)
