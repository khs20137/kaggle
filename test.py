import os
import pandas as pd
from transformers import pipeline, AutoTokenizer


pd.set_option('max_columns', 1000)
pd.set_option('max_columns', 1000)


def get_text(path):
    f = open(path, 'r', encoding='utf-8')
    for row in f:
        print(row.strip())


df = pd.read_csv('train.csv')
# print(df[df.id == '423A1CA112E2'])
# print(df[df.id == '423A1CA112E2'].discourse_text[0])
# print(df[df.id == '423A1CA112E2'].predictionstring[0])
print(df)

# print(set(df.discourse_type.values))
# {'Counterclaim', 'Evidence', 'Lead', 'Claim', 'Concluding Statement', 'Position', 'Rebuttal'}

# Modern humans today are always on their phone. They are always on their phone more than 5 hours a day no stop .All they do is text back and forward and just have group Chats on social media. They even do it while driving.
# 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44
# print('\n')
# print(text_path)

path_li = []
for n, text_path in enumerate(os.listdir('train')):
    file_path = os.path.join('train', text_path)
    # print(n, file_path)
    path_li.append(file_path)

# print([f[6:18] for f in path_li[:10]])
# print([df[df.id == f[6:18]] for f in path_li[:10]][0])
# print([df[df.id == f[6:18]] for f in path_li[:10]][0].discourse_text.values[0])
# print([df[df.id == f[6:18]] for f in path_li[:10]][0].discourse_text.values[1])
# print([df[df.id == f[6:18]] for f in path_li[:10]][0].discourse_text.values[2])
# print([df[df.id == f[6:18]] for f in path_li[:10]][0].discourse_text.values[3])
# print([df[df.id == f[6:18]] for f in path_li[:10]][0].discourse_text.values[4])
# print([df[df.id == f[6:18]] for f in path_li[:10]][0].discourse_text.values[5])
# print([df[df.id == f[6:18]] for f in path_li[:10]][0].discourse_text.values[6])
# print([df[df.id == f[6:18]] for f in path_li[:10]][0].discourse_text.values[7])
#
# print('\n')
#
# [get_text(f) for f in path_li[:1]]

