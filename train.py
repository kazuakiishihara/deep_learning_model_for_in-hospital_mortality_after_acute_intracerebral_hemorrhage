import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch

import trainer

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(178)

#--------------------------DataFrame

df = pd.read_csv('./ich.csv', encoding='shift-jis')
df.shape

df = df[df['analysis_headCT_num'] != 0]
df.shape

df = df[(df['analysis_headCT_num'] >= 1) & (df['analysis_headCT_num'] <= 45)]
df.shape

df_train, df_test = df[df['train_or_test'] == 'train'], df[df['train_or_test'] == 'test']
print(df_train['obj'].value_counts())
print(df_test['obj'].value_counts())

df_train, df_valid = train_test_split(df_train, test_size=0.3, stratify=df_train['obj'], random_state=22)
df_train.shape
df_valid.shape

#--------------------------Train
modelfiles = None

if not modelfiles:
    modelfiles = trainer.train_run(df_train, df_valid)
    print(modelfiles)

