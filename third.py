import pandas as pd
from IPython.display import display

#read in the dataset
ion = pd.read_csv('ionosphere.data', delimiter=",")
ion.columns = [f"V{i}" for i in range(1, 35)] + ["Class"]
display(ion.head())

df = ion.copy()
df['Class'] = df['Class'].map({'g': 0, 'b': 1})

df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)

df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)
df_train.dropna(axis=1, inplace=True) # drop the empty feature in column 2
df_valid.dropna(axis=1, inplace=True)

X_train = df_train.drop('Class', axis=1)
X_valid = df_valid.drop('Class', axis=1)
y_train = df_train['Class']
y_valid = df_valid['Class']