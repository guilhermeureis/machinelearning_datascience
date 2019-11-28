import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')
base.describe()
# DADOS INCONSISTENTES
# Tratar base de dados
# Retirada de idade negativo
base.loc[base['age'] < 0]
# apagar a coluna
base.drop('age', 1, inplace=True)
# apagar somente os registros com problema
base.drop(base[base.age < 0].index, inplace=True)
# preencher os valores manualmente
# preencher os valores com a média
base.mean()
base['age'].mean()
# Substituir os valores negativos com a média
mediaAge = base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = mediaAge


# DADOS FALTANTES

pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:,0:3])