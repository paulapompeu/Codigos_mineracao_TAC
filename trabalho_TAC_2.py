# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 18:43:19 2016

@author: paula.fiuza
"""
import numpy as np
import statsmodels.api as sm
import pandas as pd
import patsy as ps
import matplotlib.pyplot as plt

from sklearn import linear_model

base_TAC = pd.read_csv('Base_mineracao_final_1.csv')
base_TAC.columns= ['CD_IBGE6','CD_IBGE7','CD_UF','SIGLA_UF','regiao','Porte_mean','IDHM','prop_rural','primeira_dama','escolaridade_secretario','tem_plano_assistencia','prop_entrevist_perm_2013_2014','cadastro_equipes_servicos','prop_posto_cras','veiculo_proprio','utilizacao_definicao_publico_alvo','utilizacao_diagnostico','utilizacao_trabalho_infantil','utilizacao_situacao_rua','utilizacao_deficiencia','utilizacao_gpte','utilizacao_outros','prop_pbf','TAC_maio_2014']


# embaralha as amostras


base_TAC_2 = base_TAC.iloc[np.random.permutation(len(base_TAC))]

# faz a matriz

y,X=ps.dmatrices('TAC_maio_2014~regiao+Porte_mean+IDHM+prop_rural+primeira_dama+escolaridade_secretario+tem_plano_assistencia+prop_entrevist_perm_2013_2014+cadastro_equipes_servicos+prop_posto_cras+veiculo_proprio+utilizacao_definicao_publico_alvo+utilizacao_diagnostico+utilizacao_trabalho_infantil+utilizacao_situacao_rua+utilizacao_deficiencia+utilizacao_gpte+utilizacao_outros+prop_pbf', data=base_TAC_2, return_type='dataframe')

#faz a regressão de teste e a regressão de treino, calcula o valor predito e compara o predito com o valor real

linear_teste = linear_model.LinearRegression ()
linear_teste.fit(X[1500:],y[1500:])
linear_treino = linear_model.LinearRegression ()
linear_treino.fit(X[:1500],y[:1500])
predicted_teste=linear_teste.predict(X[1500:])
predicted_treino=linear_treino.predict(X[:1500])
plt.scatter(x=y[1500:],
           y=predicted_teste)
plt.show()

# faz a regressão de um jeito que apresenta melhor os resultados

model_teste = sm.OLS(y[1500:],X[1500:])
results_teste = model_teste.fit()
print(results_teste.summary())

model_treino = sm.OLS(y[:1500],X[:1500])
results_treino = model_treino.fit()
print(results_treino.summary())





