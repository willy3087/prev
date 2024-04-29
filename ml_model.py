import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer

vetorizador = TfidfVectorizer(max_features=50)

def treinar_modelo(caminho_csv, nome_designer):
    logging.info(f"Iniciando treinamento do modelo para o designer {nome_designer}")
    df = pd.read_csv(caminho_csv)
    
    # Checar se o designer existe no DataFrame
    if nome_designer not in df['Nome'].values:
        logging.error(f"Designer '{nome_designer}' não encontrado.")
        return None

    df_filtrado = df[df['Nome'] == nome_designer]

    # Tratar valores NaN na coluna 'Descrição'
    df_filtrado['Descrição'].fillna("", inplace=True)

    # Tratar valores NaN nas colunas 'Estimativa' e 'Execução'
    df_filtrado['Estimativa'].fillna(0, inplace=True)
    df_filtrado['Execução'].fillna(0, inplace=True)

    # Vetorização das descrições
    descricoes_vetorizadas = vetorizador.fit_transform(df_filtrado['Descrição']).toarray()

    logging.info(f"Estatísticas dos dados: {df_filtrado.describe()}")

    X = np.hstack((descricoes_vetorizadas, df_filtrado[['Estimativa', 'Execução']].values))
    y = df_filtrado['Complexidade']

    modelo = LinearRegression().fit(X, y)
    
    logging.info("Modelo treinado com sucesso")
    return modelo

def prever_entropia(modelo, descricao, estimativa, execucao):
    if modelo is None:
        return "Modelo não disponível."

    descricao_vetorizada = vetorizador.transform([descricao]).toarray()
    X = np.hstack((descricao_vetorizada, np.array([[estimativa, execucao]])))
    return modelo.predict(X)[0]
