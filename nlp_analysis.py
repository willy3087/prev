# nlp_analysis.py

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def extrair_palavras_chave(texto, num_palavras=10):
    palavras = word_tokenize(texto)
    stop_words = set(stopwords.words('portuguese'))
    palavras_filtradas = [palavra for palavra in palavras if palavra.lower() not in stop_words]
    frequencia = FreqDist(palavras_filtradas)
    return frequencia.most_common(num_palavras)

def extrair_atributos(descricao):
    palavras_chave = extrair_palavras_chave(descricao, num_palavras=5)
    atributos = [palavra for palavra, _ in palavras_chave]
    return atributos

def prever_configuracao_ideal(modelo, atributos):
    # Implemente a lógica para prever a configuração ideal usando o modelo treinado
    # Exemplo simples:
    tempo_ideal = modelo.predict_tempo(atributos)
    complexidade_ideal = modelo.predict_complexidade(atributos)
    return tempo_ideal, complexidade_ideal

def treinar_modelo_ideal(caminho_csv_transformado):
    # Carregar os dados do CSV
    df = pd.read_csv(caminho_csv_transformado)
    
    # Extrair as descrições das tarefas
    descricoes = df['Descrição'].tolist()
    
    # Extrair os valores de tempo e complexidade
    tempos = df['Execução'].tolist()
    complexidades = df['Complexidade'].tolist()
    
    # Criar o vetorizador TF-IDF
    vectorizer = TfidfVectorizer()
    
    # Vetorizar as descrições das tarefas
    descricoes_vetorizadas = vectorizer.fit_transform(descricoes)
    
    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train_tempo, y_test_tempo, y_train_complexidade, y_test_complexidade = train_test_split(
        descricoes_vetorizadas, tempos, complexidades, test_size=0.2, random_state=42
    )
    
    # Criar os modelos de regressão
    modelo_tempo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_complexidade = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Treinar os modelos
    modelo_tempo.fit(X_train, y_train_tempo)
    modelo_complexidade.fit(X_train, y_train_complexidade)
    
    # Retornar os modelos treinados e o vetorizador
    return modelo_tempo, modelo_complexidade, vectorizer