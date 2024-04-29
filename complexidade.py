import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def analisar_descricao_nlp(descricao):
    nltk.download('punkt')
    nltk.download('stopwords')

    # Palavras-chave que indicam alta complexidade
    palavras_chave_alta = set(['avançado', 'complexo', 'desafiador'])
    # Palavras-chave que indicam baixa complexidade
    palavras_chave_baixa = set(['simples', 'básico', 'fácil'])

    # Tokenizar e converter para minúsculas
    palavras = word_tokenize(descricao.lower())
    # Remover stopwords
    stop_words = set(stopwords.words('portuguese'))
    palavras_filtradas = [palavra for palavra in palavras if palavra not in stop_words]

    # Verificar a presença de palavras-chave
    ajuste_complexidade = 0
    for palavra in palavras_filtradas:
        if palavra in palavras_chave_alta:
            ajuste_complexidade += 1  # Aumenta a complexidade
        elif palavra in palavras_chave_baixa:
            ajuste_complexidade -= 1  # Diminui a complexidade

    return ajuste_complexidade

# Exemplo de uso da função
descricao_exemplo = "Este é um projeto de design avançado e complexo."
ajuste = analisar_descricao_nlp(descricao_exemplo)
print(ajuste)  # Exibirá um valor positivo indicando alta complexidade


"""from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Supondo que você tenha um DataFrame com descrições de tarefas e suas complexidades
df = pd.read_csv('seu_dataset.csv')  # Este CSV deve ter colunas de 'descrição' e 'complexidade'

# Preparando os dados para TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Ajuste o número de características conforme necessário
X = vectorizer.fit_transform(df['descrição'])
y = df['complexidade']  # As categorias de complexidade

# Dividir os dados para treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar um modelo de classificação
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Avaliar o modelo
precisao = modelo.score(X_test, y_test)
print(f"Precisão do modelo: {precisao}")

# Usar o modelo para prever a complexidade de uma nova descrição
def prever_complexidade(descricao):
    vetor_descricao = vectorizer.transform([descricao])
    return modelo.predict(vetor_descricao)[0]

# Exemplo de uso
descricao_nova = "Descreva aqui a nova tarefa."
complexidade_predita = prever_complexidade(descricao_nova)
print(f"Complexidade predita: {complexidade_predita}")
"""