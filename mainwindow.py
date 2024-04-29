import sys
import logging
import pandas as pd
from fuzzywuzzy import process
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit, QLineEdit, QHBoxLayout
from csv_transformer import transformar_csv
from nlp_analysis import extrair_palavras_chave
from ml_model import treinar_modelo, prever_entropia

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Minha Aplicação")
        self.setGeometry(100, 100, 400, 300)
        
        self.layout = QVBoxLayout()
        
        self.label_tarefa = QLabel("Nome da tarefa:", self)
        self.layout.addWidget(self.label_tarefa)
        
        self.input_tarefa = QLineEdit(self)
        self.layout.addWidget(self.input_tarefa)
        
        self.label_designer = QLabel("Nome do designer:", self)
        self.layout.addWidget(self.label_designer)
        
        self.input_designer = QLineEdit(self)
        self.layout.addWidget(self.input_designer)
        
        self.label_descricao = QLabel("Descrição da tarefa:", self)
        self.layout.addWidget(self.label_descricao)
        
        self.input_descricao = QTextEdit(self)
        self.layout.addWidget(self.input_descricao)
        
        self.label_tempo = QLabel("Tempo estimado (em horas):", self)
        self.layout.addWidget(self.label_tempo)
        
        self.input_tempo = QLineEdit(self)
        self.layout.addWidget(self.input_tempo)
        
        self.label_complexidade = QLabel("Complexidade (1 a 5):", self)
        self.layout.addWidget(self.label_complexidade)
        
        self.input_complexidade = QLineEdit(self)
        self.layout.addWidget(self.input_complexidade)
        
        self.button_prever = QPushButton("Prever Entropia", self)
        self.button_prever.clicked.connect(self.prever_entropia)
        self.layout.addWidget(self.button_prever)
        
        self.label_resultado = QLabel("Resultado:", self)
        self.layout.addWidget(self.label_resultado)
        
        self.text_resultado = QTextEdit(self)
        self.text_resultado.setReadOnly(True)
        self.layout.addWidget(self.text_resultado)
        
        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

    def prever_entropia(self):
        nome_tarefa = self.input_tarefa.text()
        nome_designer = self.input_designer.text()
        descricao_tarefa = self.input_descricao.toPlainText()
        tempo_estimado = float(self.input_tempo.text())
        complexidade = int(self.input_complexidade.text())

        nome_designer_mais_proximo = encontrar_designer_mais_proximo(nome_designer, lista_designers)
        palavras_chave = extrair_palavras_chave(descricao_tarefa)
        print("Palavras-chave da descrição:", palavras_chave)

        modelo = treinar_modelo(caminho_csv_transformado, nome_designer_mais_proximo)

        if modelo:
            entropia_predita = prever_entropia(modelo, descricao_tarefa, tempo_estimado, complexidade)
            porcentagem_aproximacao = entropia_predita * 100
            resultado = f"A entropia prevista para a tarefa '{nome_tarefa}' é {entropia_predita:.2f}.\n" \
                        f"Isso representa uma aproximação de {porcentagem_aproximacao:.2f}% da realidade."
            self.text_resultado.setPlainText(resultado)
        else:
            self.text_resultado.setPlainText("Não foi possível treinar o modelo.")

def encontrar_designer_mais_proximo(nome, lista_designers):
    nome_mais_proximo, _ = process.extractOne(nome, lista_designers)
    return nome_mais_proximo

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    caminho_csv_original = '/Users/helbertwilliamduarteteixeira/prever/dados_designers__.csv'
    caminho_csv_transformado = '/Users/helbertwilliamduarteteixeira/prever/dados_designers_tarefas_atualizado.csv'

    transformar_csv(caminho_csv_original, caminho_csv_transformado)

    df = pd.read_csv(caminho_csv_transformado)
    lista_designers = df['Nome'].unique().tolist()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())