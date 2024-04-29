import pandas as pd
import re

def remove_links_emails_html(texto):
    # Convertendo qualquer valor não-string para string
    if not isinstance(texto, str):
        texto = str(texto)

    # Usando regex para remover links, emails e tags HTML
    clean_text = re.sub('<.*?>', '', texto)  # Remove tags HTML
    clean_text = re.sub(r'https?://[^\s]+', '', clean_text)  # Remove links
    clean_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', clean_text)  # Remove emails
    return clean_text

def transformar_csv(input_path, output_path):
    df = pd.read_csv(input_path)

    # Removendo colunas indesejadas
    df.drop(['ID', 'Work Item Type', 'State'], axis=1, inplace=True)

    # Renomeando as colunas restantes
    df.rename(columns={
        'Assigned To': 'Nome',
        'Title': 'Título',
        'Description': 'Descrição',
        'Tags': 'Cliente',
        'Original Estimate': 'Estimativa',
        'Effort': 'Execução',
        'Priority': 'Complexidade'
    }, inplace=True)

    # Removendo links, emails e tags HTML de todas as colunas
    for col in df.columns:
        df[col] = df[col].apply(remove_links_emails_html)

    # Salvando o arquivo CSV modificado
    df.to_csv(output_path, index=False)
