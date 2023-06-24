# Import das bibliotecas.

# Biblioteca de logging
import logging  
# Biblioteca de download
import requests 
# Biblioteca para barra de progresso
from tqdm.notebook import tqdm as tqdm_notebook 
# Biblioteca para manipular arquivos
import os 

# Bibliotecas próprias
from textotransformer.util.utilconstantes import DIRETORIO_TEXTO_TRANSFORMER
from textotransformer.util.utiltexto import removeTags

logger = logging.getLogger(__name__)

# ============================  
def verificaDiretorioTextoTransformer():
    '''    
    Verifica se existe o diretório DIRETORIO_TEXTO_TRANSFORMER no diretório corrente.    
    '''
    
    # Verifica se o diretório existe
    if not os.path.exists(DIRETORIO_TEXTO_TRANSFORMER):  
        # Cria o diretório
        os.makedirs(DIRETORIO_TEXTO_TRANSFORMER)
        logger.info("Diretório \"{}\" criado.".format(DIRETORIO_TEXTO_TRANSFORMER))
    
    return DIRETORIO_TEXTO_TRANSFORMER

# ============================  
def downloadArquivo(url_arquivo: str, nome_arquivo_destino: str):
    '''    
    Realiza o download de um arquivo de uma url em salva em nome_arquivo_destino.
    
    Parâmetros:
       `url_arquivo` - URL do arquivo a ser feito download.      
       `nome_arquivo_destino` - Nome do arquivo a ser salvo.      
    '''
    
    # Verifica se existe o diretório base
    DIRETORIO_TEXTO_TRANSFORMER = verificaDiretorioTextoTransformer()
    
    # Realiza o download de um arquivo em uma url
    data = requests.get(url_arquivo, stream=True)
    
    # Verifica se o arquivo existe
    if data.status_code != 200:
        logger.info("Exceção ao tentar realizar download \"{}\". Response \"{}\".".format(url_arquivo, data.status_code))
        data.raise_for_status()
        return

    # Recupera o nome do arquivo a ser realizado o download    
    nome_arquivo = nome_arquivo_destino.split('/')[-1]  

    # Define o nome e caminho do arquivo temporário    
    nome_arquivo_temporario = DIRETORIO_TEXTO_TRANSFORMER + "/" + nome_arquivo + "_part"
    
    logger.info("Download do arquivo: \"{}\".".format(nome_arquivo_destino))
    
    # Baixa o arquivo
    with open(nome_arquivo_temporario, "wb") as arquivo_binario:        
        tamanho_conteudo = data.headers.get('Content-Length')        
        total = int(tamanho_conteudo) if tamanho_conteudo is not None else None
        # Barra de progresso de download
        progresso_bar = tqdm_notebook(unit="B", total=total, unit_scale=True)                
        # Atualiza a barra de progresso
        for chunk in data.iter_content(chunk_size=1024):        
            if chunk:                
                progresso_bar.update(len(chunk))
                arquivo_binario.write(chunk)
    
    # Renomeia o arquivo temporário para o arquivo definitivo
    os.rename(nome_arquivo_temporario, nome_arquivo_destino)
    
    # Fecha a barra de progresso.
    progresso_bar.close()

# ============================      
def carregar(nome_arquivo: str):
    '''
    Carrega um arquivo texto e retorna as linhas como um único parágrafo(texto).
    
    Parâmetros:
       `nome_arquivo` - Nome do arquivo a ser carregado.           
    '''
        
    # Abre o arquivo
    arquivo = open(nome_arquivo, 'r')
    
    paragrafo = ''
    for linha in arquivo:
        linha = linha.splitlines()
        linha = ' '.join(linha)
        # Remove as tags existentes no final das linhas
        linha = removeTags(linha)
        if linha != '':
            paragrafo = paragrafo + linha.strip() + ' '
     
    # Fecha o arquivo
    arquivo.close()
    
    # Remove os espaços em branco antes e depois do parágrafo
    return paragrafo.strip()

# ============================  
def carregarLista(nome_arquivo: str):
    '''
    Carrega um arquivo texto e retorna as linhas como uma lista de sentenças(texto).
    
    Parâmetros:
       `nome_arquivo` - Nome do arquivo a ser carregado.           
    '''

    # Abre o arquivo
    arquivo = open(nome_arquivo, 'r')
    
    sentencas = []
    for linha in arquivo:        
        linha = linha.splitlines()
        linha = ' '.join(linha)
        linha = removeTags(linha)
        if linha != '':
            sentencas.append(linha.strip())
            
    # Fecha o arquivo
    arquivo.close()
    
    return sentencas    

# ============================      
def salvar(nome_arquivo: str, texto: str):                       
    '''
    Salva um texto em um arquivo.
     
    Parâmetros:
       `nome_arquivo` - Nome do arquivo a ser salvo.     
       `texto` - Texto a ser salvo.     
     '''

    arquivo = open(nome_arquivo, 'w')
    arquivo.write(str(texto))
    arquivo.close() 