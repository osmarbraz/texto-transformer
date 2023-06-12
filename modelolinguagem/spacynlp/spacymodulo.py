# Import das bibliotecas.
import logging  # Biblioteca de logging
import tarfile # Biblioteca de descompactação
import os # Biblioteca de manipulação de arquivos
import spacy # Biblioteca do spaCy

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *

# ============================
def getStopwords(nlp):
    '''
    Recupera as stop words do nlp(Spacy).
    
    Parâmetros:
    `nlp` - Um modelo spaCy carregado.           
    '''
    
    spacy_stopwords = nlp.Defaults.stop_words

    return spacy_stopwords 

# ============================
def downloadSpacy(model_args):
    '''
    Realiza o download do arquivo do modelo para o diretório corrente.
    
    Parâmetros:
    `model_args` - Objeto com os argumentos do modelo.       
    '''
    # Verifica se existe o diretório base
    DIRETORIO_MODELO_LINGUAGEM = verificaDiretorioModeloLinguagem()
        
    # Nome arquivo spacy
    ARQUIVO_MODELO_SPACY = model_args.modelo_spacy
    # Versão spaCy
    VERSAO_SPACY = "-" + model_args.versao_spacy
    # Nome arquivo compactado
    NOME_ARQUIVO_MODELO_COMPACTADO = ARQUIVO_MODELO_SPACY + VERSAO_SPACY + ".tar.gz"
    
    # Url do arquivo
    URL_ARQUIVO_MODELO_COMPACTADO = "https://github.com/explosion/spacy-models/releases/download/" + ARQUIVO_MODELO_SPACY + VERSAO_SPACY + "/" + NOME_ARQUIVO_MODELO_COMPACTADO

    # Realiza o download do arquivo do modelo
    logging.info("Download do arquivo do modelo do spaCy.")
    downloadArquivo(URL_ARQUIVO_MODELO_COMPACTADO, DIRETORIO_MODELO_LINGUAGEM + "/" + NOME_ARQUIVO_MODELO_COMPACTADO)

# ============================   
def descompactaSpacy(model_args):
    '''
    Descompacta o arquivo do modelo.
    
    Parâmetros:
    `model_args` - Objeto com os argumentos do modelo.       
    '''
    
    # Verifica se existe o diretório base do cohebert e retorna o nome do diretório
    DIRETORIO_MODELO_LINGUAGEM = verificaDiretorioModeloLinguagem()
    
    # Nome arquivo spacy
    ARQUIVO_MODELO_SPACY = model_args.modelo_spacy
    # Versão spaCy
    VERSAO_SPACY = "-" + model_args.versao_spacy
    
    # Nome do arquivo a ser descompactado
    NOME_ARQUIVO_MODELO_COMPACTADO = DIRETORIO_MODELO_LINGUAGEM + "/" + ARQUIVO_MODELO_SPACY + VERSAO_SPACY + ".tar.gz"
    
    logging.info("Descompactando o arquivo do modelo do spaCy.")
    arquivoTar = tarfile.open(NOME_ARQUIVO_MODELO_COMPACTADO, "r:gz")    
    arquivoTar.extractall(DIRETORIO_MODELO_LINGUAGEM)    
    arquivoTar.close()
    
    # Apaga o arquivo compactado
    if os.path.isfile(NOME_ARQUIVO_MODELO_COMPACTADO):
        os.remove(NOME_ARQUIVO_MODELO_COMPACTADO)
    
# ============================    
def carregaSpacy(model_args):
    '''
    Realiza o carregamento do Spacy.
    
    Parâmetros:
    `model_args` - Objeto com os argumentos do modelo.           
    '''
    
    # Verifica se existe o diretório base
    DIRETORIO_MODELO_LINGUAGEM = verificaDiretorioModeloLinguagem()
                  
    # Nome arquivo spacy
    ARQUIVO_MODELO_SPACY = model_args.modelo_spacy
    # Versão spaCy
    VERSAO_SPACY = "-" + model_args.versao_spacy
    # Caminho raiz do modelo do spaCy
    DIRETORIO_MODELO_SPACY =  DIRETORIO_MODELO_LINGUAGEM + "/" + ARQUIVO_MODELO_SPACY + VERSAO_SPACY

    # Verifica se o diretório existe
    if os.path.exists(DIRETORIO_MODELO_SPACY) == False:
        # Realiza o download do arquivo modelo do spaCy
        downloadSpacy(model_args)
        # Descompacta o spaCy
        descompactaSpacy(model_args)

    # Diretório completo do spaCy
    DIRETORIO_MODELO_SPACY = DIRETORIO_MODELO_LINGUAGEM + "/" + ARQUIVO_MODELO_SPACY + VERSAO_SPACY + '/' + ARQUIVO_MODELO_SPACY + '/' + ARQUIVO_MODELO_SPACY + VERSAO_SPACY + '/'

    # Carrega o spaCy. Necessário somente 'tagger' para encontrar os substantivos
    nlp = spacy.load(DIRETORIO_MODELO_SPACY, disable=['tokenizer', 'lemmatizer', 'ner', 'parser', 'textcat', 'custom'])
    logging.info("spaCy carregado.")

    # Retorna o spacy carregado
    return nlp   
