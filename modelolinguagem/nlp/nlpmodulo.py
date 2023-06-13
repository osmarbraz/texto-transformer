# Import das bibliotecas.
import logging  # Biblioteca de logging
import tarfile # Biblioteca de descompactação
import os # Biblioteca de manipulação de arquivos
import spacy # Biblioteca do spaCy

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *

class PLN():

    ''' 
    Realiza o processamento de linguagem natural.
     
    Parâmetros:
    `nome_pln` - Nome da ferramenta de PLN a ser carregada.    
    ''' 

    # Construtor da classe
    def __init__(self, model_args):
    
        # Parâmetros do modelo
        self.model_args = model_args
        
        carregaSpacy()
        
    
    # ============================
    def getStopwords(self):
        '''
        Recupera as stop words do nlp(Spacy).
        
        Parâmetros:
        `nlp` - Um modelo spaCy carregado.           
        '''
        
        spacy_stopwords = nlp.Defaults.stop_words

        return spacy_stopwords 

    # ============================
    def downloadSpacy(self):
        '''
        Realiza o download do arquivo do modelo para o diretório corrente.
    
        '''
        # Verifica se existe o diretório base
        DIRETORIO_MODELO_LINGUAGEM = verificaDiretorioModeloLinguagem()
            
        # Nome arquivo spacy
        ARQUIVO_MODELO_SPACY = self.model_args.modelo_spacy
        # Versão spaCy
        VERSAO_SPACY = "-" + self.model_args.versao_spacy
        # Nome arquivo compactado
        NOME_ARQUIVO_MODELO_COMPACTADO = ARQUIVO_MODELO_SPACY + VERSAO_SPACY + ".tar.gz"
        
        # Url do arquivo
        URL_ARQUIVO_MODELO_COMPACTADO = "https://github.com/explosion/spacy-models/releases/download/" + ARQUIVO_MODELO_SPACY + VERSAO_SPACY + "/" + NOME_ARQUIVO_MODELO_COMPACTADO

        # Realiza o download do arquivo do modelo
        logging.info("Download do arquivo do modelo do spaCy.")
        downloadArquivo(URL_ARQUIVO_MODELO_COMPACTADO, DIRETORIO_MODELO_LINGUAGEM + "/" + NOME_ARQUIVO_MODELO_COMPACTADO)

    # ============================   
    def descompactaSpacy(self):
        '''
        Descompacta o arquivo do modelo.
     
        '''
        
        # Verifica se existe o diretório base do cohebert e retorna o nome do diretório
        DIRETORIO_MODELO_LINGUAGEM = verificaDiretorioModeloLinguagem()
        
        # Nome arquivo spacy
        ARQUIVO_MODELO_SPACY = self.model_args.modelo_spacy
        # Versão spaCy
        VERSAO_SPACY = "-" + self.model_args.versao_spacy
        
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
    def carregaSpacy(self):
        '''
        Realiza o carregamento do Spacy.
      
        '''
        
        # Verifica se existe o diretório base
        DIRETORIO_MODELO_LINGUAGEM = verificaDiretorioModeloLinguagem()
                      
        # Nome arquivo spacy
        ARQUIVO_MODELO_SPACY = self.model_args.modelo_spacy
        # Versão spaCy
        VERSAO_SPACY = "-" + self.model_args.versao_spacy
        # Caminho raiz do modelo do spaCy
        DIRETORIO_MODELO_SPACY =  DIRETORIO_MODELO_LINGUAGEM + "/" + ARQUIVO_MODELO_SPACY + VERSAO_SPACY

        # Verifica se o diretório existe
        if os.path.exists(DIRETORIO_MODELO_SPACY) == False:
            # Realiza o download do arquivo modelo do spaCy
            self.downloadSpacy()
            # Descompacta o spaCy
            self.descompactaSpacy()

        # Diretório completo do spaCy
        DIRETORIO_MODELO_SPACY = DIRETORIO_MODELO_LINGUAGEM + "/" + ARQUIVO_MODELO_SPACY + VERSAO_SPACY + '/' + ARQUIVO_MODELO_SPACY + '/' + ARQUIVO_MODELO_SPACY + VERSAO_SPACY + '/'

        # Carrega o spaCy. Necessário somente 'tagger' para encontrar os substantivos
        self.nlp = spacy.load(DIRETORIO_MODELO_SPACY, disable=['tokenizer', 'lemmatizer', 'ner', 'parser', 'textcat', 'custom'])
        
        logging.info("spaCy carregado.")1

    def get_nlp(self):
        '''
        Recupera o modelo.
        '''
        return self.nlp
