# Import das bibliotecas.
import logging  # Biblioteca de logging
import os # Biblioteca de manipulação de arquivos
import spacy # Biblioteca do spaCy
import subprocess

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *
from medidor.medidorenum import *

class NLP():

    ''' 
    Realiza o processamento de linguagem natural.
     
    Parâmetros:
    `nome_pln` - Nome da ferramenta de PLN a ser carregada.    
    ''' 

    # Construtor da classe
    def __init__(self, modelo_args):
    
        # Parâmetros do modelo
        self.modelo_args = modelo_args
        
        self.installspacy()
        
        self.carrega();
            
        logging.info("NLP carregado: {}.".format(modelo_args))    
            
    # ============================
    def getStopwords(self):
        '''
        Recupera as stop words do nlp(Spacy).
        
        Parâmetros:
        `nlp` - Um modelo spaCy carregado.           
        '''
        
        spacy_stopwords = self.nlp.Defaults.stop_words

        return spacy_stopwords 
       
    # ============================    
    def carrega(self):
        '''
        Realiza o carregamento da ferramenta de NLP.
     
        '''
          
        # Verifica se é necessário carregar a ferramenta
        if self.modelo_args.palavra_relevante != PalavrasRelevantes.ALL.value:
            # Carrega o modelo spacy
            logging.info("Carregando o spaCy")
            self.nlp = spacy.load(self.modelo_args.modelo_spacy,disable=['tokenizer', 'lemmatizer', 'ner', 'parser', 'textcat', 'custom'])    
            logging.info("spaCy carregado.")
        
        else:
            logging.info("spaCy não carregado!")
            self.nlp = None
                

    def get_nlp(self):
        '''
        Recupera o modelo de NLP.
        '''
        return self.nlp
        
    def installspacy(self):
        self.install_setuptools()
        self.install_spacy()
        self.install_model()

    def install_setuptools(self):
        try:
            subprocess.check_call(["pip", "-U", "install", "pip","setuptools", "wheel"])
            logging.info("setuptools instalado!")    
        except subprocess.CalledProcessError as e:
            logging.info("Falha em instalar setuptools. Erro: {}.".format(e))


    def install_spacy(self):
        try:
            subprocess.check_call(["pip", "-U", "install", "spacy={self.modelo_args.versao_spacy}"])
            logging.info("spaCy versão {} instalado!".format(self.modelo_args.versao_spacy))    
        except subprocess.CalledProcessError as e:
            logging.info("Falha em instalar spaCy versão {}. Erro: {}.".format(self.modelo_args.versao_spacy, e))    
    def install_model(self):
        try:
             # Download do modelo de linguagem na linguagem solicitada
            subprocess.check_call(["python", "-m", "spacy", "download", self.modelo_args.modelo_spacy])
            logging.info("Modelo spaCy {} instalado!".format(self.modelo_args.modelo_spacy))    
        except subprocess.CalledProcessError as e:
            logging.info("Falha em instalar modelo spaCy {}. Erro: {}.".format(self.modelo_args.modelo_spacy, e))

          
         