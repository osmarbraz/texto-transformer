# Import das bibliotecas.
import logging  # Biblioteca de logging
import tarfile # Biblioteca de descompactação
import os # Biblioteca de manipulação de arquivos
import spacy # Biblioteca do spaCy

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
        
        self.carrega();
            
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
            print("Carregando o spaCy")
            self.nlp = spacy.load(self.modelo_args.modelo_spacy,disable=['tokenizer', 'lemmatizer', 'ner', 'parser', 'textcat', 'custom'])    
            print("spaCy carregado.")
        
        else:
            print("spaCy não carregado!")
            self.nlp = None
                

    def get_nlp(self):
        '''
        Recupera o modelo de NLP.
        '''
        return self.nlp
