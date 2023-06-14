# Import das bibliotecas.
import logging  # Biblioteca de logging
import os # Biblioteca de manipulação de arquivos
import spacy # Biblioteca do spaCy

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *
from medidor.medidorenum import *

from util.utilambiente import *

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
        
        #Instala o spaCy
        InstaladorModelo(modelo_args=modelo_args)
        
        #Carrega o modelo do spacy
        self.carrega();
            
        logging.info("NLP carregado: {}.".format(modelo_args))    
            
    # ============================
    def getStopwords(self):
        '''
        Recupera as stop words do model_nlp(Spacy).
        
        Parâmetros:
        `nlp` - Um modelo spaCy carregado.           
        '''
        
        spacy_stopwords = self.model_nlp.Defaults.stop_words

        return spacy_stopwords 
       
    # ============================    
    def carrega(self):
        '''
        Realiza o carregamento da ferramenta de NLP.
     
        '''
          
        # Verifica se é necessário carregar a ferramenta
        if self.modelo_args.palavra_relevante != PalavrasRelevantes.ALL.value:
            # Carrega o modelo spacy            
            self.model_nlp = spacy.load(self.modelo_args.modelo_spacy)                
            #self.model_nlp = spacy.load(self.modelo_args.modelo_spacy,disable=['tokenizer', 'lemmatizer', 'ner', 'parser', 'textcat', 'custom'])                
            logging.info("Modelo spaCy versão {} carregado!".format(self.modelo_args.modelo_spacy))    
        
        else:
            logging.info("Modelo spaCy versão {} não carregado!".format(self.modelo_args.modelo_spacy)) 
            self.nlp = None
                

    def get_model_nlp(self):
        '''
        Recupera o modelo de NLP.
        '''
        return self.model_nlp
        
   
          
         