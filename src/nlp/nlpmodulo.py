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
        self.carrega()
            
        logging.info("Classe NLP carregada: {}.".format(modelo_args))    
            
    # ============================    
    def carrega(self):
        '''
        Realiza o carregamento da ferramenta de NLP.
     
        '''
       
        # Carrega o modelo spacy            
        self.model_nlp = spacy.load(self.modelo_args.modelo_spacy)                
        #self.model_nlp = spacy.load(self.modelo_args.modelo_spacy,disable=['tokenizer', 'lemmatizer', 'ner', 'parser', 'textcat', 'custom'])                
        logging.info("Modelo spaCy versão {} carregado!".format(self.modelo_args.modelo_spacy))    
    
    # ============================
    def getStopwords(self):
        '''
        Recupera as stop words do model_nlp(Spacy).
        
        Parâmetros:
        `nlp` - Um modelo spaCy carregado.           
        '''
        
        spacy_stopwords = self.model_nlp.Defaults.stop_words
        
        logging.info("Carregando as stopwords do modelo {}.".format(self.modelo_args.modelo_spacy))    

        return spacy_stopwords 

    # ============================
    def removeStopWord(self, texto):
        '''
        Remove as stopwords de um texto.
        
        Parâmetros:
        `texto` - Um texto com stopwords.
        '''

        # Recupera as stopwords
        stopwords = self.getStopwords()

        # Remoção das stop words do texto
        textoSemStopwords = [palavra for palavra in texto.split() if palavra.lower() not in stopwords]

        # Concatena o texto sem os stopwords
        textoLimpo = ' '.join(textoSemStopwords)

        # Retorna o texto
        return textoLimpo
       
    # ============================
    def retornaPalavraRelevante(self, texto, tipo_palavra_relevante='NOUN'):
        '''
        Retorna somente os palavras do texto ou sentença do tipo especificado.
        
        Parâmetros:
        `texto` - Um texto com todas as palavras.        
        `tipo_palavra_relevante` - Tipo de palavra relevante a ser selecionada.
        
        Retorno:
        `textoComRelevantesConcatenado` - Texto somente com as palavras relevantes.
        '''
      
        # Realiza o parsing no texto usando spacy
        doc = self.model_nlp(texto)

        # Retorna a lista das palavras relevantes de um tipo
        textoComRelevantes = [token.text for token in doc if token.pos_ == tipo_palavra_relevante]

        # Concatena o texto com as palavras relevantes
        textoComRelevantesConcatenado = ' '.join(textoComRelevantes)

        # Retorna o texto
        return textoComRelevantesConcatenado       
       
    # ============================
    def getListaSentencasTexto(self, texto):
        '''
        Retorna uma lista com as sentenças de um texto. Utiliza o spacy para dividir o texto em sentenças.
        
        Parâmetros:
        `texto` - Um texto a ser convertido em uma lista de sentenças.           
                 
        '''

        # Aplica sentenciação do spacy no texto
        doc = self.model_nlp(texto) 

        # Lista para as sentenças
        lista = []
        # Percorre as sentenças
        for sentenca in doc.sents: 
            # Adiciona as sentenças a lista
            lista.append(str(sentenca))

        return lista       
       
    def get_model_nlp(self):
        '''
        Recupera o modelo de NLP.
        '''
        return self.model_nlp
        
   
          
         