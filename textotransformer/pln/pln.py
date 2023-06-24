# Import das bibliotecas.

# Biblioteca de logging
import logging  
# Biblioteca de manipulação de arquivos
import os 
# Biblioteca do spaCy
import spacy 
from spacy.util import filter_spans
from spacy.matcher import Matcher

# Bibliotecas próprias
from textotransformer.util.utilambiente import InstaladorModelo 

logger = logging.getLogger(__name__)

class PLN():

    ''' 
    Realiza o processamento de linguagem natural.
     
    Parâmetros:
       `modelo_args` - Parâmetros da classe PLN a ser carregado.    
    ''' 

    # Construtor da classe
    def __init__(self, modelo_args):        
    
        # Parâmetros do modelo
        self.modelo_args = modelo_args
        
        #Instala a ferramenta pln spaCy
        InstaladorModelo(model_args=modelo_args)
        
        #Carrega o modelo do spacy
        self.carrega()
            
        logger.info("Classe \"{}\" carregada: \"{}\".".format(self.__class__.__name__, modelo_args))
            
    # ============================   
    def __repr__(self):
        '''
        Retorna uma string com descrição do objeto.
        '''

        return "Classe (\"{}\") carregada com o modelo: \"{}\"".format(self.__class__.__name__,
                                                                       self.modelo_args.modelo_spacy)         
    
    # ============================    
    def carrega(self):
        '''
        Realiza o carregamento da ferramenta de PLN.     
        '''
       
        # Carrega o modelo spacy            
        self.model_pln = spacy.load(self.modelo_args.modelo_spacy)    
        # Opção para remover funcionalidades
        #self.model_pln = spacy.load(self.modelo_args.modelo_spacy,disable=['tokenizer', 'lemmatizer', 'ner', 'parser', 'textcat', 'custom'])                
        
        logger.info("Modelo spaCy versão \"{}\" carregado!".format(self.modelo_args.modelo_spacy))
    
    # ============================
    
    def getPOSTaggingUniversalTraduzido(postagging):
        '''
        Retorna a tradução das POS-Tagging.
        
        Tags de palavras universal https://universaldependencies.org/u/pos/

        Detalhes das tags em português: http://www.dbd.puc-rio.br/pergamum/tesesabertas/1412298_2016_completo.pdf

        Parâmetros:
           `postagging` - Uma tag de uma palavra.

        Retorno:
           Uma string com a tradução da tag.        
        '''
    
        #dicionário que contêm pos tag universal e suas explicações
        postagging_universal_dict = {
          "X"    : "Outro",
          "VERB" : "Verbo ",
          "SYM"  : "Símbolo",
          "CONJ" : "Conjunção",
          "SCONJ": "Conjunção subordinativa",
          "PUNCT": "Pontuação",
          "PROPN": "Nome próprio",
          "PRON" : "Pronome substativo",
          "PART" : "Partícula, morfemas livres",
          "NUM"  : "Numeral",
          "NOUN" : "Substantivo",
          "INTJ" : "Interjeição",
          "DET"  : "Determinante, Artigo e pronomes adjetivos",
          "CCONJ": "Conjunção coordenativa",
          "AUX"  : "Verbo auxiliar",
          "ADV"  : "Advérbio",
          "ADP"  : "Preposição",
          "ADJ"  : "Adjetivo"
        }
        
        if postagging in postagging_universal_dict.keys():
            traduzido = postagging_universal_dict[postagging]
        else:
            traduzido = "NA" 
        return traduzido
    
    # ============================
    def getStopwords(self):
        '''
        Recupera as stop words do model_pln(Spacy).

        Retorno:
           Uma lista com as stopwords.                
        '''
        
        spacy_stopwords = self.model_pln.Defaults.stop_words
        
        return spacy_stopwords 

    # ============================
    def removeStopWord(self, texto):
        '''
        Remove as stopwords de um texto.
        
        Parâmetros:
           `texto` - Um texto com stopwords.

        Retorno:
           Uma string com o texto sem as stopwords.
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
        doc = self.model_pln(texto)

        # Retorna a lista das palavras relevantes de um tipo
        texto_com_palavras_relevantes = [token.text for token in doc if token.pos_ == tipo_palavra_relevante]

        # Concatena o texto com as palavras relevantes
        texto_com_palavras_relevantes_concatenado = ' '.join(texto_com_palavras_relevantes)

        # Retorna o texto
        return texto_com_palavras_relevantes_concatenado       
       
    # ============================
    def getListaSentencasTexto(self, texto):
        '''
        Retorna uma lista com as sentenças de um texto. Utiliza o spacy para dividir o texto em sentenças.
        
        Parâmetros:
           `texto` - Um texto a ser convertido em uma lista de sentenças.

        Retorno:
           `lista` - Lista com as sentenças do texto.
        '''

        # Verifica se o texto não foi processado pelo spaCy  
        if type(texto) is not spacy.tokens.doc.Doc:
            # Realiza o parsing no spacy
            doc = self.model_pln(texto)
        else:
            doc = texto

        # Lista para as sentenças
        lista = []
        # Percorre as sentenças
        for sentenca in doc.sents: 
            # Adiciona as sentenças a lista
            lista.append(str(sentenca))

        return lista       

    # ============================
    def getListaSentencasTokensTexto(self, texto):
        '''
        Retorna duas listas, uma com as sentenças de um texto e outra com a lista de lista de tokens de cada sentença.
        
        Parâmetros:
           `texto` - Um texto a ser processado em uma lista de sentenças e tokens.           

        Retorno:
           `lista_sentencas` - Lista com as sentenças do texto.
           `lista_tokens` - Lista com os tokens de cada sentença do texto.                 
        '''

        # Retorna uma lista com as sentenlas do texto        
        lista_sentencas = self.getListaSentencasTexto(texto)

        # Lista para os tokens
        lista_tokens = []
        # Percorre as sentenças
        for sentenca in lista_sentencas:
            # Adiciona os tokens a lista
            lista_tokens.append(self.getListaTokensSentenca(sentenca))
        
        # @TODO: Verificar se é necessário retornar a lista de tokens. Converter para um dicionário?

        return lista_sentencas, lista_tokens

    # ============================
    def getVerbosTexto(self, texto):
        '''
        Retorna uma lista com os verbos de um texto.

        Parâmetros:
           `texto` - Um texto a ser processado em uma lista de sentenças e tokens.
        
        Retorno:
           `lista_verbos` - Lista com os verbos do texto.
        '''
        
        # (verbo normal como auxilar ou auxilar) + vários verbos auxiliares +verbo principal ou verbo auxiliar
        gramaticav1 =  [
                        {"POS": "AUX", "OP": "?", "DEP": {"IN": ["aux","aux:pass"]}},  #verbo auxiliar                                  
                        {"POS": "VERB", "OP": "?", "DEP": {"IN": ["ROOT","aux","xcomp","aux:pass"]}},  #verbo normal como auxiliar
                        {"POS": "AUX", "OP": "*", "DEP": {"IN": ["aux","xcomp","aux:pass"]}},  #verbo auxiliar   
                        {"POS": "VERB", "OP": "+"}, #verbo principal
                        {"POS": "AUX", "OP": "?", "DEP": {"IN": ["cop","aux","xcomp","aux:pass"]}},  #verbo auxiliar
                       ] 

        # verbo auxiliar + verbo normal como auxiliar + conjunção com preposição + verbo
        gramaticav2 =  [               
                        {"POS": "AUX", "OP": "?", "DEP": {"IN": ["aux","aux:pass"]}},  #verbo auxiliar                   
                        {"POS": "VERB", "OP": "+", "DEP": {"IN": ["ROOT"]}},  #verbo principal       
                        {"POS": "SCONJ", "OP": "+", "DEP": {"IN": ["mark"]}}, #conjunção com preposição
                        {"POS": "VERB", "OP": "+", "DEP": {"IN": ["xcomp"]}}, #verbo normal como complementar
                       ] 

        #Somente verbos auxiliares
        gramaticav3 =  [
                        {"POS": "AUX", "OP": "?"},  #Verbos auxiliar 
                        {"POS": "AUX", "OP": "?", "DEP": {"IN": ["cop"]}},  #Verbos auxiliar de ligação (AUX+(cop))
                        {"POS": "ADJ", "OP": "+", "DEP": {"IN": ["ROOT"]}}, 
                        {"POS": "AUX", "OP": "?"}  #Verbos auxiliar 
                       ] 

        matcherv = Matcher(self.model_pln.vocab)
                 
        matcherv.add("frase verbal", [gramaticav1])
        matcherv.add("frase verbal", [gramaticav2])
        matcherv.add("frase verbal", [gramaticav3])

        #Processa o período        
        if isinstance(texto, str):            
            doc1 = self.model_pln(texto)
        else:
            #Processa o período
            doc1 = self.model_pln(texto.text)
          
        # Chama o mather para encontrar o padrão
        matches = matcherv(doc1)

        padrao = [doc1[start:end] for _, start, end in matches]

        #elimina as repetições e sobreposições
        #return filter_spans(padrao)
        lista1 = filter_spans(padrao)

        # Converte os itens em string
        lista2 = []
        for x in lista1:
            lista2.append(str(x))
          
        return lista2        

    # ============================
    def getDicPOSQtde(self, texto):
        '''
        Retorna a quantidade de cada POS Tagging existentes em um texto.
        
        Parâmetros:
           `texto` - Um texto a ser convertido em uma lista de sentenças.           

        Retorno:
           Um dicionário com as POS-Tagging e suas quantidades.                 
        '''
        
        # Verifica se o texto não foi processado pelo spaCy  
        if type(texto) is not spacy.tokens.doc.Doc:
            # Realiza o parsing no spacy
            doc = self.model_pln(texto)
        else:
            doc = texto

        # Retorna inteiros que mapeiam para classes gramaticais
        conta_dicionarios = doc.count_by(spacy.attrs.IDS["POS"])

        # Dicionário com as tags e quantidades
        novodic = dict()
          
        for pos, qtde in conta_dicionarios.items():
            classe_gramatical = doc.vocab[pos].text
            novodic[classe_gramatical] = qtde

        return novodic

    # ============================
    def getDicTodasPOSQtde(self, texto):
        '''
        Retorna a quantidade de cada POS Tagging em um texto. Considera um conjunto fixo de POS-Tagging.
        
        Parâmetros:
           `texto` - Um texto a ser convertido em uma lista de sentenças.           

        Retorno:
           Um dicionário com as classes POS-Tagging e suas quantidades.                 
        '''

        # Verifica se o texto não foi processado pelo spaCy  
        if type(texto) is not spacy.tokens.doc.Doc:
            # Realiza o parsing no spacy
            doc = self.model_pln(texto)
        else:
            doc = texto

        # Retorna inteiros que mapeiam para classes gramaticais
        conta_dicionarios = doc.count_by(spacy.attrs.IDS["POS"])

        # Dicionário com as tags e quantidades    
        novodic = {"PRON":0, "VERB":0, "PUNCT":0, "DET":0, "NOUN":0, "AUX":0, "CCONJ":0, "ADP":0, "PROPN":0, "ADJ":0, "ADV":0, "NUM":0, "SCONJ":0, "SYM":0, "SPACE":0, "INTJ":0, "X": 0}
        
        for pos, qtde in conta_dicionarios.items():
            classe_gramatical = doc.vocab[pos].text
            novodic[classe_gramatical] = qtde

        return novodic

    # ============================
    def getDicTodasPOSListaQtde(self, lista):
        '''
        Retorna a quantidade de cada POS Tagging de uma lista. Considera um conjunto fixo de POS-Tagging.
        
        Parâmetros:
           `lista` - Uma lista com as POS-Tagging.

        Retorno:
           Um dicionário com as POS-Tagging e suas quantidades.
        '''

        # Dicionário com as tags e quantidades
        conjunto = {"PRON":0, "VERB":0, "PUNCT":0, "DET":0, "NOUN":0, "AUX":0, "CCONJ":0, "ADP":0, "PROPN":0, "ADJ":0, "ADV":0, "NUM":0, "SCONJ":0, "SYM":0, "SPACE":0, "INTJ": 0}

        for x in lista:
            valor = conjunto.get(x)
            if valor != None:
                conjunto[x] = valor + 1
            else:
                conjunto[x] = 1

        return conjunto

    # ============================
    def getTokensTexto(self, texto):
        '''
        Retorna a lista de tokens do texto.
        
        Parâmetros:
           `texto` - Um texto a ser recuperado os tokens.

        Retorno:
           Uma lista com os tokens do texto.                 
        '''

        # Verifica se o texto não foi processado pelo spaCy  
        if type(texto) is not spacy.tokens.doc.Doc:
            # Realiza o parsing no spacy
            doc = self.model_pln(texto)
        else:
            doc = texto

        # Lista dos tokens
        lista = []

        # Percorre a sentença adicionando os tokens
        for token in doc:    
            lista.append(token.text)

        return lista
    
    # ============================
    def getPOSTokensTexto(self, texto):
        '''
        Retorna a lista das POS-Tagging dos tokens do texto.
        
        Parâmetros:
           `texto` - Um texto a ser recuperado as POS-Tagging.
                 
        Retorno:
           Uma lista com as POS-Tagging dos tokens do texto.
        '''

        # Verifica se o texto não foi processado pelo spaCy  
        if type(texto) is not spacy.tokens.doc.Doc:
            # Realiza o parsing no spacy
            doc = self.model_pln(texto)
        else:
            doc = texto

        # Lista dos tokens
        lista = []

        # Percorre a sentença adicionando os tokens
        for token in doc:    
            lista.append(token.pos_)

        return lista

    # ============================
    def getListaTokensPOSTexto(self, texto):
        '''
        Retorna duas listas uma com os tokens e a outra com a POS-Tagging dos tokens do texto.
        
        Parâmetros:
           `texto` - Um texto a ser recuperado as listas de tokens e POS-Tagging.

        Retorno:
           Duas listas com os tokens e a POS-Tagging do texto.                 
        '''

        # Verifica se o texto não foi processado pelo spaCy  
        if type(texto) is not spacy.tokens.doc.Doc:
            # Realiza o parsing no spacy
            doc = self.model_pln(texto)
        else:
            doc = texto

        # Lista dos tokens
        lista_tokens = []
        lista_pos = []

        # Percorre a sentença adicionando os tokens e as POS
        for token in doc:    
            lista_tokens.append(token.text)
            lista_pos.append(token.pos_)
        
        return lista_tokens, lista_pos

    # ============================   
    def getTextoSemStopWord(self, listaTokens):
        '''
        Retorna uma lista com tokens de um texto excluindo as stopwords.
        
        Parâmetros:
           `listaTokens` - Uma lista com os tokens de um texto.    

        Retorno:
           Uma lista com os tokens sem as stopwords.             
        '''

        # Recupera as stopwords do modelo
        stopwords = self.getStopwords()
        
        # Lista dos tokens
        lista = []

        # Percorre os tokens do texto
        for i, token in enumerate(listaTokens):
    
            # Verifica se o token é uma stopword
            if token.lower() not in stopwords:
                lista.append(token)

        # Retorna a lista de tokens sem as stopwords
        return lista

    # ============================   
    def getModelPln(self):
        '''
        Recupera o modelo de PLN.
        '''

        return self.model_pln
    