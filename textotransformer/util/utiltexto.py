# Import das bibliotecas.

# Biblioteca de logging
import logging  
# Biblioteca de expressão regular
import re 
# Biblioteca de codificação de caracteres
import unicodedata 
from collections import Counter
from functools import reduce
# Biblioteca de tipos
from typing import List

logger = logging.getLogger(__name__)

# ============================  
def convertTextoUtf8(texto: str):   
    '''    
    Converte um texto para utf-8.
    
    Parâmetros:
   `texto` - Texto a ser convertido para utf-8.

    Retorno:
    Texto convertido para utf-8.
    '''
    
    try:
        texto = texto.encode('utf-8')
    except (TypeError, NameError): 
        pass

    return texto

# ============================  
def removeAcentos(texto: str):   
    '''    
    Remove acentos de um texto.
    
    Parâmetros:
   `texto` - Texto a ser removido os acentos.

    Retorno:
    Texto sem acentos.
    '''
    
    texto = convertTextoUtf8(texto)    
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore')
    texto = texto.decode("utf-8")
    
    return str(texto)

# ============================  
def limpaTexto(texto: str):    
    '''    
    Remove acentos e espaços e outros caracteres de um texto.
    
    Parâmetros:
   `texto` - Texto a ser limpo.
    '''
    # Converte para minúsculo
    texto = removeAcentos(texto.lower())
    # Remove espaços em branco
    texto = re.sub('[ ]+', '_', texto)
    # Remove caracteres especiais
    texto = re.sub('[^.0-9a-zA-Z_-]', '', texto)
    
    return texto
    
# ============================  
def remove_tags(texto: str):
    '''
    Remove tags de um texto.
     
    Parâmetros:
    `texto` - Texto com tags a serem removidas.      
    '''
     
    textoLimpo = re.compile('<.*?>')
     
    return re.sub(textoLimpo, '', texto)

# ============================
def encontrarIndiceSubLista(lista: List, sublista: List):
    '''
    Localiza os índices de início e fim de uma sublista em uma lista.
    
    Parâmetros:
    `lista` - Uma lista.
    `sublista` - Uma sublista a ser localizada na lista.
    '''
    
    # https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore%E2%80%93Horspool_algorithm
    h = len(lista)
    n = len(sublista)
    skip = {sublista[i]: n - i - 1 for i in range(n - 1)}
    i = n - 1
    while i < h:
        for j in range(n):
            if lista[i - j] != sublista[-j - 1]:
                i += skip.get(lista[i], n)
                break
        else:
            indiceInicio = i - n + 1
            indiceFim = indiceInicio + len(sublista)-1
            return indiceInicio, indiceFim
        
    return -1, -1
    
# ============================
def getTextoLista(listaSentencas: List):
    '''
    Recebe uma lista de sentenças e faz a concatenação em uma string.
    
    Parâmetros:
    `listaTexto` - Uma lista contendo diversas sentenças.           
    '''

    stringTexto = ''  
    # Concatena as sentenças do texto
    for sentenca in listaSentencas:                
        stringTexto = stringTexto + sentenca

# ============================   
def atualizaValor(a,b):
    a.update(b)
    return a

# ============================   
def getSomaDic(lista: List):
    '''
    Soma os valores de dicionários com as mesmas chaves.
    
    Parâmetros:
    `lista` - Uma lista contendo dicionários.           
    '''
    
    # Soma os dicionários da lista
    novodic = reduce(atualizaValor, (Counter(dict(x)) for x in lista))
 
    return novodic

def limpeza(texto: str):
    '''
    Realiza limpeza dos dados.
        
    Parâmetros:
    `texto` - Um texto a ser limpo.      

    Retorno:
    `texto` - Texto limpo.  
    '''
    # Substitui \n por espaço em branco no documento
    conta_caracter_barra_n = texto.count("\n")
    if conta_caracter_barra_n > 0:
        # Transforma \n em espaços em branco 
        texto = texto.replace("\n"," ")

    # Transforma em string e remove os espaços do início e do fim
    texto = str(texto).strip()

    # Conta texto com duas ou mais interrogação
    conta_caracter_interrogacoes = texto.count("?")
    if conta_caracter_interrogacoes > 1:
        # Transforma 2 ou mais interrogações consecutivas em 1
        texto = re.sub("\?+", "?", texto)
        
    # Conta caracteres em branco repetidos
    conta_caracter_espacos = texto.count("  ")
    if conta_caracter_espacos > 0:
        # Transforma 2 ou mais caracteres em branco consecutivos em 1    
        texto = re.sub("\W*\?+\W*", "? ", texto)
        
    return texto