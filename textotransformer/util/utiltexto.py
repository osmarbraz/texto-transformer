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
from typing import List, Union

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
def removeTags(texto: str):
    '''
    Remove tags de um texto.
     
    Parâmetros:
      `texto` - Texto com tags a serem removidas.
    
    Retorno:
       O texto sem as tags.      
    '''
     
    texto_limpo = re.compile('<.*?>')
     
    return re.sub(texto_limpo, '', texto)

# ============================
def encontrarIndiceSubLista(lista: List, sublista: List):
    '''
    Localiza os índices de início e fim de uma sublista em uma lista.
    
    Parâmetros:
      `lista` - Uma lista.
      `sublista` - Uma sublista a ser localizada na lista.
    
    Retorno:
       Os índices de início e fim da sublista na lista.
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
            indice_inicio = i - n + 1
            indice_fim = indice_inicio + len(sublista)-1
            return indice_inicio, indice_fim
        
    return -1, -1
    
# ============================
def getTextoLista(lista_sentencas: List):
    '''
    Recebe uma lista de sentenças e faz a concatenação em uma string.
    
    Parâmetros:
      `listaTexto` - Uma lista contendo diversas sentenças.           
    '''

    string_texto = ''  
    # Concatena as sentenças do texto
    for sentenca in lista_sentencas:                
        string_texto = string_texto + sentenca
        
    return string_texto

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
    
    Retorno:
       `novodic` - Um dicionário com a soma dos valores dos dicionários da lista.       
    '''
    
    # Soma os dicionários da lista
    novodic = reduce(atualizaValor, (Counter(dict(x)) for x in lista))
 
    return novodic

# ============================   
def limpezaTexto(texto: str):
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

    # Conta texto com duas ou mais interrogação
    conta_caracter_interrogacoes = texto.count("?")
    if conta_caracter_interrogacoes > 1:
        # Nessa expressão, o \? representa uma interrogação, e o + indica que devemos buscar um ou mais interrogações consecutivos.
        texto = re.sub("\?+", "?", texto)
        
    # Conta caracteres em branco repetidos
    conta_caracter_espacos = texto.count("  ")
    if conta_caracter_espacos > 0:        
        # Nessa expressão, o \s representa um espaço em branco, e o + indica que devemos buscar um ou mais espaços em branco consecutivos.
        texto = re.sub("\s+", " ", texto)
  
    # Transforma em string e remove os espaços do início e do fim
    texto = str(texto).strip()
        
    return texto

# ============================    
def tamanhoTexto(texto: Union[List[int], List[List[int]]]):
    '''                
    Função de ajuda para obter o comprimento do texto. O texto pode ser uma lista de ints (o que significa um único texto como entrada) ou uma tupla de lista de ints (representando várias entradas de texto para o modelo).

    Parâmetros:
        `texto` - Um texto a ser recuperado o tamanho.

    Retorno:
        Um inteiro com tamanho do texto.
    '''

    # Se for um dic de listas
    if isinstance(texto, dict):              
        return len(next(iter(texto.values())))
    else:
        # Se o objeto não tem len()
        if not hasattr(texto, '__len__'):      
            return 1
        else:
            # Se string vazia ou lista de ints
            if len(texto) == 0 or isinstance(texto[0], int):  
                return len(texto)
            else:
                # Soma do comprimento de strings individuais
                return sum([len(t) for t in texto])      