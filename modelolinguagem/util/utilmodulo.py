# Import das bibliotecas.
import logging  # Biblioteca de logging
import re # Biblioteca de expressão regular
import unicodedata # Biblioteca de codificação de caracteres

# ============================  
def removeAcentos(texto):   
    '''    
    Remove acentos de um texto.
    
    Parâmetros:
   `texto` - Texto a ser removido os acentos.
    '''
    
    try:
        text = unicode(texto, 'utf-8')
    except (TypeError, NameError): 
        pass
    
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore')
    texto = texto.decode("utf-8")
    
    return str(texto)

# ============================  
def limpaTexto(texto):    
    '''    
    Remove acentos e espaços e outros caracteres de um texto.
    
    Parâmetros:
   `texto` - Texto a ser limpo.
    '''
    
    texto = removeAcentos(texto.lower())
    texto = re.sub('[ ]+', '_', texto)
    texto = re.sub('[^.0-9a-zA-Z_-]', '', texto)
    
    return texto
    
# ============================  
def remove_tags(texto):
    '''
    Remove tags de um texto.
     
    Parâmetros:
    `texto` - Texto com tags a serem removidas.      
    '''
     
    textoLimpo = re.compile('<.*?>')
     
    return re.sub(textoLimpo, '', texto)
