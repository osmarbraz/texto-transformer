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


# ============================
def encontrarIndiceSubLista(lista, sublista):
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
def getTextoLista(listaSentencas):
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
def getListaSentencasTexto(texto, nlp):
    '''
    Retorna uma lista com as sentenças de um texto. Utiliza o spacy para dividir o texto em sentenças.
    
    Parâmetros:
    `texto` - Um texto a ser convertido em uma lista de sentenças.           
             
    '''

    # Aplica sentenciação do spacy no texto
    doc = nlp(texto) 

    # Lista para as sentenças
    lista = []
    # Percorre as sentenças
    for sentenca in doc.sents: 
        # Adiciona as sentenças a lista
        lista.append(str(sentenca))

    return lista
    
# ============================
def removeStopWord(texto, stopwords):
    '''
    Remove as stopwords de um texto.
    
    Parâmetros:
    `texto` - Um texto com stopwords.
    `stopwords` - Uma lista com as stopwords.
    '''

    # Remoção das stop words do texto
    textoSemStopwords = [palavra for palavra in texto.split() if palavra.lower() not in stopwords]

    # Concatena o texto sem os stopwords
    textoLimpo = ' '.join(textoSemStopwords)

    # Retorna o texto
    return textoLimpo
    
# ============================
def retornaPalavraRelevante(texto, nlp, tipo_palavra_relevante='NOUN'):
    '''
    Retorna somente os palavras do texto ou sentença do tipo especificado.
    
    Parâmetros:
    `texto` - Um texto com todas as palavras.
    `nlp` - Processador de linguagem natural.
    `tipo_palavra_relevante` - Tipo de palavra relevante a ser selecionada.
    
    Retorno:
    `textoComRelevantesConcatenado` - Texto somente com as palavras relevantes.
    '''
  
    # Realiza o parsing no texto usando spacy
    doc = nlp(texto)

    # Retorna a lista das palavras relevantes de um tipo
    textoComRelevantes = [token.text for token in doc if token.pos_ == tipo_palavra_relevante]

    # Concatena o texto com as palavras relevantes
    textoComRelevantesConcatenado = ' '.join(textoComRelevantes)

    # Retorna o texto
    return textoComRelevantesConcatenado