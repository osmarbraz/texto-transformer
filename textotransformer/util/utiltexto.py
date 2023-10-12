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

# Objeto de logger
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
    # Não encontrou a sublista na lista   
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

# ============================
def getIndexTokenTexto(lista_tokens: List[str], 
                       token : str) -> List[int]:
    '''
    Recupera os indíces do token especificado token na lista de tokens.
    
    Parâmetros:
       `lista_tokens` - Uma lista de string(token).
       `token` - Um token a ser procurado na lista.
    
    Retorno:
       `lista_index` - Uma lista as posições do token na lista.
    '''
    
    # Lista com os índices do token na lista
    lista_index = []

    # Percorre a lista de tokens
    for i, token_lista in enumerate(lista_tokens):
        # Se o token for igual ao token da lista
        if token_lista == token:
            # Adiciona o índice na lista
            lista_index.append(i)
    
    return lista_index

# ============================
def contaElemento(lista, elemento):
    ''' 
    Conta o número de ocorrências do elemento na lista.
        
    Parâmetros:
    `lista` - Lista com os elementos.
    `elemento` - Elemento a ser contado a ocorrência na lista.

    Retorno:    
    `cont` - Quantidade de ocorrências de elmento na lista.
    '''
    # Inicializa contador de ocorrências
    cont = 0
    # Percorre a lista
    for i, linha in enumerate(lista):      
      # Verifica se o elemento existe na lista
      if linha in elemento:
        # conta o elemento
        cont = cont + 1
        
    return cont

# ============================
def contaItensLista(lista):
  """
     Conta os itens das lista de listas.

     Parâmetros:
       `lista` - Uma lista de lista itens.

     Retorno:
       `qtde_itens` - Quantidade de itens das listas.
  """
  # Quantidade itens da lista
  qtde_itens = 0

  for sentenca in lista:
    qtde_itens = qtde_itens + len(sentenca)

  return qtde_itens

# ============================
def truncaJanela(lista, lista_janela, tamanho_janela, lista_indice_janela, indice_passo, maximo_itens, folha_janela):
    """
     Trunca as palavras da janela até o máximo da janela.

     Parâmetros:
       Parâmetros:
       `lista` - Uma lista com todos os itens.
       `lista_janela` - Um dataframe com os itens.
       `tamanho_janela` - Tamanho da janela a ser montada.
       `lista_indice_janela` - Lista com os índices das sentenças que forma a janela.
       `indice_passo` - Índice do passo que se deseja da janela.
       `maximo_itens` - Máximo de palavras na janela. Trunca das extremidades preservando a palavra central.

     Retorno:
       `lista_janela` - Janela truncada pelo máximo de itens.
    """
    # Quantidade de itens nas janelas antes
    qtde_itens1 = contaItensLista(lista_janela)
    # print("quantidade de itens janela antes:", qtde_itens1)

    # Controle se não alcançado o máximo de palavras
    minimo_alcancado = False

    # Remove as palavras das extremidade que ultrapassam o tamanho máximo
    while qtde_itens1 > maximo_itens and minimo_alcancado == False:

      # Verifica se é uma janela de início
      if indice_passo < folha_janela and len(lista_janela) != tamanho_janela:
        # print("Janelas do inicio")
        # Verifica se a última sentença tem elementos
        if len(lista_janela[-1]) >0:
          # Meio - Remove do fim da janela da última sentença
          del lista_janela[-1][-1]
        else:
          # Verifica se a penúltima sentença tem elementos
          if len(lista_janela[-2]) >0:
            # Meio - Remove do fim da janela da penúltima sentença
            del lista_janela[-2][-1]
          else:
            # Verifica se a antepenúltima sentença tem elementos
            if len(lista_janela[-3]) >0:
              # Meio - Remove do fim da janela da antepenúltima sentença
              del lista_janela[-3][-1]
      else:
        # Verifica se é uma janela do meio
        if indice_passo < len(lista)-folha_janela and len(lista_janela) == tamanho_janela:
          # print("Janelas do meio")
          # Verifica se a primeira sentença tem elementos
          if len(lista_janela[0]) >0:
            # Meio - Remove do inicio da janela da 1a sentença
            del lista_janela[0][0]
          else:
            # Verifica se a segunda sentença tem elementos
            if len(lista_janela[1]) >0:
              # Meio - Remove do inicio da janela da 2a sentença
              del lista_janela[1][0]

          # Verifica se a última sentença tem elementos
          if len(lista_janela[-1]) >0:
            # Meio - Remove do fim da janela da 1a sentença
            del lista_janela[-1][-1]
          else:
            # Verifica se a penúltima sentença tem elementos
            if len(lista_janela[-2]) >0:
              # Meio - Remove do fim da janela da 2a sentença
              del lista_janela[-2][-1]
        else:
            # Verifica se é uma janela de fim
            if indice_passo >= len(lista)-folha_janela and len(lista_janela) != tamanho_janela:
              # Verifica se a primeira sentença tem elementos
              if len(lista_janela[0]) >0:
                # Fim - Remove do início da janela da 1a sentença
                del lista_janela[0][0]
              else:
                # Verifica se a segunda sentença tem elementos
                if len(lista_janela[1]) >0:
                  # Fim - Remove do início da janela da 2a sentença
                  del lista_janela[1][0]
                else:
                  # Verifica se a terceira sentença tem elementos
                  if len(lista_janela[2]) >0:
                    # Fim - Remove do início da janela da 3a sentença
                    del lista_janela[2][0]

      # Calcula a nova quantidade de itens
      qtde_itens2 = contaItensLista(lista_janela)
      # print("quantidade de itens janela durante:", qtde_itens2)

      # Verifica se conseguiu reduzir a quantidade de itens
      if (qtde_itens1 == qtde_itens2):
        print("Atenção: Truncamento de janela não conseguiu reduzir além de ", qtde_itens2, " para o máximo ", maximo_itens)
        minimo_alcancado = True

      # Atribui uma nova quantidade
      qtde_itens1 = qtde_itens2

    return lista_janela

# ============================
def getJanelaSentenca(lista, tamanho_janela, indice_passo, maximo_itens=None):
  """
     Cria janelas de itens de uma lista

     Parâmetros:
       `lista` - Uma lista com todos os itens.
       `tamanho_janela` - Tamanho da janela a ser montada.
       `indice_passo` - Índice do passo que se deseja da janela.
       `maximo_itens` - Máximo de itens na janela. Trunca das extremidades preservando a palavra central.

     Retorno:
       `lista_janela` - Lista com os itens em janelas.
       `string_janela` - String com os itens em janelas.
       `lista_indice_janela` - Lista com os índices das sentenças que forma a janela.
       `lista_centro_janela` - Lista com os índices dos centros da janela.
  """

  # Se a lista é menor que o tamanho da janela  
  if len(lista) <= tamanho_janela:

    # Recupera a sentenças da janela
    lista_janela = []
    for i, sentenca in enumerate(lista):
        lista_janela.append(sentenca[0])

    # Guarda os índices dos itens das janelas
    lista_indice_janela = []

    # Adiciona os índices das janelas
    for i in range(len(lista)):
      lista_indice_janela.append(i)

    # Guarda os índices dos centros das janelas
    lista_centro_janela = []
    # Calcula o centro
    centro_janela = int((len(lista)/2))
    lista_centro_janela.append(centro_janela)

    # Concatena em uma string as palavras das sentenças da janela
    lista_janela_sentenca = []
    for sentenca in lista_janela:
      lista_janela_sentenca.append(" ".join(sentenca))
          
    return lista_janela, " ".join(lista_janela_sentenca), lista_indice_janela, lista_centro_janela

  else:
    # print(">>>> Lista maior que as janelas")
    # Lista maior que o tamanho da janela
    # Calcula o tamanho da folha da janela(quantidade de valores a esquerda e direitado centro da janela).
    folha_janela = int((tamanho_janela-1) /2)
    # print("folha_janela:", folha_janela)
    # print("indice_passo:", indice_passo)
    # Define o centro da janela
    centro_janela = -1
    # Percorre a lista
    # Dentro do intervalo da lista de itens
    if indice_passo >= 0 and indice_passo < len(lista):

      # Seleciona o passo que se deseja a janela de
      # Guarda os itens da janela
      lista_janela = []
      # Guarda os índices dos itens das janelas
      lista_indice_janela = []
      # Guarda os índices dos centros das janelas
      lista_centro_janela = []

      # Inicio da lista sem janelas completas antes do passo
      if indice_passo < folha_janela:
        # print("Inicio da lista")
        # Sentenças anteriores
        #Evita estourar o início da lista
        inicio = 0
        fim = indice_passo
        # print("Anterior: inicio:", inicio, " fim:", fim)
        for j in range(inicio, fim):
          # Recupera o documento da lista
          documento = lista[j]
          lista_janela.append(documento[0])
          # Adiciona o indice do documento na lista
          lista_indice_janela.append(j)

        # Sentença central
        # Recupera o documento da lista
        documento = lista[indice_passo]
        lista_janela.append(documento[0])
        # Adiciona o indice do documento na lista
        lista_indice_janela.append(indice_passo)
        # Guarda o centro da janela
        centro_janela = len(lista_janela)-1
        lista_centro_janela.append(centro_janela)

        # Sentenças posteriores
        inicio = indice_passo + 1
        fim = indice_passo + folha_janela + 1
        # print("Posterior: inicio:", inicio, " fim:", fim)
        for j in range(inicio,fim):
          # Recupera o documento da lista
          documento = lista[j]
          lista_janela.append(documento[0])
          # Adiciona o indice do documento na lista
          lista_indice_janela.append(j)

      else:
        # Meio da lista com janelas completas antes e depois
        if indice_passo < len(lista)-folha_janela:
          # print(" Meio da lista")

          # Sentenças anteriores
          inicio = indice_passo - folha_janela
          fim = indice_passo
          # print("inicio:", inicio, " fim:", fim)
          for j in range(inicio, fim):
            # Recupera o documento da lista
            documento = lista[j]
            # Adiciona o documento a janela
            lista_janela.append(documento[0])
            # Adiciona o indice do documento na lista
            lista_indice_janela.append(j)

          # Sentença central
          # Recupera o documento da lista
          documento = lista[indice_passo]
          # Adiciona o documento a janela
          lista_janela.append(documento[0])          
          # Adiciona o indice do documento na lista
          lista_indice_janela.append(indice_passo)
          # Guarda o centro da janela
          centro_janela = len(lista_janela)-1
          lista_centro_janela.append(centro_janela)

          # Sentenças posteriores
          inicio = indice_passo + 1
          fim = indice_passo + 1 + folha_janela
          for j in range(inicio,fim):
            # Recupera o documento da lista
            documento = lista[j]
            # Adiciona o documento a janela
            lista_janela.append(documento[0])
            # Adiciona o indice do documento na lista
            lista_indice_janela.append(j)

        else:
          # Fim da lista sem janelas completas depois
          if indice_passo >= len(lista)-folha_janela:
            # print("Fim da lista")

            # Sentenças anteriores
            inicio = indice_passo - folha_janela
            fim = indice_passo
            #print("inicio:", inicio, " fim:", fim)
            for j in range(inicio, fim):
              # Recupera o documento da lista
              documento = lista[j]
              # Adiciona o documento a janela
              lista_janela.append(documento[0])
              # Adiciona o indice do documento na lista
              lista_indice_janela.append(j)

            # Sentença central
            # Recupera o documento da lista
            documento = lista[indice_passo]
            # Adiciona o documento a janela
            lista_janela.append(documento[0])
            # Adiciona o indice do documento na lista
            lista_indice_janela.append(indice_passo)
            # Guarda o centro da janela
            centro_janela = len(lista_janela)-1
            lista_centro_janela.append(centro_janela)

            # Sentenças posteriores
            inicio = indice_passo + 1
            fim = indice_passo + 1 + folha_janela
            # Evita o extrapolar o limite da lista de sentenças
            if fim > len(lista):
              fim = len(lista)
            for j in range(inicio,fim):
              # Recupera o documento da lista
              documento = lista[j]
              # Adiciona o documento a janela
              lista_janela.append(documento[0])
              # Adiciona o indice do documento na lista
              lista_indice_janela.append(j)
    else:
      print("Índice fora do intervalo da lista de itens.")

    # Se existir maximo_itens realiza o truncamento
    if maximo_itens != None:
      # Cria uma copia da lista de itens para evitar a referência
      lista_apagar = []
      for item in lista_janela:
          lista_apagar.append(item.copy())

      # Trunca a quantidade de itens da janela até o máximo de itens.
      lista_janela = truncaJanela(lista, lista_apagar, tamanho_janela, lista_indice_janela, indice_passo, maximo_itens, folha_janela)
      
    # Junta em uma string as palavras das sentenças da janela
    lista_janela_sentenca = []
    for sentenca in lista_janela:
      lista_janela_sentenca.append(" ".join(sentenca))

    return lista_janela, " ".join(lista_janela_sentenca), lista_indice_janela, lista_centro_janela