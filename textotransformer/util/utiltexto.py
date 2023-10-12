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
def truncaJanela(lista_janela,                  
                 maximo_itens, 
                 lista_centro_janela):
    """
     Trunca as palavras da janela até o máximo da janela.

     Parâmetros:
       Parâmetros:
       `lista_janela` - Um dataframe com os itens.
       `maximo_itens` - Máximo de itens na janela. Trunca das extremidades preservando a palavra central.
       `lista_centro_janela` - Lista com os índices dos centros da janela.

     Retorno:
       `lista_janela` - Janela truncada pelo máximo de itens.
    """
    # Quantidade de itens nas janelas antes
    qtde_itens1 = contaItensLista(lista_janela)
    # print("quantidade de itens janela antes:", qtde_itens1)

    # Controle se não alcançado o máximo de palavras
    minimo_alcancado = False
   
    # Indices para os elementos a serem excluídos
    indice_esquerda = 0
    indice_direita = len(lista_janela) -1

    # Remove as palavras das extremidade que ultrapassam o tamanho máximo
    while qtde_itens1 > maximo_itens and minimo_alcancado == False:
      
      # Recupera os intervalo das folhas da direita e esquerda e centro
      # Intervalo da folha da esquerda do centro
      # Sempre inicia em 0
      inicio_folha_esquerda = 0
      fim_folha_esquerda = lista_centro_janela[0]

      # Intervalo do centro
      inicio_centro_esquerda = lista_centro_janela[0]
      fim_centro_direita = lista_centro_janela[-1]+1

      # Intervalo da folha da direita do centro
      inicio_folha_direita = lista_centro_janela[-1]+1
      # Vai até o final da lista
      fim_folha_direita = len(lista_janela)

      # Conta os elementos dos intervalos
      conta_itens_esquerda = contaItensLista(lista_janela[inicio_folha_esquerda:fim_folha_esquerda])
      conta_itens_centro = contaItensLista(lista_janela[inicio_centro_esquerda:fim_centro_direita])
      conta_itens_direita = contaItensLista(lista_janela[inicio_folha_direita:fim_folha_direita])

      # print("")
      # print("inicio_folha_esquerda :", inicio_folha_esquerda, "/fim_folha_esquerda:", fim_folha_esquerda, " conta:", conta_itens_esquerda)
      # print("inicio_centro_esquerda:",inicio_centro_esquerda,"/fim_centro_direita:", fim_centro_direita, " conta:", conta_itens_centro)
      # print("inicio_folha_direita  :",inicio_folha_direita,"/fim_folha_direita:", fim_folha_direita, " conta:", conta_itens_direita)

      # Se a quantidade de itens a direita for maior apaga deste lado
      if conta_itens_direita > conta_itens_esquerda:
        # Remove da direita
        if len(lista_janela[indice_direita]) > 0:
          # Remove do fim da janela          
          lista_janela[indice_direita].pop()
          if len(lista_janela[indice_direita]) == 0:          
            # Não pode ser menor que o centro
            if indice_direita > lista_centro_janela[-1]:
              indice_direita = indice_direita - 1
      else:
          # Remove da esquerda
          if len(lista_janela[indice_esquerda]) > 0:
            # Remove do inicio da janela
            lista_janela[indice_esquerda].pop(0)
            if len(lista_janela[indice_esquerda]) == 0:          
              # Não pode ser menor que o centro
              if indice_esquerda < lista_centro_janela[0]:
                indice_esquerda = indice_esquerda + 1
              
      # Calcula a nova quantidade de itens
      qtde_itens2 = contaItensLista(lista_janela)

      # Verifica se conseguiu reduzir a quantidade de itens
      if (qtde_itens1 == qtde_itens2):
        print("Atenção!: Truncamento de janela não conseguiu reduzir além de ", qtde_itens2, " para o máximo ", maximo_itens)
        minimo_alcancado = True

      # Atribui uma nova quantidade
      qtde_itens1 = qtde_itens2

    return lista_janela

# ============================
def getJanelaLista(lista, tamanho_janela, indice_passo, maximo_itens=None):
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
       `lista_indice_janela` - Lista com os índices dos itens que forma a janela.
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

    # Concatena em uma string as palavras das itens da janela
    lista_janela_itens = []
    for item in lista_janela:
      lista_janela_itens.append(" ".join(item))

    return lista_janela, " ".join(lista_janela_itens), lista_indice_janela, lista_centro_janela

  else:
    # print(">>>> Lista maior que as janelas")
    # Lista maior que o tamanho da janela
    # Calcula o tamanho da folha da janela(quantidade de itens a esquerda e direita do centro da janela).
    folha_janela = int((tamanho_janela-1) /2)
    # Define o centro da janela
    centro_janela = -1
    # Percorre a lista
    # Dentro do intervalo da lista de itens
    if indice_passo >= 0 and indice_passo < len(lista):
      
      # Guarda os itens da janela
      lista_janela = []
      # Guarda os índices dos itens das janelas
      lista_indice_janela = []
      # Guarda os índices dos centros das janelas
      # Por enquanto somente centro com um elemento
      lista_centro_janela = []

      # Inicio da lista sem janelas completas depois do meio da janela, folha da direita do centro
      if indice_passo < folha_janela:
        # print("Inicio da lista")
        # itens anteriores
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

        # item central
        # Recupera o documento da lista
        documento = lista[indice_passo]
        lista_janela.append(documento[0])
        # Adiciona o indice do documento na lista
        lista_indice_janela.append(indice_passo)
        # Guarda o centro da janela
        centro_janela = len(lista_janela)-1
        lista_centro_janela.append(centro_janela)

        # itens posteriores
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
        # Meio da lista com janelas completas antes e depois, folhas de tamanhos iguais a esquerda e a direita
        if indice_passo < len(lista)-folha_janela:
          # print(" Meio da lista")

          # itens anteriores
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

          # item central
          # Recupera o documento da lista
          documento = lista[indice_passo]
          # Adiciona o documento a janela
          lista_janela.append(documento[0])
          # Adiciona o indice do documento na lista
          lista_indice_janela.append(indice_passo)
          # Guarda o centro da janela
          centro_janela = len(lista_janela)-1
          lista_centro_janela.append(centro_janela)

          # itens posteriores
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
          # Fim da lista sem janelas completas antes do meio da janela, folha da esquerda do centro
          if indice_passo >= len(lista)-folha_janela:
            # print("Fim da lista")

            # itens anteriores
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

            # item central
            # Recupera o documento da lista
            documento = lista[indice_passo]
            # Adiciona o documento a janela
            lista_janela.append(documento[0])
            # Adiciona o indice do documento na lista
            lista_indice_janela.append(indice_passo)
            # Guarda o centro da janela
            centro_janela = len(lista_janela)-1
            lista_centro_janela.append(centro_janela)

            # itens posteriores
            inicio = indice_passo + 1
            fim = indice_passo + 1 + folha_janela
            # Evita o extrapolar o limite da lista de itens
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
      lista_janela = truncaJanela(lista_apagar, maximo_itens, lista_centro_janela)

    # Junta em uma string os itens das listas  da janela
    lista_janela_itens = []
    for item in lista_janela:
      lista_janela_itens.append(" ".join(item))

    return lista_janela, " ".join(lista_janela_itens), lista_indice_janela, lista_centro_janela