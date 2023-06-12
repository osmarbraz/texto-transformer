# Import das bibliotecas.
import logging  # Biblioteca de logging
import torch # Biblioteca de aprendizado de máquina

# Import de bibliotecas próprias
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *

from spacynlp.spacymodulo import *

from medidor.medidas import *

# ============================
def getDocumentoLista(listaSentencas):
    '''
    Recebe uma lista de sentenças e faz a concatenação em uma string.
    
    Parâmetros:
    `listaDocumento` - Uma lista contendo diversas sentenças.           
    '''

    stringDocumento = ''  
    # Concatena as sentenças do documento
    for sentenca in listaSentencas:                
        stringDocumento = stringDocumento + sentenca

# ============================
def getListaSentencasDocumento(documento, nlp):
    '''
    Retorna uma lista com as sentenças de um documento. Utiliza o spacy para dividir o documento em sentenças.
    
    Parâmetros:
    `documento` - Um documento a ser convertido em uma lista de sentenças.           
    `nlp` - Um objeto de sentenciação de textos.           
       
    '''

    # Aplica sentenciação do spacy no documento
    doc = nlp(documento) 

    # Lista para as sentenças
    lista = []
    # Percorre as sentenças
    for sentenca in doc.sents: 
        # Adiciona as sentenças a lista
        lista.append(str(sentenca))

    return lista

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

# ============================
def getDocumentoTokenizado(documento, tokenizador):

    '''
    Retorna um documento tokenizado e concatenado com tokens especiais '[CLS]' no início e o token '[SEP]' no fim para ser submetido ao BERT.
    
    Parâmetros:
    `documento` - Um documento a ser tokenizado para o BERT.
    `tokenizador` - Tokenizador BERT.
    
    Retorno:
    `documentoTokenizado` - Documento tokenizado.
    '''

    # Adiciona os tokens especiais.
    documentoMarcado = '[CLS] ' + documento + ' [SEP]'

    # Documento tokenizado
    documentoTokenizado = tokenizador.tokenize(documentoMarcado)

    return documentoTokenizado

# ============================
# Constantes para padronizar o acesso aos dados do modelo do BERT.

TEXTO_TOKENIZADO = 0
INPUT_IDS = 1
ATTENTION_MASK = 2
TOKEN_TYPE_IDS = 3
OUTPUTS = 4
OUTPUTS_LAST_HIDDEN_STATE = 0
OUTPUTS_POOLER_OUTPUT = 1
OUTPUTS_HIDDEN_STATES = 2
 
# ============================
def getEmbeddingsTodasCamadas(documento, modelo, tokenizador):    
    '''   
    Retorna os embeddings de todas as camadas de um documento.
    
    Parâmetros:
    `documento` - Um documento a ser recuperado os embeddings do BERT.
    `model` - Modelo BERT.
    `tokenizador` - Tokenizador BERT.
    
    Retorno:
    `documentoTokenizado` - Documento tokenizado.
    `input_ids` - Input ids do documento.
    `attention_mask` - Máscara de atenção do documento
    `token_type_ids` - Token types ids do documento.
    `outputs` - Embeddings do documento.
    '''

    # Documento tokenizado
    documentoTokenizado = getDocumentoTokenizado(documento, tokenizador)

    #print('O documento (', documento, ') tem tamanho = ', len(documentoTokenizado), ' = ', documentoTokenizado)

    # Recupera a quantidade tokens do documento tokenizado.
    qtdeTokens = len(documentoTokenizado)

    #tokeniza o documento e retorna os tensores.
    dicCodificado = tokenizador.encode_plus(
                                            documento, # Documento a ser codificado.
                                            add_special_tokens=True, # Adiciona os tokens especiais '[CLS]' e '[SEP]'
                                            max_length=qtdeTokens, # Define o tamanho máximo para preencheer ou truncar.
                                            truncation=True, # Trunca o documento por max_length
                                            padding='max_length', # Preenche o documento até max_length
                                            return_attention_mask=True, # Constrói a máscara de atenção.
                                            return_tensors='pt' # Retorna os dados como tensores pytorch.
                                            )
    
    # Ids dos tokens de entrada mapeados em seus índices do vocabuário.
    input_ids = dicCodificado['input_ids']

    # Máscara de atenção de cada um dos tokens como pertencentes à sentença '1'.
    attention_mask = dicCodificado['attention_mask']

    # Recupera os tensores dos segmentos.
    token_type_ids = dicCodificado['token_type_ids']

    # Roda o documento através do BERT, e coleta todos os estados ocultos produzidos.
    # das 12 camadas. 
    with torch.no_grad():

        # Passe para a frente, calcule as previsões outputs.     
        outputs = modelo(input_ids=input_ids, 
                         attention_mask=attention_mask)

        # A avaliação do modelo retorna um número de diferentes objetos com base em
        # como é configurado na chamada do método `from_pretrained` anterior. Nesse caso,
        # porque definimos `output_hidden_states = True`, o terceiro item será o
        # estados ocultos(hidden_states) de todas as camadas. Veja a documentação para mais detalhes:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel

        # Retorno de model quando ´output_hidden_states=True´ é setado:    
        # outputs[0] = last_hidden_state, outputs[1] = pooler_output, outputs[2] = hidden_states
        # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
        
        # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    return documentoTokenizado, input_ids, attention_mask, token_type_ids, outputs

# ============================
# getEmbeddingsTodasCamadasBuffer
# Cria um buffer com os embeddings de sentenças para economizar no processamento.
buffer_embeddings = {}

# ============================
def getEmbeddingsTodasCamadasBuffer(S, modelo, tokenizador):
    '''
    Retorna os embeddings de uma sentença de um buffer ou do modelo..
    '''
    
    # Se está no dicionário retorna o embedding
    if S in buffer_embeddings:
        return buffer_embeddings.get(S)
    else:
        # Gera o embedding
        totalCamada = getEmbeddingsTodasCamadas(S, modelo, tokenizador)
        buffer_embeddings.update({S: totalCamada})
        return totalCamada

# ============================
def limpaBufferEmbedding():
    '''
    Esvazia o buffer de embeddings das sentenças.
    '''
    
    buffer_embeddings.clear()

# ============================
def getEmbeddingPrimeiraCamada(sentencaEmbedding):
    '''
    Retorna os embeddings da primeira camada.
    '''
    
    # Cada elemento do vetor sentencaEmbeddinging é formado por:  
    # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
    #[4]outpus e [2]hidden_states 
    #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
  
    # Retorna todas a primeira(-1) camada
    # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
    resultado = sentencaEmbedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][0]
    # Retorno: (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
    #print('resultado=',resultado.size())

    return resultado

# ============================
def getEmbeddingPenultimaCamada(sentencaEmbedding):
    '''
    Retorna os embeddings da penúltima camada.
    '''
    
    # Cada elemento do vetor sentencaEmbedding é formado por:  
    # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
    #[4]outpus e [2]hidden_states 
    #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
  
    # Retorna todas a penúltima(-2) camada
    # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
    resultado = sentencaEmbedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][-2]
    # Retorno: (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
    #print('resultado=',resultado.size())

    return resultado

# ============================
def getEmbeddingUltimaCamada(sentencaEmbedding):
    '''
    Retorna os embeddings da última camada.
    '''
    
    # Cada elemento do vetor sentencaEmbedding é formado por:  
    # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
    #[4]outpus e [2]hidden_states 
    #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
  
    # Retorna todas a última(-1) camada
    # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
    resultado = sentencaEmbedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][-1]
    # Retorno: (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
    #print('resultado=',resultado.size())
  
    return resultado    

# ============================
def getEmbeddingSoma4UltimasCamadas(sentencaEmbedding):
    '''
    Retorna os embeddings da soma das 4 últimas camadas.
    '''
    
    # Cada elemento do vetor sentencaEmbedding é formado por:  
    # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
    #[4]outpus e [2]hidden_states 
    #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
  
    # Retorna todas as 4 últimas camadas
    # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
    embeddingCamadas = sentencaEmbedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][-4:]
    # Retorno: List das camadas(4) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  

    # Usa o método `stack` para criar uma nova dimensão no tensor 
    # com a concateção dos tensores dos embeddings.        
    # Entrada: List das camadas(4) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
    resultadoStack = torch.stack(embeddingCamadas, dim=0)
    # Retorno: <4> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
    #print('resultadoStack=',resultadoStack.size())
  
    # Realiza a soma dos embeddings de todos os tokens para as camadas
    # Entrada: <4> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
    resultado = torch.sum(resultadoStack, dim=0)
    # Saida: <1(lote)> x <qtde_tokens> x <768 ou 1024>
    #print('resultado=',resultado.size())

    return resultado

# ============================
def getEmbeddingConcat4UltimasCamadas(sentencaEmbedding):
    '''
    Retorna os embeddings da concatenação das 4 últimas camadas.
    '''
    
    # Cada elemento do vetor sentencaEmbedding é formado por:  
    # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
    #[4]outpus e [2]hidden_states 
    #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
    
    # Cria uma lista com os tensores a serem concatenados
    # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
    # Lista com os tensores a serem concatenados
    listaConcat = []
    # Percorre os 4 últimos
    for i in [-1, -2, -3, -4]:
        # Concatena da lista
        listaConcat.append(sentencaEmbedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][i])
    # Retorno: Entrada: List das camadas(4) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
    #print('listaConcat=',len(listaConcat))

    # Realiza a concatenação dos embeddings de todos as camadas
    # Retorno: Entrada: List das camadas(4) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
    resultado = torch.cat(listaConcat, dim=-1)
    # Retorno: Entrada: (<1(lote)> x <qtde_tokens> <3072 ou 4096>)  
    # print('resultado=',resultado.size())
  
    return resultado   

# ============================
def getEmbeddingSomaTodasAsCamada(sentencaEmbedding):
    '''
    Retorna os embeddings da soma de todas as camadas.
    '''
    
    # Cada elemento do vetor sentencaEmbedding é formado por:  
    # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
    #[4]outpus e [2]hidden_states 
    #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
  
    # Retorna todas as camadas descontando a primeira(0)
    # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
    embeddingCamadas = sentencaEmbedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][1:]
    # Retorno: List das camadas(12 ou 24) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  

    # Usa o método `stack` para criar uma nova dimensão no tensor 
    # com a concateção dos tensores dos embeddings.        
    # Entrada: List das camadas(12 ou 24) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
    resultadoStack = torch.stack(embeddingCamadas, dim=0)
    # Retorno: <12 ou 24> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
    #print('resultadoStack=',resultadoStack.size())
  
    # Realiza a soma dos embeddings de todos os tokens para as camadas
    # Entrada: <12 ou 24> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
    resultado = torch.sum(resultadoStack, dim=0)
    # Saida: <1(lote)> x <qtde_tokens> x <768 ou 1024>
    # print('resultado=',resultado.size())
  
    return resultado

# ============================
def getResultadoEmbeddings(sentencaEmbedding, camada):
    '''
    Retorna o resultado da operação sobre os embeddings das camadas de acordo com tipo de camada especificada.
    
    Parâmetros:
    `sentencaEmbedding` - Embeddings da stentença.
    `camada` - Camada dos embeddings.
    '''

    # Cada elemento do vetor sentencaEmbedding é formado por:  
    # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
    # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
    #[4]outpus e [2]hidden_states 
    #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
    # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>) 

    resultadoEmbeddingCamadas = None
  
    if camada[LISTATIPOCAMADA_ID] == PRIMEIRA_CAMADA:
        resultadoEmbeddingCamadas = getEmbeddingPrimeiraCamada(sentencaEmbedding)
        #print('resultadoEmbeddingCamadas1=',resultadoEmbeddingCamadas.size())
    else:
        if camada[LISTATIPOCAMADA_ID] == PENULTIMA_CAMADA:
            resultadoEmbeddingCamadas = getEmbeddingPenultimaCamada(sentencaEmbedding)
            #print('resultadoEmbeddingCamadas1=',resultadoEmbeddingCamadas.size())
        else:
            if camada[LISTATIPOCAMADA_ID] == ULTIMA_CAMADA:
                resultadoEmbeddingCamadas = getEmbeddingUltimaCamada(sentencaEmbedding)
                #print('resultadoEmbeddingCamadas2=',resultadoEmbeddingCamadas.size())
            else:
                if camada[LISTATIPOCAMADA_ID] == SOMA_4_ULTIMAS_CAMADAS:
                    resultadoEmbeddingCamadas = getEmbeddingSoma4UltimasCamadas(sentencaEmbedding)            
                    #print('resultadoEmbeddingCamadas3=',resultadoEmbeddingCamadas.size())
                else:
                    if camada[LISTATIPOCAMADA_ID] == CONCAT_4_ULTIMAS_CAMADAS:
                        resultadoEmbeddingCamadas = getEmbeddingConcat4UltimasCamadas(sentencaEmbedding)
                        #print('resultadoEmbeddingCamadas4=',resultadoEmbeddingCamadas.size())
                    else:
                        if camada[LISTATIPOCAMADA_ID] == TODAS_AS_CAMADAS:
                            resultadoEmbeddingCamadas = getEmbeddingSomaTodasAsCamada(sentencaEmbedding)
                            #print('resultadoEmbeddingCamadas5=',resultadoEmbeddingCamadas.size())
                            # Retorno: <1> x <qtde_tokens> x <768 ou 1024>
      
    # Verifica se a primeira dimensão é igual 1 para remover a dimensão de lote 'batches'
    # Entrada: <1> x <qtde_tokens> x <768 ou 1024>
    if resultadoEmbeddingCamadas.shape[0] == 1:
        # Remove a dimensão 0 caso seja de tamanho 1.
        # Usa o método 'squeeze' para remover a primeira dimensão(0) pois possui tamanho 1
        # Entrada: <1> x <qtde_tokens> x <768 ou 1024>
        resultadoEmbeddingCamadas = torch.squeeze(resultadoEmbeddingCamadas, dim=0)     
    #print('resultadoEmbeddingCamadas2=', resultadoEmbeddingCamadas.size())    
    # Retorno: <qtde_tokens> x <768 ou 1024>
  
    # Retorna o resultados dos embeddings dos tokens da sentença  
    return resultadoEmbeddingCamadas

# ============================
def getMedidasSentencasEmbeddingMEAN(embeddingSi, embeddingSj):
    '''
    Retorna as medidas de duas sentenças Si e Sj utilizando a estratégia MEAN.
    
    Parâmetros:
    `embeddingSi` - Embeddings da primeira sentença.
    `embeddingSj` - Embeddings da segunda sentença.
    
    Retorno:
    `Scos` - Similaridade do cosseno - usando a média dos embeddings Si e Sj das camadas especificadas.
    `Seuc` - Distância euclidiana - usando a média dos embeddings Si e Sj das camadas especificadas.
    `Sman` - Distância de manhattan - usando a média dos embeddings Si e Sj das camadas especificadas.
    '''

    #print('embeddingSi=', embeddingSi.shape) 
    #print('embeddingSj=', embeddingSj.shape)
  
    # As operações de subtração(sub), mul(multiplicação/produto), soma(sum), cosseno(similaridade), euclediana(diferença) e manhattan(diferença)
    # Necessitam que os embeddings tenha a mesmo número de dimensões.
  
    # Calcula a média dos embeddings para os tokens de Si, removendo a primeira dimensão.
    # Entrada: <qtde_tokens> x <768 ou 1024>  
    mediaEmbeddingSi = torch.mean(embeddingSi, dim=0)    
    # Retorno: <768 ou 1024>
    #print('mediaCamadasSi=', mediaCamadasSi.shape)
  
    # Calcula a média dos embeddings para os tokens de Sj, removendo a primeira dimensão.
    # Entrada: <qtde_tokens> x <768 ou 1024>  
    mediaEmbeddingSj = torch.mean(embeddingSj, dim=0)    
    # Retorno: <768 ou 1024>
    #print('mediaCamadasSj=', mediaCamadasSj.shape)
    
    # Similaridade do cosseno entre os embeddings Si e Sj
    # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
    Scos = similaridadeCoseno(mediaEmbeddingSi, mediaEmbeddingSj)
    # Retorno: Número real

    # Distância euclidiana entre os embeddings Si e Sj
    # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
    Seuc = distanciaEuclidiana(mediaEmbeddingSi, mediaEmbeddingSj)
    # Retorno: Número real

    # Distância de manhattan entre os embeddings Si e Sj
    # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
    Sman = distanciaManhattan(mediaEmbeddingSi, mediaEmbeddingSj)
    # Retorno: Número real

    # Retorno das medidas das sentenças  
    return mediaEmbeddingSi, mediaEmbeddingSj, Scos, Seuc, Sman

# ============================
def getMedidasSentencasEmbeddingMAX(embeddingSi, embeddingSj):
    '''
    Retorna as medidas de duas sentenças Si e Sj utilizando a estratégia MAX.
    
    Parâmetros:
    `embeddingSi` - Embeddings da primeira sentença.
    `embeddingSj` - Embeddings da segunda sentença.
       
    Retorno:
    `Scos` - Similaridade do cosseno - usando o maior dos embeddings Si e Sj das camadas especificadas.
    `Seuc` - Distância euclidiana - usando o maior dos embeddings Si e Sj das camadas especificadas.
    `Sman` - Distância de manhattan - usando o maior dos embeddings Si e Sj das camadas especificadas.
    '''

    #print('embeddingSi=', embeddingSi.shape) 
    #print('embeddingSj=', embeddingSj.shape)

    # As operações de subtração(sub), mul(multiplicação/produto), soma(sum), cosseno(similaridade), euclediana(diferença) e manhattan(diferença)
    # Necessitam que os embeddings tenha a mesmo número de dimensões.

    # Encontra os maiores embeddings os tokens de Si, removendo a primeira dimensão.
    # Entrada: <qtde_tokens> x <768 ou 1024>  
    maiorEmbeddingSi, linha = torch.max(embeddingSi, dim=0)    
    # Retorno: <768 ou 1024>
    #print('maiorEmbeddingSi=', maiorEmbeddingSi.shape)

    # Encontra os maiores embeddings os tokens de Sj, removendo a primeira dimensão.
    # Entrada: <qtde_tokens> x <768 ou 1024>  
    maiorEmbeddingSj, linha = torch.max(embeddingSj, dim=0)    
    # Retorno: <768 ou 1024>
    #print('maiorEmbeddingSj=', maiorEmbeddingSj.shape)

    # Similaridade do cosseno entre os embeddings Si e Sj
    # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
    Scos = similaridadeCoseno(maiorEmbeddingSi, maiorEmbeddingSj)
    # Retorno: Número real

    # Distância euclidiana entre os embeddings Si e Sj
    # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
    Seuc = distanciaEuclidiana(maiorEmbeddingSi, maiorEmbeddingSj)
    # Retorno: Número real

    # Distância de manhattan entre os embeddings Si e Sj
    # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
    Sman = distanciaManhattan(maiorEmbeddingSi, maiorEmbeddingSj)
    # Retorno: Número real

    # Retorno das medidas das sentenças
    return maiorEmbeddingSi, maiorEmbeddingSj, Scos, Seuc, Sman

# ============================
def getMedidasSentencasEmbedding(embeddingSi, embeddingSj, estrategia_pooling):
    '''
    Realiza o cálculo da medida do documento de acordo com a estratégia de pooling(MAX ou MEAN).
    
    Parâmetros:
    `embeddingSi` - Embeddings da primeira sentença.
    `embeddingSj` - Embeddings da segunda sentença.
    `estrategia_pooling` - Estratégia de pooling a ser utilizada.       
    '''

    if estrategia_pooling == 0:
        return getMedidasSentencasEmbeddingMEAN(embeddingSi, embeddingSj)
    else:
        return getMedidasSentencasEmbeddingMAX(embeddingSi, embeddingSj)

# ============================
def getEmbeddingSentencaEmbeddingDocumentoALL(embeddingDocumento, documento, sentenca, tokenizador):
    '''
    Retorna os embeddings de uma sentença com todas as palavras(ALL) a partir dos embeddings do documento.
    
    '''
        
    # Tokeniza o documento
    documentoTokenizado = getDocumentoTokenizado(documento, tokenizador)
    #print(documentoTokenizado)

    # Tokeniza a sentença
    sentencaTokenizada = getDocumentoTokenizado(sentenca, tokenizador)
    #print(sentencaTokenizada)
    # Remove os tokens de início e fim da sentença
    sentencaTokenizada.remove('[CLS]')
    sentencaTokenizada.remove('[SEP]')    
    #print(len(sentencaTokenizada))

    # Localiza os índices dos tokens da sentença no documento
    inicio, fim = encontrarIndiceSubLista(documentoTokenizado, sentencaTokenizada)
    #print(inicio,fim) 

    # Recupera os embeddings dos tokens da sentença a partir dos embeddings do documento
    embeddingSentenca = embeddingDocumento[inicio:fim + 1]
    #print('embeddingSentenca=', embeddingSentenca.shape)

    # Retorna o embedding da sentença no documento
    return embeddingSentenca

# ============================
def getEmbeddingSentencaEmbeddingDocumentoCLEAN(embeddingDocumento, documento, sentenca, tokenizador, stopwords):
    '''
    Retorna os embeddings de uma sentença sem stopwords(CLEAN) a partir dos embeddings do documento.
    '''
      
    # Tokeniza o documento
    documentoTokenizado = getDocumentoTokenizado(documento, tokenizador)  
    #print(documentoTokenizado)

    # Remove as stopword da sentença
    sentencaSemStopWord = removeStopWord(sentenca, stopwords)

    # Tokeniza a sentença sem stopword
    sentencaTokenizadaSemStopWord = getDocumentoTokenizado(sentencaSemStopWord, tokenizador)
    #print(sentencaTokenizadaSemStopWord)

    # Remove os tokens de início e fim da sentença
    sentencaTokenizadaSemStopWord.remove('[CLS]')
    sentencaTokenizadaSemStopWord.remove('[SEP]')    
    #print(len(sentencaTokenizadaSemStopWord))

    # Tokeniza a sentença
    sentencaTokenizada = getDocumentoTokenizado(sentenca, tokenizador)

    # Remove os tokens de início e fim da sentença
    sentencaTokenizada.remove('[CLS]')
    sentencaTokenizada.remove('[SEP]')  
    #print(sentencaTokenizada)
    #print(len(sentencaTokenizada))

    # Localiza os índices dos tokens da sentença no documento
    inicio, fim = encontrarIndiceSubLista(documentoTokenizado, sentencaTokenizada)
    #print('Sentença inicia em:', inicio, 'até', fim) 

    # Recupera os embeddings dos tokens da sentença a partir dos embeddings do documento
    embeddingSentenca = embeddingDocumento[inicio:fim + 1]
    #print('embeddingSentenca=', embeddingSentenca.shape)

    # Lista com os tensores selecionados
    listaTokensSelecionados = []
    # Localizar os embeddings dos tokens da sentença tokenizada sem stop word na sentença 
    # Procura somente no intervalo da sentença
    for i, tokenSentenca in enumerate(sentencaTokenizada):
        for tokenSentencaSemStopWord in sentencaTokenizadaSemStopWord: 
            # Se o token da sentença é igual ao token da sentença sem stopword    
            if tokenSentenca == tokenSentencaSemStopWord:
                listaTokensSelecionados.append(embeddingSentenca[i:i + 1])

    embeddingSentencaSemStopWord = None

    if len(listaTokensSelecionados) != 0:
        # Concatena os vetores da lista pela dimensão 0
        embeddingSentencaSemStopWord = torch.cat(listaTokensSelecionados, dim=0)
        #print("embeddingSentencaSemStopWord:",embeddingSentencaSemStopWord.shape)

    # Retorna o embedding da sentença no documento
    return embeddingSentencaSemStopWord

# ============================
def getEmbeddingSentencaEmbeddingDocumentoNOUN(embeddingDocumento, documento, sentenca, tokenizador, nlp, tipo_palavra_relevante='NOUN'):
    '''
    Retorna os embeddings de uma sentença somente com as palavras relevantes(NOUN) de um tipo a partir dos embeddings do documento.
    '''

    # Tokeniza o documento
    documentoTokenizado = getDocumentoTokenizado(documento, tokenizador)  
    #print(documentoTokenizado)

    # Retorna as palavras relevantes da sentença do tipo especificado
    sentencaSomenteRelevante = retornaPalavraRelevante(sentenca, nlp, tipo_palavra_relevante)

    # Tokeniza a sentença 
    sentencaTokenizadaSomenteRelevante = getDocumentoTokenizado(sentencaSomenteRelevante, tokenizador)

    # Remove os tokens de início e fim da sentença
    sentencaTokenizadaSomenteRelevante.remove('[CLS]')
    sentencaTokenizadaSomenteRelevante.remove('[SEP]')  
    #print(sentencaTokenizadaSomenteRelevante)
    #print(len(sentencaTokenizadaSomenteRelevante))

    # Tokeniza a sentença
    sentencaTokenizada = getDocumentoTokenizado(sentenca, tokenizador)

    # Remove os tokens de início e fim da sentença
    sentencaTokenizada.remove('[CLS]')
    sentencaTokenizada.remove('[SEP]')  
    #print(sentencaTokenizada)
    #print(len(sentencaTokenizada))

    # Localiza os índices dos tokens da sentença no documento
    inicio, fim = encontrarIndiceSubLista(documentoTokenizado, sentencaTokenizada)
    #print('Sentença inicia em:', inicio, 'até', fim) 

    # Recupera os embeddings dos tokens da sentença a partir dos embeddings do documento
    embeddingSentenca = embeddingDocumento[inicio:fim + 1]
    #print('embeddingSentenca=', embeddingSentenca.shape)

    # Lista com os tensores selecionados
    listaTokensSelecionados = []
    # Localizar os embeddings dos tokens da sentença tokenizada sem stop word na sentença 
    # Procura somente no intervalo da sentença
    for i, tokenSentenca in enumerate(sentencaTokenizada):
        for tokenSentencaSomenteRelevante in sentencaTokenizadaSomenteRelevante: 
            if tokenSentenca == tokenSentencaSomenteRelevante:        
                listaTokensSelecionados.append(embeddingSentenca[i:i + 1])

    embeddingSentencaComSubstantivo = None

    if len(listaTokensSelecionados) != 0:
        # Concatena os vetores da lista pela dimensão 0  
        embeddingSentencaComSubstantivo = torch.cat(listaTokensSelecionados, dim=0)
        #print("embeddingSentencaComSubstantivo:",embeddingSentencaComSubstantivo.shape)

    # Retorna o embedding da sentença do documento
    return embeddingSentencaComSubstantivo

# ============================
def getEmbeddingSentencaEmbeddingDocumento(embeddingDocumento, documento, sentenca, tokenizador, nlp, palavra_relevante=0):
    '''
    Retorna os embeddings de uma sentença considerando a relevância das palavras (ALL, CLEAN ou NOUN) a partir dos embeddings do documento.    
    '''

    if palavra_relevante == 0:
        return getEmbeddingSentencaEmbeddingDocumentoALL(embeddingDocumento, documento, sentenca, tokenizador)
    else:
        if palavra_relevante == 1:
            stopwords = getStopwords(nlp)
            return getEmbeddingSentencaEmbeddingDocumentoCLEAN(embeddingDocumento, documento, sentenca, tokenizador, stopwords)
        else:
            if palavra_relevante == 2:
                return getEmbeddingSentencaEmbeddingDocumentoNOUN(embeddingDocumento, documento, sentenca, tokenizador, nlp, tipo_palavra_relevante='NOUN')

# ============================
def getMedidasCoerenciaDocumento(documento, modelo, tokenizador, nlp, camada, tipoDocumento='p', estrategia_pooling=0, palavra_relevante=0):
    '''
    Retorna as medidas de coerência do documento.
    Considera somente sentenças com pelo menos uma palavra.
    Estratégia de pooling padrão é MEAN(0).
    Palavra relavante padrão é ALL(0).
    '''

    # Quantidade de sentenças no documento
    n = len(documento)
    
    # Divisor da quantidade de documentos
    divisor = n - 1

    # Documento é uma lista com as sentenças
    #print('camada=',camada)
    #print('Documento=', documento)

    # Junta a lista de sentenças em um documento(string)
    stringDocumento = ' '.join(documento)

    # Envia o documento ao MCL e recupera os embeddings de todas as camadas
    # Se for o documento original pega do buffer para evitar a repetição
    if tipoDocumento == 'o':
        # Retorna os embeddings de todas as camadas do documento
        # O embedding possui os seguintes valores        
        # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
        totalCamadasDocumento = getEmbeddingsTodasCamadasBuffer(stringDocumento, modelo, tokenizador)      
        # Retorno: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> <768 ou 1024>) 
    else:
        # Retorna os embeddings de todas as camadas do documento
        # O embedding possui os seguintes valores        
        # 0-documento_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
        totalCamadasDocumento = getEmbeddingsTodasCamadas(stringDocumento, modelo, tokenizador)      
        # Retorno: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> <768 ou 1024>) 

    # Recupera os embeddings dos tokens das camadas especificadas de acordo com a estratégia especificada para camada  
    embeddingDocumento = getResultadoEmbeddings(totalCamadasDocumento, camada=camada)
    #print('embeddingDocumento=', embeddingDocumento.shape)

    # Acumuladores das medidas entre as sentenças  
    somaScos = 0
    somaSeuc = 0
    somaSman = 0

    # Seleciona os pares de sentença a serem avaliados
    posSi = 0
    posSj = posSi + 1

    #Enquanto o indíce da sentneça posSj(2a sentença) não chegou ao final da quantidade de sentenças
    while posSj <= (n-1):  

        # Seleciona as sentenças do documento  
        Si = documento[posSi]
        Sj = documento[posSj]

        # Recupera os embedding das sentenças Si e Sj do embedding do documento      
        embeddingSi = getEmbeddingSentencaEmbeddingDocumento(embeddingDocumento, stringDocumento, Si, tokenizador, nlp, palavra_relevante=palavra_relevante)
        embeddingSj = getEmbeddingSentencaEmbeddingDocumento(embeddingDocumento, stringDocumento, Sj, tokenizador, nlp, palavra_relevante=palavra_relevante)

        # Verifica se os embeddings sentenças estão preenchidos
        if embeddingSi != None and embeddingSj != None:

            # Recupera as medidas entre Si e Sj     
            ajustadoEmbeddingSi, ajustadoEmbeddingSj, Scos, Seuc, Sman = getMedidasSentencasEmbedding(embeddingSi, embeddingSj, estrategia_pooling=estrategia_pooling)

            # Acumula as medidas
            somaScos = somaScos + Scos
            somaSeuc = somaSeuc + Seuc
            somaSman = somaSman + Sman

                # avança para o próximo par de sentenças
            posSi = posSj
            posSj = posSj + 1
        else:
            # Reduz um da quantidade de sentenças pois uma delas está vazia
            divisor = divisor - 1
            # Se embeddingSi igual a None avanca pos1 e pos2
            if embeddingSi == None:
                # Avança a posição da sentença posSi para a posSj
                posSi = posSj
                # Avança para a próxima sentença de posSj
                posSj = posSj + 1        
            else:          
                # Se embeddingSj = None avança somente posJ para a próxima sentença
                if embeddingSj == None:
                    posSj = posSj + 1

    # Calcula a medida 
    Ccos = 0
    Ceuc = 0
    Cman = 0

    if divisor != 0:
        Ccos = float(somaScos) / float(divisor)
        Ceuc = float(somaSeuc) / float(divisor)
        Cman = float(somaSman) / float(divisor)

    return Ccos, Ceuc, Cman

# ============================
# listaTipoCamadas
# Define uma lista com as camadas a serem analisadas nos teste.
# Cada elemento da lista 'listaTipoCamadas' é chamado de camada sendo formado por:
#  - camada[0] = Índice da camada
#  - camada[1] = Um inteiro com o índice da camada a ser avaliada. Pode conter valores negativos.
#  - camada[2] = Operação para n camadas, CONCAT ou SUM.
#  - camada[3] = Nome do tipo camada

# Os nomes do tipo da camada pré-definidos.
#  - 0 - Primeira                    
#  - 1 - Penúltima
#  - 2 - Última
#  - 3 - Soma 4 últimas
#  - 4 - Concat 4 últimas
#  - 5 - Todas

# Constantes para facilitar o acesso os tipos de camadas
PRIMEIRA_CAMADA = 0
PENULTIMA_CAMADA = 1
ULTIMA_CAMADA = 2
SOMA_4_ULTIMAS_CAMADAS = 3
CONCAT_4_ULTIMAS_CAMADAS = 4
TODAS_AS_CAMADAS = 5

# Índice dos campos da camada
LISTATIPOCAMADA_ID = 0
LISTATIPOCAMADA_CAMADA = 1
LISTATIPOCAMADA_OPERACAO = 2
LISTATIPOCAMADA_NOME = 3

# BERT Large possui 25 camadas(1a camada com os tokens de entrada e 24 camadas dos transformers)
# BERT Large possui 13 camadas(1a camada com os tokens de entrada e 24 camadas dos transformers)
# O índice da camada com valor positivo indica uma camada específica
# O índica com um valor negativo indica as camadas da posição com base no fim descontado o valor indice até o fim.
listaTipoCamadas = [
    [PRIMEIRA_CAMADA, 1, '-', 'Primeira'], 
    [PENULTIMA_CAMADA, -2, '-', 'Penúltima'],
    [ULTIMA_CAMADA, -1, '-', 'Última'],
    [SOMA_4_ULTIMAS_CAMADAS, -4, 'SUM', 'Soma 4 últimas'],
    [CONCAT_4_ULTIMAS_CAMADAS, -4, 'CONCAT', 'Concat 4 últimas'], 
    [TODAS_AS_CAMADAS, 24, 'SUM', 'Todas']
]

# listaTipoCamadas e suas referências:
# 0 - Primeira            listaTipoCamadas[PRIMEIRA_CAMADA]
# 1 - Penúltima           listaTipoCamadas[PENULTIMA_CAMADA]
# 2 - Última              listaTipoCamadas[ULTIMA_CAMADA]
# 3 - Soma 4 últimas      listaTipoCamadas[SOMA_4_ULTIMAS_CAMADAS]
# 4 - Concat 4 últimas    listaTipoCamadas[CONCAT_4_ULTIMAS_CAMADAS]
# 5 - Todas               listaTipoCamadas[TODAS_AS_CAMADAS]

# ============================
def comparaMedidasCamadasSentencas(Si, Sj, modelo, tokenizador, camada):
    '''
    Facilita a exibição dos valores de comparação de duas orações.
    '''
  
    # Recupera os embeddings da sentença Si e sentença Sj e suas medidas
    embeddingSi, embeddingSj, Scos, Seuc, Sman = getMedidasCamadasSentencas(Si, Sj, modelo, tokenizador, camada)

    logging.info('  ->Mostra comparação da ' + camada[LISTATIPOCAMADA_NOME] + ' camada(s)')    
    logging.info('   Cosseno(SixSj)     = %.8f' % Scos)
    logging.info('   Euclidiana(SixSj)  = %.8f' % Seuc)
    logging.info('   Manhattan(SixSj)   = %.8f' % Sman)
