# Import das bibliotecas.

# Biblioteca de logging
import logging

# Biblioteca de aprendizado de máquina
import torch 

# Bibliotecas próprias
from textotransformer.modelo.transformer import Transformer
from textotransformer.modelo.modeloarguments import ModeloArgumentos 
from textotransformer.pln.pln import PLN
from textotransformer.mensurador.medidas import distanciaEuclidiana, distanciaManhattan, similaridadeCoseno
from textotransformer.mensurador.mensuradorenum import PalavrasRelevantes
from textotransformer.modelo.modeloenum import LISTATIPOCAMADA_NOME, EmbeddingsCamadas, EstrategiasPooling 
from textotransformer.util.utilconstantes import OUTPUTS, OUTPUTS_HIDDEN_STATES
from textotransformer.util.utiltexto import encontrarIndiceSubLista  

logger = logging.getLogger(__name__)

class Mensurador:

    ''' 
    Realiza mensurações em textos.
     
    Parâmetros:
    `modelo_args` - Parâmetros do modelo de linguagem.
    `transformer` - Modelo de linguagem carregado.
    `pln` - Processador de linguagem natural.
    ''' 

    # Construtor da classe
    def __init__(self, modelo_args: ModeloArgumentos,
                 transformer: Transformer, 
                 pln: PLN):
    
        # Parâmetros do modelo
        self.model_args = modelo_args
    
        # Recupera o objeto do transformer.
        self.transformer = transformer
    
        # Recupera o modelo.
        self.auto_model = transformer.getAutoMmodel()
    
        # Recupera o tokenizador.     
        self.tokenizer = transformer.getTokenizer()
        
        # Recupera a classe PLN
        self.pln = pln
        
        logger.info("Classe Mensurador carregada: {}.".format(modelo_args))

    # ============================   
    def __repr__(self):
        '''
        Retorna uma string com descrição do objeto.
        '''
        
        return "Classe ({}) com  Transformer: {}, tokenizador: {} e NLP: {} ".format(self.__class__.__name__, 
                                                                                     self.auto_model.__class__.__name__,
                                                                                     self.tokenizer.__class__.__name__,
                                                                                     self.pln.__class__.__name__)

    # ============================
    def getEmbeddingsTodasCamadas(self, texto):    
        '''   
        Retorna os embeddings de todas as camadas de um texto.
        
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings do BERT.
        
        Retorno:
        `texto_tokenizado` - Texto tokenizado.
        `input_ids` - Input ids do texto.
        `attention_mask` - Máscara de atenção do texto
        `token_type_ids` - Token types ids do texto.
        `outputs` - Embeddings do texto.
        '''

        # Texto tokenizado
        texto_tokenizado =  self.transformer.getTextoTokenizado(texto)

        #print('O texto (', texto, ') tem tamanho = ', len(texto_tokenizado), ' = ', texto_tokenizado)

        # Recupera a quantidade tokens do texto tokenizado.
        qtdeTokens = len(texto_tokenizado)

        #tokeniza o texto e retorna os tensores.
        dic_codificado = self.tokenizer.encode_plus(
                                                texto, # Texto a ser codificado.
                                                add_special_tokens=True, # Adiciona os tokens especiais '[CLS]' e '[SEP]'
                                                max_length=qtdeTokens, # Define o tamanho máximo para preencheer ou truncar.
                                                truncation=True, # Trunca o texto por max_length
                                                padding='max_length', # Preenche o texto até max_length
                                                return_attention_mask=True, # Constrói a máscara de atenção.
                                                return_tensors='pt' # Retorna os dados como tensores pytorch.
                                                )
        
        # Ids dos tokens de entrada mapeados em seus índices do vocabuário.
        input_ids = dic_codificado['input_ids']

        # Máscara de atenção de cada um dos tokens como pertencentes à sentença '1'.
        attention_mask = dic_codificado['attention_mask']

        # Recupera os tensores dos segmentos.
        token_type_ids = dic_codificado['token_type_ids']

        # Roda o texto através do BERT, e coleta todos os estados ocultos produzidos.
        # das 12 camadas. 
        with torch.no_grad():

            # Passe para a frente, calcule as previsões outputs.     
            outputs = self.auto_model(input_ids=input_ids, 
                                 attention_mask=attention_mask)

            # A avaliação do modelo retorna um número de diferentes objetos com base em
            # como é configurado na chamada do método `from_pretrained` anterior. Nesse caso,
            # porque definimos `output_hidden_states = True`, o terceiro item será o
            # estados ocultos(hidden_states) de todas as camadas. Veja a documentação para mais detalhes:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel

            # Retorno de model quando ´output_hidden_states=True´ é setado:    
            # outputs[0] = last_hidden_state, outputs[1] = pooler_output, outputs[2] = hidden_states
            # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
            
            # 0-texto_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
        return texto_tokenizado, input_ids, attention_mask, token_type_ids, outputs

    # ============================
    # Cria um buffer com os embeddings de sentenças para economizar memória no processamento.
    buffer_embeddings = {}
    
    def getEmbeddingsTodasCamadasBuffer(self, embedding):
        '''
        Retorna os embeddings de uma sentença de um buffer ou do modelo.

        Parâmetros:
        `embedding` - Uma sentença a ser recuperado os embeddings.
        '''
        
        # Se está no dicionário retorna o embedding
        if embedding in self.buffer_embeddings:
            return self.buffer_embeddings.get(embedding)
        else:
            # Gera o embedding
            total_camada = self.getEmbeddingsTodasCamadas(embedding)
            self.buffer_embeddings.update({embedding: total_camada})
            return total_camada

    # ============================
    def limpaBufferEmbedding(self):
        '''
        Esvazia o buffer de embeddings das sentenças.
        '''
        
        self.buffer_embeddings.clear(self)

    # ============================
    def getEmbeddingPrimeiraCamada(self, sentenca_embedding):
        '''
        Retorna os embeddings da primeira camada.
        '''
        
        # Cada elemento do vetor sentenca_embedding é formado por:  
        # 0-texto_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
        # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
        #[4]outpus e [2]hidden_states 
        #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
      
        # Retorna todas a primeira(-1) camada
        # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
        resultado = sentenca_embedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][0]
        # Retorno: (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
        #print('resultado=',resultado.size())

        return resultado

    # ============================
    def getEmbeddingPenultimaCamada(self, sentenca_embedding):
        '''
        Retorna os embeddings da penúltima camada.
        '''
        
        # Cada elemento do vetor sentencaEmbedding é formado por:  
        # 0-texto_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
        # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
        #[4]outpus e [2]hidden_states 
        #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
      
        # Retorna todas a penúltima(-2) camada
        # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
        resultado = sentenca_embedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][-2]
        # Retorno: (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
        #print('resultado=',resultado.size())

        return resultado

    # ============================
    def getEmbeddingUltimaCamada(self, sentenca_embedding):
        '''
        Retorna os embeddings da última camada.
        '''
        
        # Cada elemento do vetor sentencaEmbedding é formado por:  
        # 0-texto_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
        # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
        #[4]outpus e [2]hidden_states 
        #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
      
        # Retorna todas a última(-1) camada
        # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
        resultado = sentenca_embedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][-1]
        # Retorno: (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
        #print('resultado=',resultado.size())
      
        return resultado    

    # ============================
    def getEmbeddingSoma4UltimasCamadas(self, sentenca_embedding):
        '''
        Retorna os embeddings da soma das 4 últimas camadas.
        '''
        
        # Cada elemento do vetor sentencaEmbedding é formado por:  
        # 0-texto_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
        # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
        #[4]outpus e [2]hidden_states 
        #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
      
        # Retorna todas as 4 últimas camadas
        # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
        embedding_camadas = sentenca_embedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][-4:]
        # Retorno: List das camadas(4) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  

        # Usa o método `stack` para criar uma nova dimensão no tensor 
        # com a concateção dos tensores dos embeddings.        
        # Entrada: List das camadas(4) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
        resultado_stack = torch.stack(embedding_camadas, dim=0)
        # Retorno: <4> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
        #print('resultado_stack=',resultado_stack.size())
      
        # Realiza a soma dos embeddings de todos os tokens para as camadas
        # Entrada: <4> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
        resultado = torch.sum(resultado_stack, dim=0)
        # Saida: <1(lote)> x <qtde_tokens> x <768 ou 1024>
        #print('resultado=',resultado.size())

        return resultado

    # ============================
    def getEmbeddingConcat4UltimasCamadas(self, sentenca_embedding):
        '''
        Retorna os embeddings da concatenação das 4 últimas camadas.
        '''
        
        # Cada elemento do vetor sentencaEmbedding é formado por:  
        # 0-texto_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
        # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
        #[4]outpus e [2]hidden_states 
        #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
        
        # Cria uma lista com os tensores a serem concatenados
        # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
        # Lista com os tensores a serem concatenados
        lista_concatenada = []
        # Percorre os 4 últimos
        for i in [-1, -2, -3, -4]:
            # Concatena da lista
            lista_concatenada.append(sentenca_embedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][i])
        # Retorno: Entrada: List das camadas(4) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
        #print('lista_concatenada=',len(lista_concatenada))

        # Realiza a concatenação dos embeddings de todos as camadas
        # Retorno: Entrada: List das camadas(4) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
        resultado = torch.cat(lista_concatenada, dim=-1)
        # Retorno: Entrada: (<1(lote)> x <qtde_tokens> <3072 ou 4096>)  
        # print('resultado=',resultado.size())
      
        return resultado   

    # ============================
    def getEmbeddingSomaTodasAsCamada(self, sentenca_embedding):
        '''
        Retorna os embeddings da soma de todas as camadas.
        '''
        
        # Cada elemento do vetor sentencaEmbedding é formado por:  
        # 0-texto_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
        # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
        #[4]outpus e [2]hidden_states 
        #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
      
        # Retorna todas as camadas descontando a primeira(0)
        # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
        embedding_camadas = sentenca_embedding[OUTPUTS][OUTPUTS_HIDDEN_STATES][1:]
        # Retorno: List das camadas(12 ou 24) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  

        # Usa o método `stack` para criar uma nova dimensão no tensor 
        # com a concateção dos tensores dos embeddings.        
        # Entrada: List das camadas(12 ou 24) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
        resultado_stack = torch.stack(embedding_camadas, dim=0)
        # Retorno: <12 ou 24> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
        #print('resultado_stack=',resultado_stack.size())
      
        # Realiza a soma dos embeddings de todos os tokens para as camadas
        # Entrada: <12 ou 24> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
        resultado = torch.sum(resultado_stack, dim=0)
        # Saida: <1(lote)> x <qtde_tokens> x <768 ou 1024>
        # print('resultado=',resultado.size())
      
        return resultado

    # ============================
    def getResultadoEmbeddings(self, sentenca_embedding, camada):
        '''
        Retorna o resultado da operação sobre os embeddings das camadas de acordo com tipo de camada especificada.
        
        Parâmetros:
        `sentencaEmbedding` - Embeddings da stentença.
        `camada` - Camada dos embeddings.
        '''

        # Cada elemento do vetor sentencaEmbedding é formado por:  
        # 0-texto_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
        # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.
        #[4]outpus e [2]hidden_states 
        #[OUTPUTS]outpus e [OUTPUTS_HIDDEN_STATES]hidden_states      
        # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>) 

        resultado_embedding_camadas = None
      
        if camada == EmbeddingsCamadas.PRIMEIRA_CAMADA.value[0]:
            resultado_embedding_camadas = self.getEmbeddingPrimeiraCamada(sentenca_embedding)
            #print('resultadoEmbeddingCamadas1=',resultadoEmbeddingCamadas.size())
        else:
            if camada == EmbeddingsCamadas.PENULTIMA_CAMADA.value[0]:
                resultado_embedding_camadas = self.getEmbeddingPenultimaCamada(sentenca_embedding)
                #print('resultadoEmbeddingCamadas1=',resultadoEmbeddingCamadas.size())
            else:                
                if camada == EmbeddingsCamadas.ULTIMA_CAMADA.value[0]:
                    resultado_embedding_camadas = self.getEmbeddingUltimaCamada(sentenca_embedding)
                    #print('resultadoEmbeddingCamadas2=',resultadoEmbeddingCamadas.size())
                else:
                    if camada == EmbeddingsCamadas.SOMA_4_ULTIMAS_CAMADAS.value[0]:
                        resultado_embedding_camadas = self.getEmbeddingSoma4UltimasCamadas(sentenca_embedding)
                        #print('resultadoEmbeddingCamadas3=',resultadoEmbeddingCamadas.size())
                    else:                        
                        if camada == EmbeddingsCamadas.CONCAT_4_ULTIMAS_CAMADAS.value[0]:
                            resultado_embedding_camadas = self.getEmbeddingConcat4UltimasCamadas(sentenca_embedding)
                            #print('resultadoEmbeddingCamadas4=',resultadoEmbeddingCamadas.size())
                        else:
                            if camada == EmbeddingsCamadas.TODAS_AS_CAMADAS.value[0]:                            
                                resultado_embedding_camadas = self.getEmbeddingSomaTodasAsCamada(sentenca_embedding)
                                #print('resultadoEmbeddingCamadas5=',resultadoEmbeddingCamadas.size())
                                # Retorno: <1> x <qtde_tokens> x <768 ou 1024>
                            else:
                                logger.info("Nenhuma seleção da camada dos embeddings foi especificada.")
          
        # Verifica se a primeira dimensão é igual 1 para remover a dimensão de lote 'batches'
        # Entrada: <1> x <qtde_tokens> x <768 ou 1024>
        if resultado_embedding_camadas.shape[0] == 1:
            # Remove a dimensão 0 caso seja de tamanho 1.
            # Usa o método 'squeeze' para remover a primeira dimensão(0) pois possui tamanho 1
            # Entrada: <1> x <qtde_tokens> x <768 ou 1024>
            resultado_embedding_camadas = torch.squeeze(resultado_embedding_camadas, dim=0)     
        #print('resultadoEmbeddingCamadas2=', resultadoEmbeddingCamadas.size())    
        # Retorno: <qtde_tokens> x <768 ou 1024>
      
        # Retorna o resultados dos embeddings dos tokens da sentença  
        return resultado_embedding_camadas

    # ============================
    def getMedidasSentencasEmbeddingMEAN(self, embedding_si, embedding_sj):
        '''
        Retorna as medidas de duas sentenças Si e Sj utilizando a estratégia MEAN.
        
        Parâmetros:
        `embedding_si` - Embeddings da primeira sentença.
        `embedding_sj` - Embeddings da segunda sentença.
        
        Retorno:
        `Scos` - Similaridade do cosseno - usando a média dos embeddings Si e Sj das camadas especificadas.
        `Seuc` - Distância euclidiana - usando a média dos embeddings Si e Sj das camadas especificadas.
        `Sman` - Distância de manhattan - usando a média dos embeddings Si e Sj das camadas especificadas.
        '''

        #print('embedding_si=', embedding_si.shape) 
        #print('embedding_sj=', embedding_sj.shape)
      
        # As operações de subtração(sub), mul(multiplicação/produto), soma(sum), cosseno(similaridade), euclediana(diferença) e manhattan(diferença)
        # Necessitam que os embeddings tenha a mesmo número de dimensões.
      
        # Calcula a média dos embeddings para os tokens de Si, removendo a primeira dimensão.
        # Entrada: <qtde_tokens> x <768 ou 1024>  
        media_embedding_si = torch.mean(embedding_si, dim=0)    
        # Retorno: <768 ou 1024>
        #print('mediaCamadasSi=', mediaCamadasSi.shape)
      
        # Calcula a média dos embeddings para os tokens de Sj, removendo a primeira dimensão.
        # Entrada: <qtde_tokens> x <768 ou 1024>  
        media_embedding_sj = torch.mean(embedding_sj, dim=0)    
        # Retorno: <768 ou 1024>
        #print('mediaCamadasSj=', mediaCamadasSj.shape)
        
        # Similaridade do cosseno entre os embeddings Si e Sj
        # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
        Scos = similaridadeCoseno(media_embedding_si, media_embedding_sj)
        # Retorno: Número real

        # Distância euclidiana entre os embeddings Si e Sj
        # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
        Seuc = distanciaEuclidiana(media_embedding_si, media_embedding_sj)
        # Retorno: Número real

        # Distância de manhattan entre os embeddings Si e Sj
        # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
        Sman = distanciaManhattan(media_embedding_si, media_embedding_sj)
        # Retorno: Número real

        # Retorno das medidas das sentenças  
        return media_embedding_si, media_embedding_sj, Scos, Seuc, Sman

    # ============================
    def getMedidasSentencasEmbeddingMAX(self, embedding_si, embedding_sj):
        '''
        Retorna as medidas de duas sentenças Si e Sj utilizando a estratégia MAX.
        
        Parâmetros:
        `embedding_si` - Embeddings da primeira sentença.
        `embedding_sj` - Embeddings da segunda sentença.
           
        Retorno:
        `Scos` - Similaridade do cosseno - usando o maior dos embeddings Si e Sj das camadas especificadas.
        `Seuc` - Distância euclidiana - usando o maior dos embeddings Si e Sj das camadas especificadas.
        `Sman` - Distância de manhattan - usando o maior dos embeddings Si e Sj das camadas especificadas.
        '''

        #print('embedding_si=', embedding_si.shape) 
        #print('embedding_sj=', embedding_sj.shape)

        # As operações de subtração(sub), mul(multiplicação/produto), soma(sum), cosseno(similaridade), euclediana(diferença) e manhattan(diferença)
        # Necessitam que os embeddings tenha a mesmo número de dimensões.

        # Encontra os maiores embeddings os tokens de Si, removendo a primeira dimensão.
        # Entrada: <qtde_tokens> x <768 ou 1024>  
        maior_embedding_si, linha = torch.max(embedding_si, dim=0)    
        # Retorno: <768 ou 1024>
        #print('maior_embedding_si=', maior_embedding_si.shape)

        # Encontra os maiores embeddings os tokens de Sj, removendo a primeira dimensão.
        # Entrada: <qtde_tokens> x <768 ou 1024>  
        maior_embedding_sj, linha = torch.max(embedding_sj, dim=0)    
        # Retorno: <768 ou 1024>
        #print('maior_embedding_sj=', maior_embedding_sj.shape)

        # Similaridade do cosseno entre os embeddings Si e Sj
        # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
        Scos = similaridadeCoseno(maior_embedding_si, maior_embedding_sj)
        # Retorno: Número real

        # Distância euclidiana entre os embeddings Si e Sj
        # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
        Seuc = distanciaEuclidiana(maior_embedding_si, maior_embedding_sj)
        # Retorno: Número real

        # Distância de manhattan entre os embeddings Si e Sj
        # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
        Sman = distanciaManhattan(maior_embedding_si, maior_embedding_sj)
        # Retorno: Número real

        # Retorno das medidas das sentenças
        return maior_embedding_si, maior_embedding_sj, Scos, Seuc, Sman

    # ============================
    def getMedidasSentencasEmbedding(self, embedding_si, embedding_sj):
        '''
        Realiza o cálculo da medida do texto de acordo com a estratégia de pooling(MAX ou MEAN).
        
        Parâmetros:
        `embedding_si` - Embeddings da primeira sentença.
        `embedding_sj` - Embeddings da segunda sentença.
        `estrategia_pooling` - Estratégia de pooling a ser utilizada.       
        '''

        if self.model_args.estrategia_pooling == EstrategiasPooling.MEAN.value:
            return self.getMedidasSentencasEmbeddingMEAN(embedding_si, embedding_sj)
        else:
            if self.model_args.estrategia_pooling == EstrategiasPooling.MAX.value:
                return self.getMedidasSentencasEmbeddingMAX(embedding_si, embedding_sj)
            else:
                logger.info("Nenhuma seleção da estratégia de pooling foi especificada.")
                return None

    # ============================
    def getEmbeddingSentencaEmbeddingTextoALL(self, 
                                              embedding_texto, 
                                              texto, 
                                              sentenca):
        '''
        Retorna os embeddings de uma sentença com todas as palavras(ALL) a partir dos embeddings do texto.
        
        '''
            
        # Tokeniza o texto
        texto_tokenizado =  self.transformer.getTextoTokenizado(texto)
        #print(texto_tokenizado)

        # Tokeniza a sentença
        sentenca_tokenizada =  self.transformer.getTextoTokenizado(sentenca)
        #print(sentencaTokenizada)
        # Remove os tokens de início e fim da sentença
        sentenca_tokenizada.remove('[CLS]')
        sentenca_tokenizada.remove('[SEP]')    
        #print(len(sentenca_tokenizada))

        # Localiza os índices dos tokens da sentença no texto
        inicio, fim = encontrarIndiceSubLista(texto_tokenizado, sentenca_tokenizada)
        #print(inicio,fim) 

        # Recupera os embeddings dos tokens da sentença a partir dos embeddings do texto
        embedding_sentenca = embedding_texto[inicio:fim + 1]
        #print('embedding_sentenca=', embedding_sentenca.shape)

        # Retorna o embedding da sentença no texto
        return embedding_sentenca

    # ============================
    def getEmbeddingSentencaEmbeddingTextoCLEAN(self, 
                                                embedding_texto, 
                                                texto, 
                                                sentenca):
        '''
        Retorna os embeddings de uma sentença sem stopwords(CLEAN) a partir dos embeddings do texto.
        '''
          
        # Tokeniza o texto
        texto_tokenizado =  self.transformer.getTextoTokenizado(texto)  
        #print(textoTokenizado)

        # Remove as stopword da sentença
        sentenca_sem_stopword = self.pln.removeStopWord(sentenca)

        # Tokeniza a sentença sem stopword
        sentenca_tokenizada_sem_stopword =  self.transformer.getTextoTokenizado(sentenca_sem_stopword)
        #print(sentenca_tokenizada_sem_stopword)

        # Remove os tokens de início e fim da sentença
        sentenca_tokenizada_sem_stopword.remove('[CLS]')
        sentenca_tokenizada_sem_stopword.remove('[SEP]')    
        #print(len(sentenca_tokenizada_sem_stopword))

        # Tokeniza a sentença
        sentenca_tokenizada =  self.transformer.getTextoTokenizado(sentenca)

        # Remove os tokens de início e fim da sentença
        sentenca_tokenizada.remove('[CLS]')
        sentenca_tokenizada.remove('[SEP]')  
        #print(sentenca_tokenizada)
        #print(len(sentenca_tokenizada))

        # Localiza os índices dos tokens da sentença no texto
        inicio, fim = encontrarIndiceSubLista(texto_tokenizado, sentenca_tokenizada)
        #print('Sentença inicia em:', inicio, 'até', fim) 

        # Recupera os embeddings dos tokens da sentença a partir dos embeddings do texto
        embedding_sentenca = embedding_texto[inicio:fim + 1]
        #print('embedding_sentenca=', embedding_sentenca.shape)

        # Lista com os tensores selecionados
        lista_tokens_selecionados = []
        # Localizar os embeddings dos tokens da sentença tokenizada sem stop word na sentença 
        # Procura somente no intervalo da sentença
        for i, token_sentenca in enumerate(sentenca_tokenizada):
            for token_sentenca_sem_stopword in sentenca_tokenizada_sem_stopword: 
                # Se o token da sentença é igual ao token da sentença sem stopword    
                if token_sentenca == token_sentenca_sem_stopword:
                    lista_tokens_selecionados.append(embedding_sentenca[i:i + 1])

        embedding_sentenca_sem_stopword = None

        if len(lista_tokens_selecionados) != 0:
            # Concatena os vetores da lista pela dimensão 0
            embedding_sentenca_sem_stopword = torch.cat(lista_tokens_selecionados, dim=0)
            #print("embedding_sentenca_sem_stopword:",embedding_sentenca_sem_stopword.shape)

        # Retorna o embedding da sentença no texto
        return embedding_sentenca_sem_stopword

    # ============================
    def getEmbeddingSentencaEmbeddingTextoNOUN(self, 
                                               embedding_texto, 
                                               texto, 
                                               sentenca):
        '''
        Retorna os embeddings de uma sentença somente com as palavras relevantes(NOUN) de um tipo a partir dos embeddings do texto.
        '''

        # Tokeniza o texto
        texto_tokenizado =  self.transformer.getTextoTokenizado(texto)  
        #print(textoTokenizado)

        # Retorna as palavras relevantes da sentença do tipo especificado
        sentenca_somente_relevante = self.pln.retornaPalavraRelevante(sentenca, self.model_args.palavra_relevante)

        # Tokeniza a sentença 
        sentenca_tokenizada_somente_relevante =  self.transformer.getTextoTokenizado(sentenca_somente_relevante)

        # Remove os tokens de início e fim da sentença
        sentenca_tokenizada_somente_relevante.remove('[CLS]')
        sentenca_tokenizada_somente_relevante.remove('[SEP]')  
        #print(sentenca_tokenizada_somente_relevante)
        #print(len(sentenca_tokenizada_somente_relevante))

        # Tokeniza a sentença
        sentenca_tokenizada =  self.transformer.getTextoTokenizado(sentenca)

        # Remove os tokens de início e fim da sentença
        sentenca_tokenizada.remove('[CLS]')
        sentenca_tokenizada.remove('[SEP]')  
        #print(sentenca_tokenizada)
        #print(len(sentenca_tokenizada))

        # Localiza os índices dos tokens da sentença no texto
        inicio, fim = encontrarIndiceSubLista(texto_tokenizado, sentenca_tokenizada)
        #print('Sentença inicia em:', inicio, 'até', fim) 

        # Recupera os embeddings dos tokens da sentença a partir dos embeddings do texto
        embedding_sentenca = embedding_texto[inicio:fim + 1]
        #print('embedding_sentenca=', embedding_sentenca.shape)

        # Lista com os tensores selecionados
        listaTokensSelecionados = []
        # Localizar os embeddings dos tokens da sentença tokenizada sem stop word na sentença 
        # Procura somente no intervalo da sentença
        for i, token_sentenca in enumerate(sentenca_tokenizada):
            for token_sentenca_somente_relevante in sentenca_tokenizada_somente_relevante: 
                if token_sentenca == token_sentenca_somente_relevante:        
                    listaTokensSelecionados.append(embedding_sentenca[i:i + 1])

        embedding_sentenca_com_substantivo = None

        if len(listaTokensSelecionados) != 0:
            # Concatena os vetores da lista pela dimensão 0  
            embedding_sentenca_com_substantivo = torch.cat(listaTokensSelecionados, dim=0)
            #print("embedding_sentenca_com_substantivo:",embedding_sentenca_com_substantivo.shape)

        # Retorna o embedding da sentença do texto
        return embedding_sentenca_com_substantivo

    # ============================
    def getEmbeddingSentencaEmbeddingTexto(self, 
                                           embedding_texto, 
                                           texto, 
                                           sentenca):
        '''
        Retorna os embeddings de uma sentença considerando a relevância das palavras (ALL, CLEAN ou NOUN) a partir dos embeddings do texto.    
        '''

        if self.model_args.palavra_relevante == PalavrasRelevantes.ALL.value:
            return self.getEmbeddingSentencaEmbeddingTextoALL(embedding_texto, texto, sentenca)
        else:
            if self.model_args.palavra_relevante == PalavrasRelevantes.CLEAN.value:                
                return self.getEmbeddingSentencaEmbeddingTextoCLEAN(embedding_texto, texto, sentenca)
            else:
                if self.model_args.palavra_relevante == PalavrasRelevantes.NOUN.value:
                    return self.getEmbeddingSentencaEmbeddingTextoNOUN(embedding_texto, texto, sentenca)
                else:
                    logger.info("Nenhuma estratégia de relevância de palavras foi especificada.") 

    # ============================
    def getMedidasComparacaoTexto(self, 
                                  texto, 
                                  camada, 
                                  tipo_texto='p'):
        '''
        Retorna as medidas do texto.
        Considera somente sentenças com pelo menos uma palavra.
        Estratégia de pooling padrão é MEAN(0).
        Palavra relavante padrão é ALL(0).
        
        Retorno um dicionário com:
        `cos` - Medida de cos do do texto.
        `euc` - Medida de euc do do texto.
        `man` - Medida de man do do texto.
        '''

        # Quantidade de sentenças no texto
        n = len(texto)
        
        # Divisor da quantidade de textos
        divisor = n - 1

        # Texto é uma lista com as sentenças
        #print('camada=',camada)
        #print('Texto=', texto)

        # Junta a lista de sentenças em um texto(string)
        string_texto = ' '.join(texto)

        # Envia o texto ao MCL e recupera os embeddings de todas as camadas
        # Se for o texto original pega do buffer para evitar a repetição
        if tipo_texto == 'o':
            # Retorna os embeddings de todas as camadas do texto
            # O embedding possui os seguintes valores        
            # 0-texto_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
            total_camadas_texto = self.getEmbeddingsTodasCamadasBuffer(string_texto)      
            # Retorno: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> <768 ou 1024>) 
        else:
            # Retorna os embeddings de todas as camadas do texto
            # O embedding possui os seguintes valores        
            # 0-texto_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
            total_camadas_texto = self.getEmbeddingsTodasCamadas(string_texto)      
            # Retorno: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> <768 ou 1024>) 

        # Recupera os embeddings dos tokens das camadas especificadas de acordo com a estratégia especificada para camada  
        embedding_texto = self.getResultadoEmbeddings(total_camadas_texto, camada=camada)
        #print('embedding_texto=', embedding_texto.shape)

        # Acumuladores das medidas entre as sentenças  
        somaScos = 0
        somaSeuc = 0
        somaSman = 0

        # Seleciona os pares de sentença a serem avaliados
        posSi = 0
        posSj = posSi + 1

        #Enquanto o indíce da sentneça posSj(2a sentença) não chegou ao final da quantidade de sentenças
        while posSj <= (n-1):  

            # Seleciona as sentenças do texto  
            Si = texto[posSi]
            Sj = texto[posSj]

            # Recupera os embedding das sentenças Si e Sj do embedding do texto      
            embedding_si = self.getEmbeddingSentencaEmbeddingTexto(embedding_texto, string_texto, Si)
            embedding_sj = self.getEmbeddingSentencaEmbeddingTexto(embedding_texto, string_texto, Sj)

            # Verifica se os embeddings sentenças estão preenchidos
            if embedding_si != None and embedding_sj != None:

                # Recupera as medidas entre Si e Sj     
                ajustado_embedding_si, ajustado_embedding_sj, Scos, Seuc, Sman = self.getMedidasSentencasEmbedding(embedding_si, embedding_sj)

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
                if embedding_si == None:
                    # Avança a posição da sentença posSi para a posSj
                    posSi = posSj
                    # Avança para a próxima sentença de posSj
                    posSj = posSj + 1        
                else:          
                    # Se embeddingSj = None avança somente posJ para a próxima sentença
                    if embedding_sj == None:
                        posSj = posSj + 1

        # Calcula a medida 
        Ccos = 0
        Ceuc = 0
        Cman = 0

        if divisor != 0:
            Ccos = float(somaScos) / float(divisor)
            Ceuc = float(somaSeuc) / float(divisor)
            Cman = float(somaSman) / float(divisor)

        # Retorna as medidas em um dicionário
        saida = {}
        saida.update({'cos' : Ccos,
                      'euc' : Ceuc,
                      'man' : Cman})

        return saida   