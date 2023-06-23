# Import das bibliotecas.

# Biblioteca de logging
import logging  
# Biblioteca de tipos
from typing import Union

# Bibliotecas próprias
from textotransformer.modelo.modeloenum import AbordagemExtracaoEmbeddingsCamadas, EstrategiasPooling 
from textotransformer.mensurador.mensuradorenum import PalavraRelevante
from textotransformer.util.utilconstantes import LISTATIPOCAMADA_NOME
 
logger = logging.getLogger(__name__)

# ============================
def getIntParaPalavraRelevante(palavra_relevante: Union[int, PalavraRelevante]):
    '''
    Converte um inteiro para um objeto da classe PalavraRelevante.
    
    Parâmetros:        
    `palavra_relevante` - Um valor inteiro a ser convertido.

    Retorno:
    Um objeto da classe PalavraRelevante.
    '''
        
    # Verifica o tipo de dado do parâmetro 'palavra_relevante'
    if isinstance(palavra_relevante, int):
        if palavra_relevante == 0:
            palavra_relevante = PalavraRelevante.ALL
        else:
            if palavra_relevante == 1:
                palavra_relevante = PalavraRelevante.CLEAN
            else:
                if palavra_relevante == 2:
                    palavra_relevante = PalavraRelevante.NOUN
                else:
                    palavra_relevante = None
                    logger.info("Não foi especificado um valor inteiro para a estratégia de relevância de palavra.") 
    
    return palavra_relevante

# ============================
def getIntParaEstrategiasPooling(estrategia_pooling: Union[int, EstrategiasPooling]):
    '''
    Converte um inteiro para um objeto da classe EstrategiasPooling.
    
    Parâmetros:        
    `estrategia_pooling` - Um valor inteiro a ser convertido.

    Retorno:
    Um objeto da classe EstrategiasPooling.
    '''
    
    # Verifica o tipo de dado do parâmetro 'estrategia_pooling'
    if isinstance(estrategia_pooling, int):
        if estrategia_pooling == 0:
            estrategia_pooling = EstrategiasPooling.MEAN
        else:
            if estrategia_pooling == 1:
                estrategia_pooling = EstrategiasPooling.MAX
            else:
                estrategia_pooling = None
                logger.info("Não foi especificado um valor inteiro para a estratégia de pooling.") 
                
    return estrategia_pooling                

# ============================
def getIntParaAbordagemExtracaoEmbeddingsCamadas(abordagem_extracao_embeddings_camadas: Union[int, AbordagemExtracaoEmbeddingsCamadas]):
    '''
    Converte um inteiro para um objeto da classe AbordagemExtracaoEmbeddingsCamadas.
    
    Parâmetros:        
    `abordagem_extracao_embeddings_camadas` - Um valor inteiro de onde deve ser recupera os embeddings.

    Retorno:
    Um objeto da classe AbordagemExtracaoEmbeddingsCamadas.
    '''
    
    # Verifica o tipo de dado do parâmetro 'abordagem_extracao_embeddings_camadas'
    if isinstance(abordagem_extracao_embeddings_camadas, int):
        if abordagem_extracao_embeddings_camadas == 0:
            abordagem_extracao_embeddings_camadas = AbordagemExtracaoEmbeddingsCamadas.PRIMEIRA_CAMADA
        else:
            if abordagem_extracao_embeddings_camadas == 1:
                abordagem_extracao_embeddings_camadas = AbordagemExtracaoEmbeddingsCamadas.PENULTIMA_CAMADA
            else:
                if abordagem_extracao_embeddings_camadas == 2:
                    abordagem_extracao_embeddings_camadas = AbordagemExtracaoEmbeddingsCamadas.ULTIMA_CAMADA
                else:
                    if abordagem_extracao_embeddings_camadas == 3:
                        abordagem_extracao_embeddings_camadas = AbordagemExtracaoEmbeddingsCamadas.SOMA_4_ULTIMAS_CAMADAS
                    else:
                        if abordagem_extracao_embeddings_camadas == 4:
                            abordagem_extracao_embeddings_camadas = AbordagemExtracaoEmbeddingsCamadas.CONCAT_4_ULTIMAS_CAMADAS
                        else:
                            if abordagem_extracao_embeddings_camadas == 5:
                                abordagem_extracao_embeddings_camadas = AbordagemExtracaoEmbeddingsCamadas.TODAS_AS_CAMADAS
                            else:
                                abordagem_extracao_embeddings_camadas = None
                                logger.info("Não foi especificado um valor inteiro para o tipo de embedding camada.") 
    
    return abordagem_extracao_embeddings_camadas

# ============================
def getAbordagemExtracaoEmbeddingsCamadasStr(abordagem_extracao_embeddings_camadas: Union[int, AbordagemExtracaoEmbeddingsCamadas]):
    '''
    Retorna uma string com o nome da abordagem de extração de embeddings das camadas.
    
    Parâmetros:        
    `abordagem_extracao_embeddings_camadas` - Um valor inteiro de onde deve ser recupera os embeddings.

    Retorno:
    Uma string com o nome da abordagem.
    '''
    
    # Verifica o tipo de dado do parâmetro 'abordagem_extracao_embeddings_camadas'
    abordagem_extracao_embeddings_camadas = getIntParaAbordagemExtracaoEmbeddingsCamadas(abordagem_extracao_embeddings_camadas)
    
    return abordagem_extracao_embeddings_camadas.value[LISTATIPOCAMADA_NOME]