# Import das bibliotecas.

# Biblioteca de logging
import logging  

# Biblioteca de Enum
from enum import Enum 

# Biblitecas próprias
from textotransformer.util.utilconstantes import LISTATIPOCAMADA_NOME

logger = logging.getLogger(__name__)

# ============================
class GranularidadeTexto(Enum):
    TOKEN = 0 # Token / Token do tokenizador do MCL.
    PALAVRA = 1 # Palavra / Palavra do texto.
    SENTENCA = 2 # Sentença / Sentença do texto.
    TEXTO = 3 # Texto / Texto completo.
    
    # ============================
    @classmethod
    def converteInt(self, granularidade_texto: int):
        '''
        Converte um inteiro para um objeto da classe GranularidadeTexto.
        
        Parâmetros:        
           `granularidade_texto` - Um valor inteiro a ser convertido.

        Retorno:
           Um objeto da classe GranularidadeTexto.
        '''

        # Verifica o tipo de dado do parâmetro 'granularidade_texto'
        if isinstance(granularidade_texto, int):
            if granularidade_texto == 0:
                granularidade_texto = self.TOKEN
            else:
                if granularidade_texto == 1:
                    granularidade_texto = self.PALAVRA
                else:
                    if granularidade_texto == 2:
                        granularidade_texto = self.SENTENCA
                    else:
                        if granularidade_texto == 3:
                            granularidade_texto = self.TEXTO
                        else:
                            granularidade_texto = None
                            logger.error("Não foi especificado um valor inteiro válido para a granularidade texto.") 
        else:
            logger.error("Não foi especificado um valor inteiro para a granularidade texto.") 
            return None                    
                    
        return granularidade_texto   

# ============================
class EstrategiasPooling(Enum):
    MEAN = 0 # Média / Use a média em cada dimensão sobre todos os tokens.
    MAX = 1 # Máximo / Use o máximo em cada dimensão sobre todos os tokens.
    
    # ============================
    @classmethod
    def converteInt(self, estrategia_pooling: int):
        '''
        Converte um inteiro para um objeto da classe EstrategiasPooling.
        
        Parâmetros:        
           `estrategia_pooling` - Um valor inteiro a ser convertido.

        Retorno:
           Um objeto da classe EstrategiasPooling.
        '''

        # Verifica o tipo de dado do parâmetro 'abordagem_extracao_embeddings_camadas'
        if isinstance(estrategia_pooling, int):
            if estrategia_pooling == 0:
                estrategia_pooling = self.MEAN
            else:
                if estrategia_pooling == 1:
                    estrategia_pooling = self.MAX
                else:
                    estrategia_pooling = None
                    logger.error("Não foi especificado um valor inteiro válido para a estratégia de pooling.") 
        else:
            logger.error("Não foi especificado um valor inteiro para a estratégia de pooling.") 
            return None                    
                    
        return estrategia_pooling    

# ============================
# AbordagemExtracaoEmbeddingsCamadas
# Define um enum com as abordagems de extração a serem analisadas nos teste.
# Cada elemento do enum 'AbordagemExtracaoEmbeddingsCamadas' é chamado de abordagem_extracao sendo formado por:
#  - abordagem_extracao.value[0] = Índice da abordagem_extracao.
#  - abordagem_extracao.value[1] = Um inteiro com o índice da camada a ser avaliada. Pode conter valores negativos.
#  - abordagem_extracao.value[2] = Operação para n camadas, CONCAT ou SUM.
#  - abordagem_extracao.value[3] = Nome da abordagem de extração.

# ============================
# Abordagem para extrair os embeddings das camadas do BERT
class AbordagemExtracaoEmbeddingsCamadas(Enum):
    
    # BERT Large possui 25 camadas(1a camada com os tokens de entrada e 24 camadas dos transformers)
    # BERT Large possui 13 camadas(1a camada com os tokens de entrada e 24 camadas dos transformers)
    # O índice da camada com valor positivo indica uma camada específica
    # O índice com um valor negativo indica as camadas da posição com base no fim descontado o valor indice até o fim.

    PRIMEIRA_CAMADA = [0, 1, '-', 'Primeira']
    PENULTIMA_CAMADA = [1, -2, '-', 'Penúltima']
    ULTIMA_CAMADA = [2, -1, '-', 'Última']
    SOMA_4_ULTIMAS_CAMADAS =[3, -4, 'SUM', 'Soma 4 últimas']
    CONCAT_4_ULTIMAS_CAMADAS = [4, -4, 'CONCAT', 'Concat 4 últimas']
    TODAS_AS_CAMADAS = [5, 24, 'SUM', 'Soma todas']
    
    # ============================
    @classmethod
    def converteInt(self, abordagem_extracao_embeddings_camadas: int):
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
                abordagem_extracao_embeddings_camadas = self.PRIMEIRA_CAMADA
            else:
                if abordagem_extracao_embeddings_camadas == 1:
                    abordagem_extracao_embeddings_camadas = self.PENULTIMA_CAMADA
                else:
                    if abordagem_extracao_embeddings_camadas == 2:
                        abordagem_extracao_embeddings_camadas = self.ULTIMA_CAMADA
                    else:
                        if abordagem_extracao_embeddings_camadas == 3:
                            abordagem_extracao_embeddings_camadas = self.SOMA_4_ULTIMAS_CAMADAS
                        else:
                            if abordagem_extracao_embeddings_camadas == 4:
                                abordagem_extracao_embeddings_camadas = self.CONCAT_4_ULTIMAS_CAMADAS
                            else:
                                if abordagem_extracao_embeddings_camadas == 5:
                                    abordagem_extracao_embeddings_camadas = self.TODAS_AS_CAMADAS
                                else:
                                    abordagem_extracao_embeddings_camadas = None
                                    logger.error("Não foi especificado um valor inteiro válido para o tipo de embedding camada.") 
        else:
            logger.error("Não foi especificado um valor inteiro para o tipo de embedding camada.") 
            return None
        
        return abordagem_extracao_embeddings_camadas

    # ============================    
    def getStr(self):
        '''
        Retorna uma string com o nome da abordagem de extração de embeddings das camadas.
        
        Retorno:
           Uma string com o nome da abordagem.
        '''
        
        return self.value[LISTATIPOCAMADA_NOME]
