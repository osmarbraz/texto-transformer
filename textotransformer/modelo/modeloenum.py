# Import das bibliotecas.

# Biblioteca de logging
import logging  

# Biblioteca de Enum
from enum import Enum 

logger = logging.getLogger(__name__)

# ============================
class EstrategiasPooling(Enum):
    MEAN = 0 # Média / Use a média em cada dimensão sobre todos os tokens.
    MAX = 1 # Máximo / Use o máximo em cada dimensão sobre todos os tokens.

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