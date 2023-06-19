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
# EmbeddingsCamadas
# Define um enum com as camadas a serem analisadas nos teste.
# Cada elemento do enum 'EmbeddingsCamadas' é chamado de camada sendo formado por:
#  - camada.value[0] = Índice da camada
#  - camada.value[1] = Um inteiro com o índice da camada a ser avaliada. Pode conter valores negativos.
#  - camada.value[2] = Operação para n camadas, CONCAT ou SUM.
#  - camada.value[3] = Nome do tipo camada

# ============================
class EmbeddingsCamadas(Enum):
    
    # BERT Large possui 25 camadas(1a camada com os tokens de entrada e 24 camadas dos transformers)
    # BERT Large possui 13 camadas(1a camada com os tokens de entrada e 24 camadas dos transformers)
    # O índice da camada com valor positivo indica uma camada específica
    # O índice com um valor negativo indica as camadas da posição com base no fim descontado o valor indice até o fim.

    PRIMEIRA_CAMADA = [0, 1, '-', 'Primeira']
    PENULTIMA_CAMADA = [1, -2, '-', 'Penúltima']
    ULTIMA_CAMADA = [2, -1, '-', 'Última']
    SOMA_4_ULTIMAS_CAMADAS =[3, -4, 'SUM', 'Soma 4 últimas']
    CONCAT_4_ULTIMAS_CAMADAS = [4, -4, 'CONCAT', 'Concat 4 últimas']
    TODAS_AS_CAMADAS = [5, 24, 'SUM', 'Todas']
    
# ============================
# listaTipoCamadas
# Define uma lista com as camadas a serem analisadas nos teste.
# Cada elemento da lista 'listaTipoCamadas' é chamado de camada sendo formado por:
#  - camada[0] = Índice da camada
#  - camada[1] = Um inteiro com o índice da camada a ser avaliada. Pode conter valores negativos.
#  - camada[2] = Operação para n camadas, CONCAT ou SUM.
#  - camada[3] = Nome do tipo camada

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