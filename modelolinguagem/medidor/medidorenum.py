# Import das bibliotecas.
import logging  # Biblioteca de logging
from enum import Enum # Biblioteca de Enum

class MedidasCoerencia(Enum):
    COSSENO = 0 # Similaridade do Cosseno
    EUCLIDIANA = 1 # Distância Euclidiana
    MANHATTAN = 2 # Distância de Manhattan
    

class EstrategiasPooling(Enum):
    MEAN = 0 # Média
    MAX = 1 # Máximo


class PalavrasRelevantes(Enum):
    ALL = 0 # Todas as palavras
    CLEAN = 1 # Sem stopwords
    NOUN = 2 # Somente substantivos


# Índice dos campos do enum EmbeddingsCamadasBERT
LISTATIPOCAMADA_ID = 0
LISTATIPOCAMADA_CAMADA = 1
LISTATIPOCAMADA_OPERACAO = 2
LISTATIPOCAMADA_NOME = 3

# ============================
# EmbeddingsCamadasBERT
# Define um enum com as camadas a serem analisadas nos teste.
# Cada elemento do enum 'EmbeddingsCamadasBERT' é chamado de camada sendo formado por:
#  - camada.value[0] = Índice da camada
#  - camada.value[1] = Um inteiro com o índice da camada a ser avaliada. Pode conter valores negativos.
#  - camada.value[2] = Operação para n camadas, CONCAT ou SUM.
#  - camada.value[3] = Nome do tipo camada

class EmbeddingsCamadasBERT(Enum):
    
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
    
    
