# Import das bibliotecas.

# Biblioteca de logging
import logging  

# Biblioteca de Enum
from enum import Enum 

logger = logging.getLogger(__name__)

# ============================
class MedidasComparacao(Enum):
    COSSENO = 0 # Similaridade do Cosseno
    EUCLIDIANA = 1 # Distância Euclidiana
    MANHATTAN = 2 # Distância de Manhattan

# ============================
class PalavrasRelevantes(Enum):
    ALL = 0 # Todas as palavras
    CLEAN = 1 # Sem stopwords
    NOUN = 2 # Somente substantivos