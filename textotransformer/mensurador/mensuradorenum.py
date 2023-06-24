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
    @classmethod
    def converteInt(self, medidas_comparacao: int):
        '''
        Converte um inteiro para um objeto da classe MedidasComparacao.
        
        Parâmetros:        
           `medidas_comparacao` - Um valor inteiro a ser convertido.

        Retorno:
           Um objeto da classe MedidasComparacao.
        '''
            
         # Verifica o tipo de dado do parâmetro 'medidas_comparacao'
        if isinstance(medidas_comparacao, int):
            if medidas_comparacao == 0:
                medidas_comparacao = self.COSSENO
            else:
                if medidas_comparacao == 1:
                    medidas_comparacao = self.EUCLIDIANA
                else:
                    if medidas_comparacao == 2:
                        medidas_comparacao = self.MANHATTAN
                    else:
                        medidas_comparacao = None
                        logger.error("Não foi especificado um valor inteiro válido para a medida de comparação.") 
        else:
            logger.error("Não foi especificado um valor inteiro para a a medida de comparação.") 
            return None                        
        
        return medidas_comparacao

# ============================
class PalavraRelevante(Enum):
    ALL = 0 # Todas as palavras
    CLEAN = 1 # Sem stopwords
    NOUN = 2 # Somente substantivos
    
    # ============================
    @classmethod
    def converteInt(self, palavra_relevante: int):
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
                palavra_relevante = self.ALL
            else:
                if palavra_relevante == 1:
                    palavra_relevante = self.CLEAN
                else:
                    if palavra_relevante == 2:
                        palavra_relevante = self.NOUN
                    else:
                        palavra_relevante = None
                        logger.error("Não foi especificado um valor inteiro válido para a estratégia de relevância de palavra.") 
        else:
            logger.error("Não foi especificado um valor inteiro para a estratégia de relevância de palavra.") 
            return None                        
        
        return palavra_relevante