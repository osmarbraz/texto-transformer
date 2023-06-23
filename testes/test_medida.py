# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de testes unitários
import unittest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Bibliotecas próprias
from textotransformer.mensurador.medidas import PytorchSimilaridadeCoseno

logger = logging.getLogger(__name__)

class TestMedida(unittest.TestCase):
        
    # Testes PytorchSimilaridadeCoseno
    def test_PytorchSimilaridadeCoseno(self):
        logger.info("Testando o PytorchSimilaridadeCoseno")
        
        # Valores de entrada
        a = np.random.randn(50, 100)
        b = np.random.randn(50, 100)

        # Valores de saída
        sklearn_pairwise = cosine_similarity(a, b)        
        pytorch_cos_scores = PytorchSimilaridadeCoseno(a, b).numpy()
        
        for i in range(len(sklearn_pairwise)):
            for j in range(len(sklearn_pairwise[i])):
                assert abs(sklearn_pairwise[i][j] - pytorch_cos_scores[i][j]) < 0.001                     
   
if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.info("Teste Medidas")
    unittest.main()
    