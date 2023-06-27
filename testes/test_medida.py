# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de testes unitários
import unittest

import numpy

# Bibliotecas próprias
from textotransformer.mensurador.medidas import distanciaEuclidiana, distanciaManhattan, similaridadeCosseno, similaridadeCossenoMatriz

logger = logging.getLogger(__name__)

class TestMedida(unittest.TestCase):

    # Testes similaridadeCosenoMatriz_1D
    def test_similaridadeCosenoMatriz_1D(self):
        logger.info("Testando o similaridadeCosenoMatriz_1D")
        
        # Valores de entrada
        a = [1.0, 2.0, 3.0, 4.0]
        b = [1.0, 2.0, 3.0, 3.0]
                    
        # Valores de saída                        
        saida = similaridadeCossenoMatriz(a, b)
                
        # Saída esperada
        saidaEsperada = 0.9898030757904053
        
        self.assertEqual(saidaEsperada, saida)
        
    # Testes similaridadeCosenoMatriz_2D
    def test_similaridadeCosenoMatriz_2D(self):
        logger.info("Testando o similaridadeCosenoMatriz_2D")
        
        # Valores de entrada
        a = [[1.0, 2.0, 3.0, 4.0],[1.0, 2.0 ,5.0, 6.0]]
        b = [[1.0, 2.0, 3.0, 4.0],[1.0, 2.0, 5.0, 6.0]]
                    
        # Valores de saída                        
        saida = similaridadeCossenoMatriz(a, b)
        
        # Arredonda para 5 casas decimais
        saida = numpy.round(saida, decimals=5)
        
        # Saída esperada
        saidaEsperada =  [[1.0, 0.98882646], [0.98882646, 1.0]]
        
        # Arredonda para 5 casas decimais
        saidaEsperada = numpy.round(saidaEsperada, decimals=5)
        
        # Converte todos os valores para float32
        saidaEsperada = saidaEsperada.astype(numpy.float32)
                
        self.assertTrue((saida == saidaEsperada).all())
                
    # Testes similaridadeCoseno
    def test_similaridadeCoseno(self):
        logger.info("Testando o similaridadeCoseno(a, b)")
        
        # Valores de entrada
        a = [1, 2, 3, 4]
        b = [1, 2, 3, 3]

        # Valores de saída
        saida = similaridadeCosseno(a, b)
        
        # Saída esperada
        saidaEsperada = 0.9898030839149452
        
        self.assertEqual(saidaEsperada, saida)

    # Testes distanciaEuclidiana
    def test_distanciaEuclidiana(self):
        logger.info("Testando o distanciaEuclidiana")
        
        # Valores de entrada
        a = [1, 2, 3, 4]
        b = [1, 2, 3, 6]

        # Valores de saída
        saida = distanciaEuclidiana(a, b)
        
        # Saída esperada
        saidaEsperada = 2.
                
        self.assertEqual(saidaEsperada, saida)        

    # Testes distanciaManhattan
    def test_distanciaManhattan(self):
        logger.info("Testando o distanciaManhattan")
        
        # Valores de entrada
        a = [1, 2, 3, 4]
        b = [1, 2, 3, 6]

        # Valores de saída        
        saida = distanciaManhattan(a, b)
        
        # Saída esperada
        saidaEsperada = 2
        
        self.assertEqual(saidaEsperada, saida)             
   
   
if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.info("Teste Medidas")
    unittest.main()
    