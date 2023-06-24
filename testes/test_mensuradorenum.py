# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de testes unitários
import unittest

# Bibliotecas próprias
from textotransformer.mensurador.mensuradorenum import PalavraRelevante, MedidasComparacao

logger = logging.getLogger(__name__)

class TestMensuradorEnum(unittest.TestCase):

    # Testes MedidasComparacao_converteInt
    def test_MedidasComparacao_converteInt(self):
        logger.info("Testando o MedidasComparacao_converteInt")
        
        # Testa a conversão de int para MedidasComparacao
        self.assertEqual(MedidasComparacao.converteInt(0), MedidasComparacao.COSSENO)
        self.assertEqual(MedidasComparacao.converteInt(1), MedidasComparacao.EUCLIDIANA)
        self.assertEqual(MedidasComparacao.converteInt(2), MedidasComparacao.MANHATTAN)
        
    # Testes PalavraRelevante_converteInt
    def test_PalavraRelevante_converteInt(self):
        logger.info("Testando o PalavraRelevante_converteInt")
        
        # Testa a conversão de int para PalavraRelevante
        self.assertEqual(PalavraRelevante.converteInt(0), PalavraRelevante.ALL)
        self.assertEqual(PalavraRelevante.converteInt(1), PalavraRelevante.CLEAN)
        self.assertEqual(PalavraRelevante.converteInt(2), PalavraRelevante.NOUN)

    
if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.info("Teste Util Conversão")
    unittest.main()
    