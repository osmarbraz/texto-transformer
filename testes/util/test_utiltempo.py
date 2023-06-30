# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de testes unitários
import unittest

# Bibliotecas próprias
from textotransformer.util.utiltempo import formataTempo, mediaTempo, somaTempo

logger = logging.getLogger(__name__)

class TestUtilTempo(unittest.TestCase):
        
    # Testes formataTempo
    def test_formataTempo(self):
        logger.info("Testando o formataTempo")
        
        tempo = 100000000
        saidaEsperada = "1157 days, 9:46:40"
        
        saida = formataTempo(tempo)        
                
        self.assertEqual(saida, saidaEsperada)
        
    # Testes mediaTempo
    def test_mediaTempo(self):
        logger.info("Testando o mediaTempo")
                
        listaTempo = ["01:00:00","03:00:00"]
        saidaEsperada = "01:59:59"
        
        saida = mediaTempo(listaTempo)                
                
        self.assertEqual(saida, saidaEsperada)        
        
    # Testes somaTempo
    def test_somaTempo(self):
        logger.info("Testando o somaTempo")
        
        listaTempo = ["01:00:00","03:00:00"]
        saidaEsperada = "04:00:00"
        
        saida = somaTempo(listaTempo)        
                
        self.assertEqual(saida, saidaEsperada)    

if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.info("Teste Util Tempo")
    unittest.main()
    