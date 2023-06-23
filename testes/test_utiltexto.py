# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de testes unitários
import unittest

from textotransformer.util.utiltexto import getTextoLista, limpezaTexto, removeTags

# Bibliotecas próprias
logger = logging.getLogger(__name__)

class TestUtilTexto(unittest.TestCase):
        
    # Testes limpeza
    def test_limpeza(self):
        logger.info("Testando o limpeza")
        
        texto = "   Qual o \nsabor   do  \n  sorvete   ??????????   "
        
        saida = limpezaTexto(texto)
        saidaEsperada = "Qual o sabor do sorvete ?"
                
        self.assertEqual(saida, saidaEsperada)
        
    # Testes getTextoLista
    def test_getTextoLista(self):
        logger.info("Testando o getTextoLista")
        
        texto = ['um','dois']
        
        saida = getTextoLista(texto)
        saidaEsperada = "umdois"
                
        self.assertEqual(saida, saidaEsperada)        
        
    # Testes removeTags
    def test_removeTags(self):
        logger.info("Testando o removeTags")
        
        texto = '<html><body>texto</body></html>'
        
        saida = removeTags(texto)
        saidaEsperada = "texto"
                
        self.assertEqual(saida, saidaEsperada)        

if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.info("Teste Util Texto")
    unittest.main()
    