# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de testes unitários
import unittest

# Bibliotecas próprias
from textotransformer.util.utiltexto import getTextoLista, limpezaTexto, removeTags, tamanhoTexto

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
        
    # Testes tamanhoTexto
    def test_tamanhoTexto(self):
        logger.info("Testando o tamanhoTexto")
        
        texto1 = ""
        texto2 = "manga"
        texto3 = []
        texto4 = [["manga","banana"]]
        texto5 = [["manga","banana"],["uva","laranja"]]
        texto6 = {'lista1' :[["manga","banana"]]}        
        texto7 = [{'lista1' :[["manga","banana"]], 
                   'lista2' : [["uva","laranja"]]}]
               
        self.assertEqual(tamanhoTexto(texto1), 0)
        self.assertEqual(tamanhoTexto(texto2), 5)
        self.assertEqual(tamanhoTexto(texto3), 0)
        self.assertEqual(tamanhoTexto(texto4), 2)
        self.assertEqual(tamanhoTexto(texto5), 4)
        self.assertEqual(tamanhoTexto(texto6), 1)
        self.assertEqual(tamanhoTexto(texto7), 2)

if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.info("Teste Util Texto")
    unittest.main()
    