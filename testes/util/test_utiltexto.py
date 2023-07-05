# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de testes unitários
import unittest

# Bibliotecas próprias
from textotransformer.util.utiltexto import getIndexTokenTexto, getTextoLista, limpezaTexto, removeTags, tamanhoTexto

# Objeto de logger
logger = logging.getLogger(__name__)

class TestUtilTexto(unittest.TestCase):
        
    # Testes limpeza
    def test_limpeza(self):
        logger.info("Testando o limpeza")
        
        # Valores de entrada
        texto = "   Qual o \nsabor   do  \n  sorvete   ??????????   "
        
        # Valores de saída
        saida = limpezaTexto(texto)
        
        # Valores esperados
        saidaEsperada = "Qual o sabor do sorvete ?"
                
        self.assertEqual(saida, saidaEsperada)
        
    # Testes getTextoLista
    def test_getTextoLista(self):
        logger.info("Testando o getTextoLista")
        
        # Valores de entrada
        texto = ['um','dois']
        
        # Valores de saída
        saida = getTextoLista(texto)
        
        # Valores esperados
        saidaEsperada = "umdois"
                
        self.assertEqual(saida, saidaEsperada)        
        
    # Testes removeTags
    def test_removeTags(self):
        logger.info("Testando o removeTags")
        
        # Valores de entrada
        texto = '<html><body>texto</body></html>'
        
        # Valores de saída
        saida = removeTags(texto)
        
        # Valores esperados
        saidaEsperada = "texto"
                
        self.assertEqual(saida, saidaEsperada)
        
    # Testes tamanhoTexto
    def test_tamanhoTexto(self):
        logger.info("Testando o tamanhoTexto")
        
        # Valores de entrada
        texto1 = ""
        texto2 = "manga"
        texto3 = []
        texto4 = [["manga","banana"]]
        texto5 = [["manga","banana"],["uva","laranja"]]
        texto6 = {'lista1' :[["manga","banana"]]}        
        texto7 = [{'lista1' :[["manga","banana"]], 
                   'lista2' : [["uva","laranja"]]}]

        # Avalia a saida do método
        self.assertEqual(tamanhoTexto(texto1), 0)
        self.assertEqual(tamanhoTexto(texto2), 5)
        self.assertEqual(tamanhoTexto(texto3), 0)
        self.assertEqual(tamanhoTexto(texto4), 2)
        self.assertEqual(tamanhoTexto(texto5), 4)
        self.assertEqual(tamanhoTexto(texto6), 1)
        self.assertEqual(tamanhoTexto(texto7), 2)

    # Testes getIndexTokenTexto
    def test_tamanhoTexto(self):
        logger.info("Testando o getIndexTokenTexto")
        
        # Valores de entrada
        lista_tokens = ['Depois', 'de', 'roubar', 'o', 'co', '##fre', 'do', 'banco', ',', 'o', 'lad', '##rão', 'de', 'banco', 'foi', 'visto', 'sentado', 'no', 'banco', 'da', 'praça', 'central', '.']
        token = "banco"
        # O token "banco" se encontra nas posições  7, 13 e 18
        
        # Valores de saída
        idx_tokens = getIndexTokenTexto(lista_tokens, token)
                       
        # Avalia a saida do método       
        self.assertEqual(len (idx_tokens), 3)
        self.assertEqual(idx_tokens[0], 7)
        self.assertEqual(idx_tokens[1], 13)
        self.assertEqual(idx_tokens[2], 18)
        
if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.info("Teste Util Texto")
    unittest.main()
    