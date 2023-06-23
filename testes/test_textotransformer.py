# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de testes unitários
import unittest

# Biblioteca texto-transformer
from textotransformer.textotransformer import TextoTransformer

logger = logging.getLogger(__name__)

class TestTextTransformer(unittest.TestCase):
    
    @classmethod     
    def setUpClass(self):
        logger.info("Inicializando o modelo para os métodos de teste")
        # Instancia um objeto da classe TextoTransformer e recupera o MCL especificado
        self.modelo = TextoTransformer("neuralmind/bert-base-portuguese-cased") # BERTimbau base
    
    # Testes TextoTransformer   
    def test_textotransformer(self):
        logger.info("Testando o construtor de TextoTransformer")
                
        self.assertIsNotNone(self.modelo)

    # Testes getCodificacao string
    def test_getCodificacao_string(self):
        logger.info("Testando o getCodificacao com string")
                
        texto = "Adoro sorvete de manga."

        saida = self.modelo.getCodificacao(texto)
        
        self.assertEqual(len(saida), 10)
    
    # Testes getCodificacao lista de string    
    def test_getCodificacao_list_string(self):
        print("Testando o getCodificacao com lista de strings")
                
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]

        saida = self.modelo.getCodificacao(texto)
                
        self.assertEqual(len(saida), 2)
        self.assertEqual(len(saida[0]), 10)
        self.assertEqual(len(saida[1]), 11)
        
    # Testes getCodificacaoCompleta string
    def test_getCodificacaoCompleta_string(self):
        logger.info("Testando o getCodificacaoCompleta com string")
                
        texto = "Adoro sorvete de manga."

        saida = self.modelo.getCodificacaoCompleta(texto)
        
        self.assertEqual(len(saida), 7)
        self.assertEqual(len(saida['token_embeddings']), 10)
        self.assertEqual(len(saida['input_ids']), 10)
        self.assertEqual(len(saida['attention_mask']), 10)
        self.assertEqual(len(saida['token_type_ids']), 10)
        self.assertEqual(len(saida['tokens_texto_mcl']), 10)
        self.assertEqual(saida['texto_original'], texto)
        
    # Testes getCodificacaoCompleta lista de string
    def test_getCodificacaoCompleta_list_string(self):
        logger.info("Testando o getCodificacaoCompleta com lista de strings")
                
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]

        saida = self.modelo.getCodificacaoCompleta(texto)
        
        self.assertEqual(len(saida), 7)       
    
    # Testes getCodificacaoToken string
    def test_getCodificacaoToken_string(self):
        logger.info("Testando o getCodificacaoToken com strings")
        
        texto = "Adoro sorvete de manga."
        
        saida = self.modelo.getCodificacaoToken(texto)
        
        self.assertEqual(len(saida), 3)
        self.assertEqual(len(saida['tokens_texto_mcl']), 8)
        self.assertEqual(len(saida['token_embeddings']), 8)
        self.assertEqual(saida['texto_original'], texto)
        
    # Testes getCodificacaoCompleta lista de string
    def test_getCodificacaoToken_lista_string(self):
        logger.info("Testando o getCodificacaoToken lista com strings")
        
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]
        
        saida = self.modelo.getCodificacaoToken(texto)
        
        self.assertEqual(len(saida), 3)
        self.assertEqual(len(saida['tokens_texto_mcl']), 2)
        self.assertEqual(len(saida['tokens_texto_mcl'][0]), 8)
        self.assertEqual(len(saida['tokens_texto_mcl'][1]), 9)
        self.assertEqual(len(saida['token_embeddings']), 2)
        self.assertEqual(len(saida['token_embeddings'][0]), 8)
        self.assertEqual(len(saida['token_embeddings'][1]), 9)
        self.assertEqual(len(saida['texto_original']), 2)
        self.assertEqual(saida['texto_original'][0], texto[0])
        self.assertEqual(saida['texto_original'][1], texto[1])
        
        
if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("Teste TextoTransformer")
    unittest.main()
    