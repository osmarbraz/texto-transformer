# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de testes unitários
import unittest

# Biblioteca texto-transformer
from textotransformer.textotransformer import TextoTransformer

logger = logging.getLogger(__name__)

class TestTextTransformer(unittest.TestCase):
    
    # Inicialização do modelo para os testes
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
        
        self.assertEqual(len(saida), 10) # Dicionário possui 10 chaves
    
    # Testes getSaidaRede 
    def test_getSaidaRede(self):
        logger.info("Testando o getSaidaRede")
                
        texto = "Adoro sorvete de manga."
        
        saida = self.modelo.getSaidaRede(texto)
        
        self.assertEqual(len(saida), 7) # Dicionário possui 7 chaves
        
    # Testes getSaidaRedeCamada
    def test_getSaidaRedeCamada(self):
        logger.info("Testando o getSaidaRedeCamada")
                
        texto = "Adoro sorvete de manga."
        
        saida = self.modelo.getSaidaRedeCamada(texto, 2) # Camada 2 - Ultima camada dos transformers
        
        self.assertEqual(len(saida), 9) # Dicionário possui 9 chaves    
    
    # Testes getCodificacao lista de string    
    def test_getCodificacao_list_string(self):
        print("Testando o getCodificacao com lista de strings")
                
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]

        saida = self.modelo.getCodificacao(texto)
                
        self.assertEqual(len(saida), 2) # Dicionário possui 2 chaves
        self.assertEqual(len(saida[0]), 10)
        self.assertEqual(len(saida[1]), 11)
        
    # Testes getCodificacaoCompleta string
    def test_getCodificacaoCompleta_string(self):
        logger.info("Testando o getCodificacaoCompleta com string")
                
        texto = "Adoro sorvete de manga."

        saida = self.modelo.getCodificacaoCompleta(texto)
        
        self.assertEqual(len(saida), 7) # Dicionário possui 7 chaves
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
        
        self.assertEqual(len(saida), 7) # Dicionário possui 7 chaves   
    
    # Testes getCodificacaoToken string
    def test_getCodificacaoToken_string(self):
        logger.info("Testando o getCodificacaoToken com strings")
        
        texto = "Adoro sorvete de manga."
        
        saida = self.modelo.getCodificacaoToken(texto)
        
        self.assertEqual(len(saida), 3) # Dicionário possui 3 chaves
        self.assertEqual(len(saida['tokens_texto_mcl']), 8)
        self.assertEqual(len(saida['token_embeddings']), 8)
        self.assertEqual(saida['texto_original'], texto)
        
    # Testes getCodificacaoCompleta lista de string
    def test_getCodificacaoToken_lista_string(self):
        logger.info("Testando o getCodificacaoToken lista com strings")
        
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]
        
        saida = self.modelo.getCodificacaoToken(texto)
        
        self.assertEqual(len(saida), 3) # Dicionário possui 3 chaves
        self.assertEqual(len(saida['tokens_texto_mcl']), 2)
        self.assertEqual(len(saida['tokens_texto_mcl'][0]), 8)
        self.assertEqual(len(saida['tokens_texto_mcl'][1]), 9)
        self.assertEqual(len(saida['token_embeddings']), 2)
        self.assertEqual(len(saida['token_embeddings'][0]), 8)
        self.assertEqual(len(saida['token_embeddings'][1]), 9)
        self.assertEqual(len(saida['texto_original']), 2)
        self.assertEqual(saida['texto_original'][0], texto[0])
        self.assertEqual(saida['texto_original'][1], texto[1])
    
    # Testes getMedidasTexto
    def test_getMedidasTexto(self):
        print("Rodando getMedidasTexto")
        
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]

        # Recupera as medida do texto
        saida = self.modelo.getMedidasTexto(texto, palavra_relevante=0)
        
        CcosEsperado = 0.7125453352928162                
        CeucEsperado = 5.883016586303711
        CmanEsperado = 125.89885711669922
                       
        # Compara somente 5 casas decimais
        self.assertEqual(round(saida['cos'],5), round(CcosEsperado,5))
        self.assertEqual(round(saida['euc'],5), round(CeucEsperado,5))
        self.assertEqual(round(saida['man'],5), round(CmanEsperado,5))
               
if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("Teste TextoTransformer")
    unittest.main()
    