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
    
    # Testes getSaidaRede 
    def test_getSaidaRede(self):
        logger.info("Testando o getSaidaRede")
                
        texto = "Adoro sorvete de manga."
        
        saida = self.modelo.getSaidaRede(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 7) 
        
    # Testes getSaidaRedeCamada
    def test_getSaidaRedeCamada(self):
        logger.info("Testando o getSaidaRedeCamada")
                
        texto = "Adoro sorvete de manga."
        
        saida = self.modelo.getSaidaRedeCamada(texto, 2) # Camada 2 - Ultima camada dos transformers
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 9)
    
    # Testes getCodificacaoCompleta string
    def test_getCodificacaoCompleta_string(self):
        logger.info("Testando o getCodificacaoCompleta com string")
                
        texto = "Adoro sorvete de manga."

        saida = self.modelo.getCodificacaoCompleta(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 7)
        
        # Testa o nome das chaves
        self.assertTrue("token_embeddings" in saida)
        self.assertTrue("input_ids" in saida)
        self.assertTrue("attention_mask" in saida)
        self.assertTrue("token_type_ids" in saida)
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("texto_original" in saida)
        self.assertTrue("all_layer_embeddings" in saida)
        
        # Testa a saida das chaves
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
    
    # Testes getCodificacao
    def test_getCodificacao_string(self):
        logger.info("Testando o getCodificacao(texto)")
        
        texto = "Adoro sorvete de manga."
        
        saida = self.modelo.getCodificacao(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 3) 
        
        # Testa o nome das chaves
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("token_embeddings" in saida)
        self.assertTrue("texto_original" in saida)
    
    # Testes getCodificacaoGranularidade0
    def test_getCodificacao_granularidade_0(self):
        logger.info("Testando o getCodificacao(texto,granularidade_texto=0)")
        
        texto = "Adoro sorvete de manga."
        
        saida = self.modelo.getCodificacao(texto,granularidade_texto=0)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 3) 
        
        # Testa o nome das chaves
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("token_embeddings" in saida)
        self.assertTrue("texto_original" in saida)
             
    # Testes getCodificacaoToken string
    def test_getCodificacaoToken_string(self):
        logger.info("Testando o getCodificacaoToken com strings")
        
        texto = "Adoro sorvete de manga."
        
        saida = self.modelo.getCodificacaoToken(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 3) 
        
        # Testa o nome das chaves
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("token_embeddings" in saida)
        self.assertTrue("texto_original" in saida)
        
        # Testa a saida das chaves
        self.assertEqual(len(saida['tokens_texto_mcl']), 8)
        self.assertEqual(len(saida['token_embeddings']), 8)
        self.assertEqual(saida['texto_original'], texto)
        
    # Testes getCodificacaoCompleta lista de string
    def test_getCodificacaoToken_lista_string(self):
        logger.info("Testando o getCodificacaoToken lista com strings")
        
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]
        
        saida = self.modelo.getCodificacaoToken(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 3)
                
         # Testa o nome das chaves
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("token_embeddings" in saida)
        self.assertTrue("texto_original" in saida)
        
        # Testa a saida das chaves
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
        logger.info("Testando o getMedidasTexto(texto)")
        
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]

        # Recupera as medida do texto
        saida = self.modelo.getMedidasTexto(texto)
        
        CcosEsperado = 0.7125453352928162                
        CeucEsperado = 5.883016586303711
        CmanEsperado = 125.89885711669922
                       
        # Testa o nome das chaves
        self.assertTrue("cos" in saida)
        self.assertTrue("euc" in saida)
        self.assertTrue("man" in saida)
                       
        # Compara somente 5 casas decimais
        self.assertEqual(round(saida['cos'],5), round(CcosEsperado,5))
        self.assertEqual(round(saida['euc'],5), round(CeucEsperado,5))
        self.assertEqual(round(saida['man'],5), round(CmanEsperado,5))
        
    # Testes getMedidasTextoPalavraRelevante_0
    def test_getMedidasTexto_PalavraRelevante_0(self):
        logger.info("Testando o getMedidasTexto(texto, palavra_relevante=0)")
        
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]

        # Recupera as medida do texto
        saida = self.modelo.getMedidasTexto(texto, palavra_relevante=0)
        
        CcosEsperado = 0.7125453352928162                
        CeucEsperado = 5.883016586303711
        CmanEsperado = 125.89885711669922
                       
        # Testa o nome das chaves
        self.assertTrue("cos" in saida)
        self.assertTrue("euc" in saida)
        self.assertTrue("man" in saida)
                       
        # Compara somente 5 casas decimais
        self.assertEqual(round(saida['cos'],5), round(CcosEsperado,5))
        self.assertEqual(round(saida['euc'],5), round(CeucEsperado,5))
        self.assertEqual(round(saida['man'],5), round(CmanEsperado,5))
        
    # Testes getMedidasTextoPalavraRelevante_1
    def test_getMedidasTexto_PalavraRelevante_1(self):
        logger.info("Rodando getMedidasTexto(texto, palavra_relevante=1)")
        
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]

        # Recupera as medida do texto
        saida = self.modelo.getMedidasTexto(texto, palavra_relevante=1)
        
        CcosEsperado = 0.726082324981689              
        CeucEsperado = 5.497143745422363
        CmanEsperado = 117.38613891601562
                                              
        # Testa o nome das chaves
        self.assertTrue("cos" in saida)
        self.assertTrue("euc" in saida)
        self.assertTrue("man" in saida)
                       
        # Compara somente 5 casas decimais
        self.assertEqual(round(saida['cos'],5), round(CcosEsperado,5))
        self.assertEqual(round(saida['euc'],5), round(CeucEsperado,5))
        self.assertEqual(round(saida['man'],5), round(CmanEsperado,5))

    # Testes getMedidasTextoPalavraRelevante_2
    def test_getMedidasTexto_PalavraRelevante_2(self):
        logger.info("Rodando .getMedidasTexto(texto, palavra_relevante=2)")
        
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]

        # Recupera as medida do texto
        saida = self.modelo.getMedidasTexto(texto, palavra_relevante=2)
        
        CcosEsperado = 0.0                
        CeucEsperado = 0.0
        CmanEsperado = 0.0
                       
        # Testa o nome das chaves
        self.assertTrue("cos" in saida)
        self.assertTrue("euc" in saida)
        self.assertTrue("man" in saida)
                       
        # Compara somente 5 casas decimais
        self.assertEqual(round(saida['cos'],5), round(CcosEsperado,5))
        self.assertEqual(round(saida['euc'],5), round(CeucEsperado,5))
        self.assertEqual(round(saida['man'],5), round(CmanEsperado,5))    
                       
if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("Teste TextoTransformer")
    unittest.main()
    