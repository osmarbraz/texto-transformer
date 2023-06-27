# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de testes unitários
import unittest
# Biblioteca de aprendizado de máquina
import torch 

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
        
        # Valores de entrada                
        texto = "Adoro sorvete de manga."
        
        # Valores de saída
        saida = self.modelo.getSaidaRede(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 7) 
        
    # Testes getSaidaRedeCamada
    def test_getSaidaRedeCamada(self):
        logger.info("Testando o getSaidaRedeCamada")
         
        # Valores de entrada       
        texto = "Adoro sorvete de manga."
        
        # Valores de saída
        saida = self.modelo.getSaidaRedeCamada(texto, 2) # Camada 2 - Ultima camada dos transformers
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 9)
    
    # Testes getCodificacaoCompleta string
    def test_getCodificacaoCompleta_string(self):
        logger.info("Testando o getCodificacaoCompleta com string")
        
        # Valores de entrada        
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
        
        # Testa a saida dos valores das chaves
        self.assertEqual(len(saida['token_embeddings']), 10)
        self.assertEqual(len(saida['input_ids']), 10)
        self.assertEqual(len(saida['attention_mask']), 10)
        self.assertEqual(len(saida['token_type_ids']), 10)
        self.assertEqual(len(saida['tokens_texto_mcl']), 10)        
        self.assertEqual(saida['texto_original'], texto)
        self.assertEqual(len(saida['all_layer_embeddings']), 12) # Camadas
        self.assertEqual(len(saida['all_layer_embeddings'][0]), 10) # tokens
        self.assertEqual(len(saida['all_layer_embeddings'][0][0]), 768) # dimensões
        
        # Testa o tipo das saida dos valores das chaves                
        self.assertTrue(isinstance(saida['token_embeddings'], torch.Tensor))
        self.assertTrue(isinstance(saida['token_embeddings'][0], torch.Tensor))
        
        self.assertTrue(isinstance(saida['all_layer_embeddings'], list))
        self.assertTrue(isinstance(saida['all_layer_embeddings'][0], torch.Tensor))
        self.assertTrue(isinstance(saida['all_layer_embeddings'][0][0], torch.Tensor))
                
    # Testes getCodificacaoCompleta lista de string
    def test_getCodificacaoCompleta_list_string(self):
        logger.info("Testando o getCodificacaoCompleta com lista de strings")
        
        # Valores de entrada        
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]

        # Valores de saída
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
                
        # Testa a saida dos valores das chaves
        self.assertEqual(len(saida['token_embeddings']), 2) # Textos
        self.assertEqual(len(saida['token_embeddings'][0]), 10) # tokens
        self.assertEqual(len(saida['token_embeddings'][0][0]), 768) # embeddings
        self.assertEqual(len(saida['token_embeddings'][1]), 11) # tokens
        self.assertEqual(len(saida['token_embeddings'][1][0]), 768) # embeddings
        self.assertEqual(len(saida['input_ids']), 2)
        self.assertEqual(len(saida['attention_mask']), 2)
        self.assertEqual(len(saida['token_type_ids']), 2)
        self.assertEqual(len(saida['tokens_texto_mcl']), 2)
        self.assertEqual(saida['texto_original'][0], texto[0])
        self.assertEqual(saida['texto_original'][1], texto[1])
        self.assertEqual(len(saida['all_layer_embeddings']), 2) # Textos
        self.assertEqual(len(saida['all_layer_embeddings'][0]), 12) # Camadas do transformer
        self.assertEqual(len(saida['all_layer_embeddings'][0][0]), 10) # 10 tokens
        self.assertEqual(len(saida['all_layer_embeddings'][0][0][0]), 768) # embeddings
        self.assertEqual(len(saida['all_layer_embeddings'][1][0]), 11) # 11 tokens
        self.assertEqual(len(saida['all_layer_embeddings'][1][0][0]), 768) # embeddings
                
        # Testa o tipo das saida dos valores das chaves        
        self.assertTrue(isinstance(saida['token_embeddings'], list))
        self.assertTrue(isinstance(saida['token_embeddings'][0], torch.Tensor))
        self.assertTrue(isinstance(saida['token_embeddings'][0][0], torch.Tensor))
        self.assertTrue(isinstance(saida['token_embeddings'][1][0], torch.Tensor))
        
        self.assertTrue(isinstance(saida['all_layer_embeddings'], list))
        self.assertTrue(isinstance(saida['all_layer_embeddings'][0], list))
        self.assertTrue(isinstance(saida['all_layer_embeddings'][0][0], torch.Tensor))
        self.assertTrue(isinstance(saida['all_layer_embeddings'][0][0][0], torch.Tensor))
        self.assertTrue(isinstance(saida['all_layer_embeddings'][1], list))
        self.assertTrue(isinstance(saida['all_layer_embeddings'][1][0], torch.Tensor))
        self.assertTrue(isinstance(saida['all_layer_embeddings'][1][0][0], torch.Tensor))
            
    # Testes getCodificacao
    def test_getCodificacao_string(self):
        logger.info("Testando o getCodificacao(texto)")
        
        # Valores de entrada
        texto = "Adoro sorvete de manga."
        
        # Valores de saída
        saida = self.modelo.getCodificacao(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 3) 
        
       # Testa a saida dos valores das chaves
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("token_embeddings" in saida)
        self.assertTrue("texto_original" in saida)
    
    # Testes getCodificacaoGranularidade0
    def test_getCodificacao_granularidade_0(self):
        logger.info("Testando o getCodificacao(texto,granularidade_texto=0)")
        
        # Valores de entrada
        texto = "Adoro sorvete de manga."
        
        # Valores de saída
        saida = self.modelo.getCodificacao(texto,granularidade_texto=0)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 3) 
        
        # Testa o nome das chaves
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("token_embeddings" in saida)
        self.assertTrue("texto_original" in saida)

    # Testes getCodificacaoTexto string
    def test_getCodificacaoTexto_string(self):
        logger.info("Testando o getCodificacaoTexto com string")             
        
        # Valores de entrada
        texto = "Adoro sorvete de manga."
        
        # Valores de saída
        saida = self.modelo.getCodificacaoTexto(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 4) 
        
        # Testa o nome das chaves
        self.assertTrue("texto_original" in saida)
        self.assertTrue("tokens_texto_mcl" in saida)        
        self.assertTrue("texto_embeddings_MEAN" in saida)
        self.assertTrue("texto_embeddings_MAX" in saida)
                
        # Testa a saida dos valores das chaves
        self.assertEqual(len(saida['texto_original']), 23)
        self.assertEqual(len(saida['tokens_texto_mcl']), 8)
        # Testa a quantidade de embeddings
        self.assertEqual(len(saida['texto_embeddings_MEAN']), 768)
        self.assertEqual(len(saida['texto_embeddings_MAX']), 768)
        self.assertEqual(saida['texto_original'], texto)
        
        # Testa o tipo das saida dos valores das chaves        
        self.assertTrue(isinstance(saida['texto_embeddings_MEAN'], torch.Tensor))        
        self.assertTrue(isinstance(saida['texto_embeddings_MAX'], torch.Tensor))
        
    # Testes getCodificacaoTexto lista_string
    def test_getCodificacaoTexto_lista_string(self):
        logger.info("Testando o getCodificacaoTexto com lista de string")             
        
        # Valores de entrada
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]
        
        # Valores de saída
        saida = self.modelo.getCodificacaoTexto(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 4) 
        
        # Testa o nome das chaves
        self.assertTrue("texto_original" in saida)
        self.assertTrue("tokens_texto_mcl" in saida)        
        self.assertTrue("texto_embeddings_MEAN" in saida)
        self.assertTrue("texto_embeddings_MAX" in saida)
                
        # Testa a saida dos valores das chaves
        self.assertEqual(len(saida['texto_original']), 2)
        self.assertEqual(len(saida['tokens_texto_mcl']), 2)
        # Testa a quantidade de embeddings
        self.assertEqual(len(saida['texto_embeddings_MEAN']), 2)
        self.assertEqual(len(saida['texto_embeddings_MAX']), 2)
        self.assertEqual(saida['texto_original'][0], texto[0])
        self.assertEqual(saida['texto_original'][1], texto[1])
        
        # Testa o tipo das saida dos valores das chaves        
        self.assertTrue(isinstance(saida['texto_embeddings_MEAN'], list))
        self.assertTrue(isinstance(saida['texto_embeddings_MEAN'][0], torch.Tensor))
        self.assertTrue(isinstance(saida['texto_embeddings_MEAN'][1], torch.Tensor))
        
        self.assertTrue(isinstance(saida['texto_embeddings_MAX'], list))
        self.assertTrue(isinstance(saida['texto_embeddings_MAX'][0], torch.Tensor))
        self.assertTrue(isinstance(saida['texto_embeddings_MAX'][1], torch.Tensor))        

    # Testes getCodificacaoSentenca string
    def test_getCodificacaoSentenca_string(self):
        logger.info("Testando o getCodificacaoSentenca com string")             
        
        # Valores de entrada
        texto = "Adoro sorvete de manga. Sujei a manga da camisa."
        
        # Valores de saída
        saida = self.modelo.getCodificacaoSentenca(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 5) 
        
        # Testa o nome das chaves
        self.assertTrue("texto_original" in saida)
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("sentencas_texto" in saida)
        self.assertTrue("sentenca_embeddings_MEAN" in saida)
        self.assertTrue("sentenca_embeddings_MAX" in saida)
                
        # Testa a saida dos valores das chaves        
        self.assertEqual(len(saida['tokens_texto_mcl']), 17)
        self.assertEqual(len(saida['sentencas_texto']), 2)
        # Testa a quantidade de embeddings
        self.assertEqual(len(saida['sentenca_embeddings_MEAN']), 2)
        self.assertEqual(len(saida['sentenca_embeddings_MAX']), 2)
        self.assertEqual(saida['texto_original'], texto)
        
        # Testa o tipo das saida dos valores das chaves        
        self.assertTrue(isinstance(saida['sentenca_embeddings_MEAN'], list))
        self.assertTrue(isinstance(saida['sentenca_embeddings_MEAN'][0], torch.Tensor))
        
        self.assertTrue(isinstance(saida['sentenca_embeddings_MAX'], list))
        self.assertTrue(isinstance(saida['sentenca_embeddings_MAX'][0], torch.Tensor))

    # Testes getCodificacaoSentenca lista string
    def test_getCodificacaoSentenca_lista_string(self):
        logger.info("Testando o getCodificacaoSentenca com lista de string")             
        
        # Valores de entrada        
        texto = ["Adoro sorvete de manga. Sujei a manga da camisa.","Bom dia."]
        
        # Valores de saída
        saida = self.modelo.getCodificacaoSentenca(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 5) 
        
        # Testa o nome das chaves
        self.assertTrue("texto_original" in saida)
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("sentencas_texto" in saida)
        self.assertTrue("sentenca_embeddings_MEAN" in saida)
        self.assertTrue("sentenca_embeddings_MAX" in saida)
                
        # Testa a saida dos valores das chaves        
        self.assertEqual(len(saida['tokens_texto_mcl']), 2)
        self.assertEqual(len(saida['tokens_texto_mcl'][0]), 17)
        self.assertEqual(len(saida['tokens_texto_mcl'][1]), 3)
        self.assertEqual(len(saida['sentencas_texto']), 2)
        self.assertEqual(len(saida['sentencas_texto'][0]), 2)
        self.assertEqual(len(saida['sentencas_texto'][1]), 1)
        
        # Testa a quantidade de embeddings
        self.assertEqual(len(saida['sentenca_embeddings_MEAN']), 2)
        self.assertEqual(len(saida['sentenca_embeddings_MEAN'][0]), 2)
        self.assertEqual(len(saida['sentenca_embeddings_MEAN'][1]), 1)
        self.assertEqual(len(saida['sentenca_embeddings_MAX']), 2)
        self.assertEqual(len(saida['sentenca_embeddings_MAX'][0]), 2)
        self.assertEqual(len(saida['sentenca_embeddings_MAX'][1]), 1)
        self.assertEqual(saida['texto_original'], texto)
        
        # Testa o tipo das saida dos valores das chaves        
        self.assertTrue(isinstance(saida['sentenca_embeddings_MEAN'], list))
        self.assertTrue(isinstance(saida['sentenca_embeddings_MEAN'][0], list))
        self.assertTrue(isinstance(saida['sentenca_embeddings_MEAN'][0][0], torch.Tensor))
        self.assertTrue(isinstance(saida['sentenca_embeddings_MEAN'][0][1], torch.Tensor))
        self.assertTrue(isinstance(saida['sentenca_embeddings_MEAN'][1], list))
        self.assertTrue(isinstance(saida['sentenca_embeddings_MEAN'][1][0], torch.Tensor))
                
        self.assertTrue(isinstance(saida['sentenca_embeddings_MAX'], list))
        self.assertTrue(isinstance(saida['sentenca_embeddings_MAX'][0], list))
        self.assertTrue(isinstance(saida['sentenca_embeddings_MAX'][0][0], torch.Tensor))
        self.assertTrue(isinstance(saida['sentenca_embeddings_MAX'][0][1], torch.Tensor))
        self.assertTrue(isinstance(saida['sentenca_embeddings_MAX'][1], list))
        self.assertTrue(isinstance(saida['sentenca_embeddings_MAX'][1][0], torch.Tensor))

    # Testes getCodificacaoPalavra string
    def test_getCodificacaoPalavra_string(self):
        logger.info("Testando o getCodificacaoPalavra com strings")             
        
        # Valores de entrada
        texto = "Adoro sorvete de manga."
        
        # Valores de saída
        saida = self.modelo.getCodificacaoPalavra(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 8) 
        
        # Testa o nome das chaves
        self.assertTrue("texto_original" in saida)
        self.assertTrue("tokens_texto" in saida)
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("tokens_oov_texto_mcl" in saida)
        self.assertTrue("tokens_texto_pln" in saida)
        self.assertTrue("pos_texto_pln" in saida)
        self.assertTrue("palavra_embeddings_MEAN" in saida)
        self.assertTrue("palavra_embeddings_MAX" in saida)
                
        # Testa a saida dos valores das chaves
        self.assertEqual(len(saida['tokens_texto']), 5)
        self.assertEqual(len(saida['tokens_texto_mcl']), 8)
        self.assertEqual(len(saida['tokens_oov_texto_mcl']), 5)
        self.assertEqual(len(saida['tokens_texto_pln']), 5)
        self.assertEqual(len(saida['pos_texto_pln']), 5)
        # Testa a quantidade de embeddings
        self.assertEqual(len(saida['palavra_embeddings_MEAN']), 5)
        self.assertEqual(len(saida['palavra_embeddings_MAX']), 5)
        self.assertEqual(saida['texto_original'], texto)
        
        # Testa o tipo das saida dos valores das chaves        
        self.assertTrue(isinstance(saida['palavra_embeddings_MEAN'], list))
        self.assertTrue(isinstance(saida['palavra_embeddings_MEAN'][0], torch.Tensor))
        
        self.assertTrue(isinstance(saida['palavra_embeddings_MAX'], list))
        self.assertTrue(isinstance(saida['palavra_embeddings_MAX'][0], torch.Tensor))
        
    # Testes getCodificacaoPalavra lista de string
    def test_getCodificacaoPalavra_lista_string(self):
        logger.info("Testando o getCodificacaoPalavra lista com string")
        
        # Valores de entrada
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]
        
        # Valores de saída
        saida = self.modelo.getCodificacaoPalavra(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 8)
                
        # Testa o nome das chaves
        self.assertTrue("texto_original" in saida)
        self.assertTrue("tokens_texto" in saida)
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("tokens_oov_texto_mcl" in saida)
        self.assertTrue("tokens_texto_pln" in saida)
        self.assertTrue("pos_texto_pln" in saida)
        self.assertTrue("palavra_embeddings_MEAN" in saida)
        self.assertTrue("palavra_embeddings_MAX" in saida)
        
        # Testa a saida dos valores das chaves
        # Testa a quantidade de textos
        self.assertEqual(len(saida['tokens_texto_mcl']), 2)
        # Testa a quantidde de tokens das textos
        self.assertEqual(len(saida['tokens_texto_mcl'][0]), 8)
        self.assertEqual(len(saida['tokens_texto_mcl'][1]), 9)
        # Testa a quantidade de embeddings
        self.assertEqual(len(saida['palavra_embeddings_MEAN']), 2)
        # Testa a quantidde de de embeddings
        self.assertEqual(len(saida['palavra_embeddings_MEAN'][0]), 5)
        self.assertEqual(len(saida['palavra_embeddings_MEAN'][1]), 6)        
        # Testa a quantidade de embeddings
        self.assertEqual(len(saida['palavra_embeddings_MAX']), 2)
        # Testa a quantidde de de embeddings
        self.assertEqual(len(saida['palavra_embeddings_MAX'][0]), 5)
        self.assertEqual(len(saida['palavra_embeddings_MAX'][1]), 6)
        
        # Testa a quantidade de textos
        self.assertEqual(len(saida['texto_original']), 2)
        # Testa o valor dos textos
        self.assertEqual(saida['texto_original'][0], texto[0])
        self.assertEqual(saida['texto_original'][1], texto[1])
        
        # Testa o tipo das saida dos valores das chaves      
        # MEAN  
        self.assertTrue(isinstance(saida['palavra_embeddings_MEAN'], list))
        # Tipo do primeiro texto
        self.assertTrue(isinstance(saida['palavra_embeddings_MEAN'][0], list))
        # Tipo do segundo texto
        self.assertTrue(isinstance(saida['palavra_embeddings_MEAN'][1], list))
        # Tipo dos elementos da lista do primeiro texto
        self.assertTrue(isinstance(saida['palavra_embeddings_MEAN'][0][0], torch.Tensor))
        # Tipo dos elementos da lista do segundo texto
        self.assertTrue(isinstance(saida['palavra_embeddings_MEAN'][1][0], torch.Tensor)) 
        # MAX
        self.assertTrue(isinstance(saida['palavra_embeddings_MAX'], list))
        # Tipo do primeiro texto
        self.assertTrue(isinstance(saida['palavra_embeddings_MAX'][0], list))
        # Tipo do segundo texto
        self.assertTrue(isinstance(saida['palavra_embeddings_MAX'][1], list))
        # Tipo dos elementos da lista do primeiro texto
        self.assertTrue(isinstance(saida['palavra_embeddings_MAX'][0][0], torch.Tensor))
        # Tipo dos elementos da lista do segundo texto
        self.assertTrue(isinstance(saida['palavra_embeddings_MAX'][1][0], torch.Tensor))          
             
    # Testes getCodificacaoToken string
    def test_getCodificacaoToken_string(self):
        logger.info("Testando o getCodificacaoToken com string")
        
        # Valores de entrada
        texto = "Adoro sorvete de manga."
        
        # Valores de saída
        saida = self.modelo.getCodificacaoToken(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 3) 
        
        # Testa o nome das chaves
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("token_embeddings" in saida)
        self.assertTrue("texto_original" in saida)
        
        # Testa a saida dos valores das chaves
        self.assertEqual(len(saida['tokens_texto_mcl']), 8)
        self.assertEqual(len(saida['token_embeddings']), 8)
        self.assertEqual(saida['texto_original'], texto)
        
        # Testa o tipo das saida dos valores das chaves        
        self.assertTrue(isinstance(saida['token_embeddings'], list))
        self.assertTrue(isinstance(saida['token_embeddings'][0], torch.Tensor))        
        
    # Testes getCodificacaoToken lista de string
    def test_getCodificacaoToken_lista_string(self):
        logger.info("Testando o getCodificacaoToken lista com strings")
        
        # Valores de entrada
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]
        
        # Valores de saída
        saida = self.modelo.getCodificacaoToken(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 3)
                
         # Testa o nome das chaves
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("token_embeddings" in saida)
        self.assertTrue("texto_original" in saida)
        
        # Testa a saida dos valores das chaves
        # Testa a quantidade de textos
        self.assertEqual(len(saida['tokens_texto_mcl']), 2)
        # Testa a quantidde de tokens das textos
        self.assertEqual(len(saida['tokens_texto_mcl'][0]), 8)
        self.assertEqual(len(saida['tokens_texto_mcl'][1]), 9)
        # Testa a quantidade de textos
        self.assertEqual(len(saida['token_embeddings']), 2)
        # Testa a quantidde de tokens das textos
        self.assertEqual(len(saida['token_embeddings'][0]), 8)
        self.assertEqual(len(saida['token_embeddings'][1]), 9)
        # Testa a quantidade de textos
        self.assertEqual(len(saida['texto_original']), 2)
        # Testa o valor dos textos
        self.assertEqual(saida['texto_original'][0], texto[0])
        self.assertEqual(saida['texto_original'][1], texto[1])
        
        # Testa o tipo das saida dos valores das chaves        
        self.assertTrue(isinstance(saida['token_embeddings'], list))
        # Tipo do primeiro texto
        self.assertTrue(isinstance(saida['token_embeddings'][0], list))
        # Tipo do segundo texto
        self.assertTrue(isinstance(saida['token_embeddings'][1], list))
        # Tipo dos elementos da lista do primeiro texto
        self.assertTrue(isinstance(saida['token_embeddings'][0][0], torch.Tensor))
        # Tipo dos elementos da lista do segundo texto
        self.assertTrue(isinstance(saida['token_embeddings'][1][0], torch.Tensor))
    
    # Testes getMedidasTexto
    def test_getMedidasTexto(self):
        logger.info("Testando o getMedidasTexto(texto)")
        
        # Valores de entrada
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]

        # Valores de saída
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
        
        # Valores de entrada
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]

        # Valores de saída
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
        
        # Valores de entrada
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]

        # Valores de saída
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
        
        # Valores de entrada
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]

        # Valores de saída
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
    