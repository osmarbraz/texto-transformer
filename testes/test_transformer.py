# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de testes unitários
import unittest
# Biblioteca de aprendizado de máquina
import torch 

# Bibliotecas próprias
from textotransformer.modelo.modeloarguments import ModeloArgumentos
from textotransformer.modelo.transformer import Transformer
from textotransformer.modelo.modeloenum import AbordagemExtracaoEmbeddingsCamadas

logger = logging.getLogger(__name__)

# Definição dos parâmetros do Modelo para os cálculos das Medidas
modelo_argumentos = ModeloArgumentos(
    max_seq_len=512,
    pretrained_model_name_or_path="neuralmind/bert-base-portuguese-cased", # Nome do modelo de linguagem pré-treinado Transformer
    modelo_spacy="pt_core_news_lg",             # Nome do modelo de linguagem da ferramenta de PLN
    do_lower_case=False,                        # default True
    output_attentions=False,                    # default False
    output_hidden_states=True,                  # default False  /Retornar os embeddings das camadas ocultas  
    abordagem_extracao_embeddings_camadas=2,    # 0-Primeira/1-Penúltima/2-Ùltima/3-Soma 4 últimas/4-Concat 4 últiamas/5-Todas
    estrategia_pooling=0,                       # 0 - MEAN estratégia média / 1 - MAX  estratégia maior
    palavra_relevante=0                         # 0 - Considera todas as palavras das sentenças / 1 - Desconsidera as stopwords / 2 - Considera somente as palavras substantivas
)

class TestTransformer(unittest.TestCase):
    
    # Inicialização do modelo para os testes
    @classmethod     
    def setUpClass(self):
        logger.info("Inicializando o modelo para os métodos de teste")        
        self.modelo = Transformer(modelo_args=modelo_argumentos) 
    
    # Testes construtor    
    def test_transformer(self):
        logger.info("Testando o construtor de Transformer")
                
        self.assertIsNotNone(self.modelo)

    #Testes getTextoTokenizado string
    def test_getTextoTokenizado(self):
        logger.info("Testando o getTextoTokenizado com string")
        
        # Valores de entrada
        texto = "Adoro sorvete de manga."

        # Valores de saída
        saida = self.modelo.getTextoTokenizado(texto)
        
        saidaEsperada = ['[CLS]', 'Ado', '##ro', 'sor', '##vete', 'de', 'mang', '##a', '.', '[SEP]']
        
        self.assertEqual(saida, saidaEsperada)

    #Testes tokenize string
    def test_tokenize_string(self):
        logger.info("Testando o tokenize com string")
        
        # Valores de entrada        
        texto = "Adoro sorvete de manga."

        # Valores de saída
        saida = self.modelo.tokenize(texto)
          
         # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 5) 
        
        # Testa o nome das chaves
        self.assertTrue("input_ids" in saida)
        self.assertTrue("token_type_ids" in saida)
        self.assertTrue("attention_mask" in saida)
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("texto_original" in saida)
        
        # Testa a saida dos valores das chaves
        self.assertEqual(len(saida['input_ids']), 1)
        self.assertEqual(len(saida['input_ids'][0]), 10)
        self.assertEqual(len(saida['token_type_ids']), 1)  
        self.assertEqual(len(saida['token_type_ids'][0]), 10)
        self.assertEqual(len(saida['attention_mask']), 1)    
        self.assertEqual(len(saida['attention_mask'][0]), 10)  
        self.assertEqual(len(saida['tokens_texto_mcl']), 1)   
        self.assertEqual(len(saida['tokens_texto_mcl'][0]), 10)                
        self.assertEqual(len(saida['texto_original']), 1)
        self.assertEqual(saida['texto_original'][0], texto)

    # Testes tokenize Lista de Strings
    def test_tokenize_list_string(self):
        logger.info("Testando o tokenize com lista de strings")
        
        # Valores de entrada        
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]

        # Valores de saída
        saida = self.modelo.tokenize(texto)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 5) 
        
        # Testa o nome das chaves
        self.assertTrue("input_ids" in saida)
        self.assertTrue("token_type_ids" in saida)
        self.assertTrue("attention_mask" in saida)
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("texto_original" in saida)
        
        # Testa a saida dos valores das chaves
        self.assertEqual(len(saida['input_ids']), 2)
        self.assertEqual(len(saida['input_ids'][0]), 11)
        self.assertEqual(len(saida['input_ids'][1]), 11)
        self.assertEqual(len(saida['token_type_ids']), 2)
        self.assertEqual(len(saida['token_type_ids'][0]), 11)
        self.assertEqual(len(saida['token_type_ids'][1]), 11)
        self.assertEqual(len(saida['attention_mask']), 2)
        self.assertEqual(len(saida['attention_mask'][0]), 11)
        self.assertEqual(len(saida['attention_mask'][1]), 11)
        self.assertEqual(len(saida['tokens_texto_mcl']), 2)
        self.assertEqual(len(saida['tokens_texto_mcl'][0]), 11)
        self.assertEqual(len(saida['tokens_texto_mcl'][1]), 11) 
        self.assertEqual(len(saida['texto_original']), 2)
        self.assertEqual(saida['texto_original'][0], texto[0])
        self.assertEqual(saida['texto_original'][1], texto[1])

    # Testes getSaidaRede String
    def test_getSaidaRede_string(self):
        logger.info("Testando o getSaidaRede com string")
         
        # Valores de entrada        
        texto = "Adoro sorvete de manga."
        
        # Valores de saída
        saida = self.modelo.getSaidaRede(texto)
        
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
        self.assertEqual(len(saida['input_ids']), 1)
        self.assertEqual(len(saida['input_ids'][0]), 10)
        self.assertEqual(len(saida['token_type_ids']), 1)  
        self.assertEqual(len(saida['token_type_ids'][0]), 10)
        self.assertEqual(len(saida['attention_mask']), 1)    
        self.assertEqual(len(saida['attention_mask'][0]), 10)  
        self.assertEqual(len(saida['tokens_texto_mcl']), 1)   
        self.assertEqual(len(saida['tokens_texto_mcl'][0]), 10)        
        self.assertEqual(len(saida['texto_original']), 1)
        self.assertEqual(saida['texto_original'][0], texto)
        self.assertEqual(len(saida['token_embeddings']), 1)
        self.assertEqual(len(saida['token_embeddings'][0]), 10)
        self.assertEqual(len(saida['all_layer_embeddings']), 13) #Camadas
        self.assertEqual(len(saida['all_layer_embeddings'][0]), 1) #Textos
        self.assertEqual(len(saida['all_layer_embeddings'][0][0]), 10) #Tokens
        
        # Testa o tipo das saida dos valores das chaves        
        self.assertTrue(isinstance(saida['token_embeddings'], torch.Tensor))
        self.assertTrue(isinstance(saida['token_embeddings'][0], torch.Tensor))
        
        self.assertTrue(isinstance(saida['all_layer_embeddings'], tuple))
        self.assertTrue(isinstance(saida['all_layer_embeddings'][0], torch.Tensor))                   
        self.assertTrue(isinstance(saida['all_layer_embeddings'][0][0], torch.Tensor))         
        
    # Testes getSaidaRede Lista de Strings
    def test_getSaidaRede_lista_string(self):
        logger.info("Testando o getSaidaRede com lista de strings")
         
        # Valores de entrada       
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]
        
        # Valores de saída
        saida = self.modelo.getSaidaRede(texto)
        
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
        self.assertEqual(len(saida['input_ids']), 2)
        self.assertEqual(len(saida['input_ids'][0]), 11)
        self.assertEqual(len(saida['input_ids'][1]), 11)
        self.assertEqual(len(saida['token_type_ids']), 2)
        self.assertEqual(len(saida['token_type_ids'][0]), 11)
        self.assertEqual(len(saida['token_type_ids'][1]), 11)
        self.assertEqual(len(saida['attention_mask']), 2)
        self.assertEqual(len(saida['attention_mask'][0]), 11)
        self.assertEqual(len(saida['attention_mask'][1]), 11)
        self.assertEqual(len(saida['tokens_texto_mcl']), 2)
        self.assertEqual(len(saida['tokens_texto_mcl'][0]), 11)
        self.assertEqual(len(saida['tokens_texto_mcl'][1]), 11) 
        self.assertEqual(len(saida['texto_original']), 2)
        self.assertEqual(saida['texto_original'][0], texto[0])
        self.assertEqual(saida['texto_original'][1], texto[1])        
        self.assertEqual(len(saida['token_embeddings']), 2)
        self.assertEqual(len(saida['token_embeddings'][0]), 11)
        self.assertEqual(len(saida['token_embeddings'][1]), 11)                
        self.assertEqual(len(saida['all_layer_embeddings']), 13) #Camadas
        self.assertEqual(len(saida['all_layer_embeddings'][0]), 2) #Textos
        self.assertEqual(len(saida['all_layer_embeddings'][0][0]), 11) #Tokens
        self.assertEqual(len(saida['all_layer_embeddings'][1]), 2) #Textos
        self.assertEqual(len(saida['all_layer_embeddings'][1][0]), 11) #Tokens
                
        # Testa o tipo das saida dos valores das chaves        
        self.assertTrue(isinstance(saida['token_embeddings'], torch.Tensor))
        self.assertTrue(isinstance(saida['token_embeddings'][0], torch.Tensor))
        
        self.assertTrue(isinstance(saida['all_layer_embeddings'], tuple))
        self.assertTrue(isinstance(saida['all_layer_embeddings'][0], torch.Tensor))                   
        self.assertTrue(isinstance(saida['all_layer_embeddings'][0][0], torch.Tensor)) 

    # Testes getSaidaRedeCamada String
    def test_getSaidaRedeCamada_string(self):
        logger.info("Testando o getSaidaRedeCamada com strings")
        
        # Valores de entrada        
        texto = ["Adoro sorvete de manga.","Sujei a manga da camisa."]
        
        # Valores de saída
        # Abordagem extração de embeddings das camadas
        # 0-Primeira/1-Penúltima/2-Ùltima/3-Soma 4 últimas/4-Concat 4 últimas/5-Soma de todas
        saida = self.modelo.getSaidaRedeCamada(texto, 2)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 9) 
        
        # Testa o nome das chaves
        self.assertTrue("token_embeddings" in saida)
        self.assertTrue("input_ids" in saida)
        self.assertTrue("attention_mask" in saida)
        self.assertTrue("token_type_ids" in saida)
        self.assertTrue("tokens_texto_mcl" in saida)
        self.assertTrue("texto_original" in saida)
        self.assertTrue("all_layer_embeddings" in saida)
        self.assertTrue("embedding_extraido" in saida)
        self.assertTrue("abordagem_extracao_embeddings_camadas" in saida)
                
        # Testa a saida dos valores das chaves
        self.assertEqual(len(saida['input_ids']), 2)
        self.assertEqual(len(saida['input_ids'][0]), 11)
        self.assertEqual(len(saida['input_ids'][1]), 11)
        self.assertEqual(len(saida['token_type_ids']), 2)
        self.assertEqual(len(saida['token_type_ids'][0]), 11)
        self.assertEqual(len(saida['token_type_ids'][1]), 11)
        self.assertEqual(len(saida['attention_mask']), 2)
        self.assertEqual(len(saida['attention_mask'][0]), 11)
        self.assertEqual(len(saida['attention_mask'][1]), 11)
        self.assertEqual(len(saida['tokens_texto_mcl']), 2)
        self.assertEqual(len(saida['tokens_texto_mcl'][0]), 11)
        self.assertEqual(len(saida['tokens_texto_mcl'][1]), 11) 
        self.assertEqual(len(saida['texto_original']), 2)
        self.assertEqual(saida['texto_original'][0], texto[0])
        self.assertEqual(saida['texto_original'][1], texto[1])        
        
        self.assertEqual(len(saida['token_embeddings']), 2)
        self.assertEqual(len(saida['token_embeddings'][0]), 11)
        self.assertEqual(len(saida['token_embeddings'][1]), 11)        
        
        self.assertEqual(len(saida['all_layer_embeddings']), 13) #13 Camadas
        self.assertEqual(len(saida['all_layer_embeddings'][0]), 2) # Dois textos
        self.assertEqual(len(saida['all_layer_embeddings'][0][0]), 11) # 11 tokens
        self.assertEqual(len(saida['all_layer_embeddings'][1]),  2) # Dois textos
        self.assertEqual(len(saida['all_layer_embeddings'][1][0]), 11) # 11 tokens
        
        # Chaves adicionais para abordagem extração de embeddings das camadas
        self.assertEqual(len(saida['embedding_extraido']), 2) # Dois textos
        self.assertEqual(len(saida['embedding_extraido'][0]), 11) # 11 tokens        
        self.assertEqual(len(saida['embedding_extraido'][1]), 11) # 11 tokens
        
        self.assertEqual(saida['abordagem_extracao_embeddings_camadas'], AbordagemExtracaoEmbeddingsCamadas.ULTIMA_CAMADA) 
      
if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.info("Teste Transformer")
    unittest.main()
    