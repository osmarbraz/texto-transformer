# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de testes unitários
import unittest
# Biblioteca de aprendizado de máquina
import torch 

# Biblioteca texto-transformer
from textotransformer.textotransformer import TextoTransformer
from textotransformer.modelo.transformerbert import TransformerBert

# Objeto de logger
logger = logging.getLogger(__name__)

class TestTextTransformerMask_bert_en(unittest.TestCase):
    
    # Inicialização do modelo para os testes
    @classmethod     
    def setUpClass(self):
        logger.info("Inicializando o modelo para os métodos de teste")
        # Instancia um objeto da classe TextoTransformer e recupera o MCL especificado
        self.modelo = TextoTransformer("bert-base-cased", 
                                       modelo_spacy="en_core_web_sm",
                                       tipo_modelo_pretreinado="mascara")
    
    # Testes TextoTransformer_bert
    def test_textotransformer(self):
        logger.info("Testando o construtor de TextoTransformerMask_bert")
                
        self.assertIsNotNone(self.modelo)
        self.assertIsInstance(self.modelo.getTransformer(), TransformerBert)
    
    # Testes getPrevisaoPalavraTexto
    def test_getPrevisaoPalavraTexto(self):
        logger.info("Testando o removeTokensEspeciais")
        
        # Valores de entrada                
        texto = "The sky is [MASK]."
        
        # Valores de saída
        predicao = self.modelo.getPrevisaoPalavraTexto(texto, top_k_predicao=2)
        
        # Lista esperada
        token_esperado1 = 'blue'
        probabilidade_esperada1 = 0.20729941129684448
        token_esperado2 = 'clear'
        probabilidade_esperada2 = 0.18247811496257782
        
        # Testa as saídas
        self.assertEqual(len(predicao), 2)
        # Compara somente n casas decimais
        casas_decimais = 4
        self.assertEqual(predicao[0][0], token_esperado1) 
        self.assertAlmostEqual(predicao[0][1].item(), probabilidade_esperada1, places=casas_decimais)
        self.assertEqual(predicao[1][0], token_esperado2) 
        self.assertAlmostEqual(predicao[1][1].item(), probabilidade_esperada2, places=casas_decimais)

    # Testes getModificacaoTextoSequencial
    def test_getModificacaoTextoSequencial(self):
        logger.info("Testando o getModificacaoTextoSequencial")
        
        # Valores de entrada                
        texto = "How to enqueue elements in a queue?"
        
        # Valores de saída
        saida = self.modelo.getModificacaoTextoSequencial(texto, top_k_predicao=2)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 6) 
        
        # Testa as saídas         
        self.assertEqual(saida['texto_modificado'][0], 'How to put elements in a queue ?') 
        self.assertEqual(saida['texto_mascarado'][0], 'How to [MASK] elements in a queue ?')
        self.assertEqual(saida['palavra_mascarada'][0], 'enqueue') 
        self.assertEqual(saida['token_predito'][0], 'put')
        self.assertEqual(saida['token_peso'][0], 0.09357264637947083) 
        self.assertEqual(saida['token_predito_marcado'][0], 'put')
                   
if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("Teste TextoTransformerMask_bert_en")
    unittest.main()
    