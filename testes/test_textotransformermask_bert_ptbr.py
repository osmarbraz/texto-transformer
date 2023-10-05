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

class TestTextTransformerMask_bert_ptbr(unittest.TestCase):
    
    # Inicialização do modelo para os testes
    @classmethod     
    def setUpClass(self):
        logger.info("Inicializando o modelo para os métodos de teste")
        # Instancia um objeto da classe TextoTransformer e recupera o MCL especificado
        self.modelo = TextoTransformer("neuralmind/bert-base-portuguese-cased",
                                       tipo_modelo_pretreinado="mascara") # BERTimbau base
    
    # Testes TextoTransformer_bert   
    def test_TextoTransformer_bert(self):
        logger.info("Testando o construtor de TextoTransformer_bert")
                
        self.assertIsNotNone(self.modelo)
        self.assertIsInstance(self.modelo.getTransformer(), TransformerBert)
            
    # Testes getPrevisaoPalavraTexto
    def test_getPrevisaoPalavraTexto(self):
        logger.info("Testando o removeTokensEspeciais")
        
        # Valores de entrada                
        texto = "O céu é [MASK]."
        
        # Valores de saída
        predicao = self.modelo.getPrevisaoPalavraTexto(texto, top_k_predicao=2)
        
        # Lista esperada
        token_esperado1 = 'azul'
        probabilidade_esperada1 = 0.34259724617004395
        token_esperado2 = 'claro'
        probabilidade_esperada2 = 0.04515097662806511
        
        # Testa as saídas 
        self.assertEqual(len(predicao), 2)
        # Compara somente n casas decimais
        casas_decimais = 3
        self.assertEqual(predicao[0][0], token_esperado1) 
        self.assertAlmostEqual(predicao[0][1].item(), probabilidade_esperada1, places=casas_decimais)
        self.assertEqual(predicao[1][0], token_esperado2) 
        self.assertAlmostEqual(predicao[1][1].item(), probabilidade_esperada2, places=casas_decimais)
        
    # Testes getModificacaoTextoSequencial
    def test_getModificacaoTextoSequencial(self):
        logger.info("Testando o getModificacaoTextoSequencial")
        
        # Valores de entrada                
        texto = "Como enfileirar elementos em uma fila?"
        
        # Valores de saída
        saida = self.modelo.getModificacaoTextoSequencial(texto, top_k_predicao=2)
        
        # Testa o tamanho do dicionário
        self.assertEqual(len(saida), 6) 
        
        # Testa as saídas         
        self.assertEqual(saida['texto_modificado'][0], 'Como encontrar elementos em uma fila ?') 
        self.assertEqual(saida['texto_mascarado'][0], 'Como [MASK] elementos em uma fila ?')
        self.assertEqual(saida['palavra_mascarada'][0], 'enfileirar') 
        self.assertEqual(saida['token_predito'][0], 'encontrar')
        self.assertEqual(saida['token_peso'][0], 0.13462261855602264) 
        self.assertEqual(saida['token_predito_marcado'][0], 'encontrar')
                       
if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.info("Teste TextoTransformerMask_bert_ptbr")
    unittest.main()
    