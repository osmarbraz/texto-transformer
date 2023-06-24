# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de testes unitários
import unittest

# Bibliotecas próprias
from textotransformer.modelo.modeloenum import AbordagemExtracaoEmbeddingsCamadas, EstrategiasPooling
from textotransformer.util.utilconstantes import LISTATIPOCAMADA_NOME

logger = logging.getLogger(__name__)

class TestModeloEnum(unittest.TestCase):
        
    # Testes EstrategiasPooling_converteInt
    def test_EstrategiasPooling_converteInt(self):
        logger.info("Testando o EstrategiasPooling_converteInt")
                
        # Testa a conversão de int para EstrategiasPooling
        self.assertEqual(EstrategiasPooling.converteInt(0), EstrategiasPooling.MEAN)
        self.assertEqual(EstrategiasPooling.converteInt(1), EstrategiasPooling.MAX)
       
    # Testes AbordagemExtracaoEmbeddingsCamadas_converteInt
    def test_AbordagemExtracaoEmbeddingsCamadas_converteInt(self):
        logger.info("Testando o AbordagemExtracaoEmbeddingsCamadas_converteInt")
        
        # Testa a conversão de int para AbordagemExtracaoEmbeddingsCamadas
        self.assertEqual(AbordagemExtracaoEmbeddingsCamadas.converteInt(0), AbordagemExtracaoEmbeddingsCamadas.PRIMEIRA_CAMADA)
        self.assertEqual(AbordagemExtracaoEmbeddingsCamadas.converteInt(1), AbordagemExtracaoEmbeddingsCamadas.PENULTIMA_CAMADA)
        self.assertEqual(AbordagemExtracaoEmbeddingsCamadas.converteInt(2), AbordagemExtracaoEmbeddingsCamadas.ULTIMA_CAMADA)
        self.assertEqual(AbordagemExtracaoEmbeddingsCamadas.converteInt(3), AbordagemExtracaoEmbeddingsCamadas.SOMA_4_ULTIMAS_CAMADAS)
        self.assertEqual(AbordagemExtracaoEmbeddingsCamadas.converteInt(4), AbordagemExtracaoEmbeddingsCamadas.CONCAT_4_ULTIMAS_CAMADAS)
        self.assertEqual(AbordagemExtracaoEmbeddingsCamadas.converteInt(5), AbordagemExtracaoEmbeddingsCamadas.TODAS_AS_CAMADAS)
   
    # Testes AbordagemExtracaoEmbeddingsCamadas_getStr
    def test_AbordagemExtracaoEmbeddingsCamadas_getStr(self):
        logger.info("Testando o AbordagemExtracaoEmbeddingsCamadas_getStr")
        
        self.assertEqual(AbordagemExtracaoEmbeddingsCamadas.converteInt(0).getStr(), AbordagemExtracaoEmbeddingsCamadas.PRIMEIRA_CAMADA.value[LISTATIPOCAMADA_NOME])
        self.assertEqual(AbordagemExtracaoEmbeddingsCamadas.converteInt(1).getStr(), AbordagemExtracaoEmbeddingsCamadas.PENULTIMA_CAMADA.value[LISTATIPOCAMADA_NOME])
        self.assertEqual(AbordagemExtracaoEmbeddingsCamadas.converteInt(2).getStr(), AbordagemExtracaoEmbeddingsCamadas.ULTIMA_CAMADA.value[LISTATIPOCAMADA_NOME])
        self.assertEqual(AbordagemExtracaoEmbeddingsCamadas.converteInt(3).getStr(), AbordagemExtracaoEmbeddingsCamadas.SOMA_4_ULTIMAS_CAMADAS.value[LISTATIPOCAMADA_NOME])
        self.assertEqual(AbordagemExtracaoEmbeddingsCamadas.converteInt(4).getStr(), AbordagemExtracaoEmbeddingsCamadas.CONCAT_4_ULTIMAS_CAMADAS.value[LISTATIPOCAMADA_NOME])
        self.assertEqual(AbordagemExtracaoEmbeddingsCamadas.converteInt(5).getStr(), AbordagemExtracaoEmbeddingsCamadas.TODAS_AS_CAMADAS.value[LISTATIPOCAMADA_NOME])
   
   
if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.info("Teste Util Conversão")
    unittest.main()
    