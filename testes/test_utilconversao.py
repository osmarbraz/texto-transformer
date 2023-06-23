# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de testes unitários
import unittest

# Bibliotecas próprias
from textotransformer.mensurador.mensuradorenum import PalavraRelevante
from textotransformer.modelo.modeloenum import AbordagemExtracaoEmbeddingsCamadas, EstrategiasPooling
from textotransformer.util.utilconstantes import LISTATIPOCAMADA_NOME
from textotransformer.util.utilconversao import getAbordagemExtracaoEmbeddingsCamadasStr, getIntParaAbordagemExtracaoEmbeddingsCamadas, getIntParaEstrategiasPooling, getIntParaPalavraRelevante

logger = logging.getLogger(__name__)

class TestUtilConversao(unittest.TestCase):
        
    # Testes getIntParaPalavraRelevante
    def test_getIntParaPalavraRelevante(self):
        logger.info("Testando o getIntParaPalavraRelevante")
        
        self.assertEqual(getIntParaPalavraRelevante(0), PalavraRelevante.ALL)
        self.assertEqual(getIntParaPalavraRelevante(1), PalavraRelevante.CLEAN)
        self.assertEqual(getIntParaPalavraRelevante(2), PalavraRelevante.NOUN)

    # Testes getIntParaEstrategiasPooling
    def test_getIntParaEstrategiasPooling(self):
        logger.info("Testando o getIntParaEstrategiasPooling")
        
        self.assertEqual(getIntParaEstrategiasPooling(0), EstrategiasPooling.MEAN)
        self.assertEqual(getIntParaEstrategiasPooling(1), EstrategiasPooling.MAX)
        
    # Testes getIntParaAbordagemExtracaoEmbeddingsCamadas
    def test_getIntParaAbordagemExtracaoEmbeddingsCamadas(self):
        logger.info("Testando o getIntParaAbordagemExtracaoEmbeddingsCamadas")
        
        self.assertEqual(getIntParaAbordagemExtracaoEmbeddingsCamadas(0), AbordagemExtracaoEmbeddingsCamadas.PRIMEIRA_CAMADA)
        self.assertEqual(getIntParaAbordagemExtracaoEmbeddingsCamadas(1), AbordagemExtracaoEmbeddingsCamadas.PENULTIMA_CAMADA)
        self.assertEqual(getIntParaAbordagemExtracaoEmbeddingsCamadas(2), AbordagemExtracaoEmbeddingsCamadas.ULTIMA_CAMADA)
        self.assertEqual(getIntParaAbordagemExtracaoEmbeddingsCamadas(3), AbordagemExtracaoEmbeddingsCamadas.SOMA_4_ULTIMAS_CAMADAS)
        self.assertEqual(getIntParaAbordagemExtracaoEmbeddingsCamadas(4), AbordagemExtracaoEmbeddingsCamadas.CONCAT_4_ULTIMAS_CAMADAS)
        self.assertEqual(getIntParaAbordagemExtracaoEmbeddingsCamadas(5), AbordagemExtracaoEmbeddingsCamadas.TODAS_AS_CAMADAS)
   
    # Testes getAbordagemExtracaoEmbeddingsCamadasStr
    def test_getAbordagemExtracaoEmbeddingsCamadasStr(self):
        logger.info("Testando o getAbordagemExtracaoEmbeddingsCamadasStr")
        
        self.assertEqual(getAbordagemExtracaoEmbeddingsCamadasStr(0), AbordagemExtracaoEmbeddingsCamadas.PRIMEIRA_CAMADA.value[LISTATIPOCAMADA_NOME])
        self.assertEqual(getAbordagemExtracaoEmbeddingsCamadasStr(1), AbordagemExtracaoEmbeddingsCamadas.PENULTIMA_CAMADA.value[LISTATIPOCAMADA_NOME])
        self.assertEqual(getAbordagemExtracaoEmbeddingsCamadasStr(2), AbordagemExtracaoEmbeddingsCamadas.ULTIMA_CAMADA.value[LISTATIPOCAMADA_NOME])
        self.assertEqual(getAbordagemExtracaoEmbeddingsCamadasStr(3), AbordagemExtracaoEmbeddingsCamadas.SOMA_4_ULTIMAS_CAMADAS.value[LISTATIPOCAMADA_NOME])
        self.assertEqual(getAbordagemExtracaoEmbeddingsCamadasStr(4), AbordagemExtracaoEmbeddingsCamadas.CONCAT_4_ULTIMAS_CAMADAS.value[LISTATIPOCAMADA_NOME])
        self.assertEqual(getAbordagemExtracaoEmbeddingsCamadasStr(5), AbordagemExtracaoEmbeddingsCamadas.TODAS_AS_CAMADAS.value[LISTATIPOCAMADA_NOME])
   
   
if "__main__" == __name__:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logger.info("Teste Util Conversão")
    unittest.main()
    