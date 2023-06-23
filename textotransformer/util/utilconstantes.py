# Diretório do pacote
DIRETORIO_TEXTO_TRANSFORMER = 'modelo_linguagem'

# Os nomes do tipo da camada pré-definidos.
#  - 0 - Primeira                    
#  - 1 - Penúltima
#  - 2 - Última
#  - 3 - Soma 4 últimas
#  - 4 - Concat 4 últimas
#  - 5 - Todas

# Constantes para facilitar o acesso as abordagens de extração de embeddings das camadas do transformer.
PRIMEIRA_CAMADA = 0
PENULTIMA_CAMADA = 1
ULTIMA_CAMADA = 2
SOMA_4_ULTIMAS_CAMADAS = 3
CONCAT_4_ULTIMAS_CAMADAS = 4
TODAS_AS_CAMADAS = 5

# Índice dos campos da abordagem de extração de embeddings das camadas do transformer.
LISTATIPOCAMADA_ID = 0
LISTATIPOCAMADA_CAMADA = 1
LISTATIPOCAMADA_OPERACAO = 2
LISTATIPOCAMADA_NOME = 3

# Constantes para padronizar o acesso aos dados do modelo de linguagem.
TEXTO_TOKENIZADO = 0
INPUT_IDS = 1
ATTENTION_MASK = 2
TOKEN_TYPE_IDS = 3
OUTPUTS = 4
OUTPUTS_LAST_HIDDEN_STATE = 0
OUTPUTS_POOLER_OUTPUT = 1
OUTPUTS_HIDDEN_STATES = 2