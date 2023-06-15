# Diretório do cohebert
DIRETORIO_MODELO_LINGUAGEM = 'modelo_linguagem'

# ============================
# listaTipoCamadas
# Define uma lista com as camadas a serem analisadas nos teste.
# Cada elemento da lista 'listaTipoCamadas' é chamado de camada sendo formado por:
#  - camada[0] = Índice da camada
#  - camada[1] = Um inteiro com o índice da camada a ser avaliada. Pode conter valores negativos.
#  - camada[2] = Operação para n camadas, CONCAT ou SUM.
#  - camada[3] = Nome do tipo camada

# Os nomes do tipo da camada pré-definidos.
#  - 0 - Primeira                    
#  - 1 - Penúltima
#  - 2 - Última
#  - 3 - Soma 4 últimas
#  - 4 - Concat 4 últimas
#  - 5 - Todas

# Constantes para facilitar o acesso os tipos de camadas
PRIMEIRA_CAMADA = 0
PENULTIMA_CAMADA = 1
ULTIMA_CAMADA = 2
SOMA_4_ULTIMAS_CAMADAS = 3
CONCAT_4_ULTIMAS_CAMADAS = 4
TODAS_AS_CAMADAS = 5

# Constantes para padronizar o acesso aos dados do modelo de linguagem.
TEXTO_TOKENIZADO = 0
INPUT_IDS = 1
ATTENTION_MASK = 2
TOKEN_TYPE_IDS = 3
OUTPUTS = 4
OUTPUTS_LAST_HIDDEN_STATE = 0
OUTPUTS_POOLER_OUTPUT = 1
OUTPUTS_HIDDEN_STATES = 2

# Modelo de Linguagem BERT Large possui 25 camadas(1a camada com os tokens de entrada e 24 camadas dos transformers)
# Modelo de Linguagem BERT Large possui 13 camadas(1a camada com os tokens de entrada e 24 camadas dos transformers)
# O índice da camada com valor positivo indica uma camada específica
# O índica com um valor negativo indica as camadas da posição com base no fim descontado o valor indice até o fim.
listaTipoCamadas = [
    [PRIMEIRA_CAMADA, 1, '-', 'Primeira'], 
    [PENULTIMA_CAMADA, -2, '-', 'Penúltima'],
    [ULTIMA_CAMADA, -1, '-', 'Última'],
    [SOMA_4_ULTIMAS_CAMADAS, -4, 'SUM', 'Soma 4 últimas'],
    [CONCAT_4_ULTIMAS_CAMADAS, -4, 'CONCAT', 'Concat 4 últimas'], 
    [TODAS_AS_CAMADAS, 24, 'SUM', 'Todas']
]

# listaTipoCamadas e suas referências:
# 0 - Primeira            listaTipoCamadas[PRIMEIRA_CAMADA]
# 1 - Penúltima           listaTipoCamadas[PENULTIMA_CAMADA]
# 2 - Última              listaTipoCamadas[ULTIMA_CAMADA]
# 3 - Soma 4 últimas      listaTipoCamadas[SOMA_4_ULTIMAS_CAMADAS]
# 4 - Concat 4 últimas    listaTipoCamadas[CONCAT_4_ULTIMAS_CAMADAS]
# 5 - Todas               listaTipoCamadas[TODAS_AS_CAMADAS]