# Import das bibliotecas.

# Biblioteca de logging
import logging  
# Biblioteca do transformer hunggingface
from transformers import AutoModel, AutoTokenizer, AutoConfig

# Bibliotecas próprias
from textotransformer.modelo.transformer import Transformer  
from textotransformer.modelo.modeloargumentos import ModeloArgumentos
from textotransformer.pln.pln import PLN

# Objeto de logger
logger = logging.getLogger(__name__)

# T5 em pt-br tem dependência do sentencepiece==0.1.99 e protobuf==3.20.
class TransformerT5(Transformer):
    '''
    Classe que encapsula a classe T5Model da Huggingface para gerar embeddings de token, palavra, sentença ou texto.
    '''

    def __init__(self, auto_model: AutoModel = None, 
                 auto_config: AutoConfig = None, 
                 auto_tokenizer: AutoTokenizer = None, 
                 modelo_args: ModeloArgumentos = None):
        '''
        Construtor da classe TransformerT5.

        Parâmetros:
            `auto_model` - Auto model modelo carregado.
            `auto_config` - Auto config carregado.
            `auto_tokenizer` - Auto tokenizer carregado.
            `modelo_args' - Argumentos passados para o modelo Huggingface Transformers.
        '''
        
        # Inicializa o construtor da superclasse
        super(TransformerT5, self).__init__(
                auto_model=auto_model, 
                auto_config=auto_config, 
                auto_tokenizer=auto_tokenizer, 
                modelo_args=modelo_args
        )
        
        # Define os tokens especiais e separadores 
        self.defineTokensEspeciais()
        
        # Se não possuir um token de preenchimento, adiciona um        
        if self.auto_tokenizer.pad_token is None:
            self.auto_tokenizer.add_special_tokens(({'pad_token': self.TOKEN_PADDING})) 
            self.auto_model.resize_token_embeddings(len(self.auto_tokenizer))
                           
        logger.info("Classe \"{}\" carregada: \"{}\".".format(self.__class__.__name__, modelo_args))

    # ============================   
    def __repr__(self):
        '''
        Retorna uma string com descrição do objeto.
        '''

        return "Classe (\"{}\") carregada com o modelo \"{}\", AutoConfig \"{}\", Transformer \"{}\" e tokenizador: \"{}\".".format(self.__class__.__name__,
                                                                                                                                      self.modelo_args.pretrained_model_name_or_path,
                                                                                                                                      self.auto_config.__class__.__name__,
                                                                                                                                      self.auto_model.__class__.__name__,
                                                                                                                                      self.auto_tokenizer.__class__.__name__)
    
    # ============================   
    def defineTokensEspeciais(self):
        '''
        Define os tokens especiais e separadores considerando o modelo instânciado.
        
        # A maioria dos modelos a posição do token de início é 1 e o token separador é -1
        # Em alguns a posição do token de início é 0(não existe) e o token separador é -2 e o último <sep> é o token de classificação <CLS>
        '''
      
        # Uma sentença simples: X <sep> <cls>
        # Um par de sentenças: A <sep> B <sep> <cls>
        self.TOKEN_INICIO = None  # O token de início está no final da sentença junto como separador.
        self.TOKEN_FIM = "</s>" # Não existe token de fim de texto.
        self.TOKEN_SEPARADOR = "</s>" # Token separador de sentença.
        self.TOKEN_CLASSIFICACAO = None # Token de classificação.
        self.TOKEN_PADDING = "<pad>" # O token de padding.
        self.PADDING_SIDE = 1 # Define o lado que será realizado o preenchimento das lista de tokens. 0: esquerda, 1: direita.
        self.TOKEN_MASCARA = "<extra_id_0>" # Token de máscara.
        self.TOKEN_DESCONHECIDO = "<unk>"  # Token desconhecido.        
        self.SEPARADOR_SUBTOKEN = "▁"  # Caracter que separa palavras fora do vocabulário segundo o Algoritmo SentencePiece.
        self.POSICAO_TOKEN_INICIO = 0 # Posição do primeiro token válido do início da lista de tokens.        
        self.POSICAO_TOKEN_FINAL = -1 ## Posição último do token válido do final da lista de tokens. Valor "None" indica que é o último token.        
        self.SERAPADOR_SUBTOKEN_REPETICAO = 1 # Repetição do separador subtoken. -1 - Sem separador subtoken, 0 - nos subtokens(menos primeiro), 1 - somente primeiro subtoken, 2 - somente último subtoken.
        self.SEPARADOR_SUBTOKEN_POSICAO = 0 # Posição do separador de subtoken. -1 - Sem separador de subtoken, 0 - no início do token,  1 - no fim do token.
        self.PRIMEIRO_TOKEN_SEM_SEPARADOR = False # Define se o primeiro token não terá separador de substoken. Ex. True - ['token1','Ġtoken2', 'Ġtoken3'] False - ['Ġtoken1','Ġtoken2', 'Ġtoken3'].
        self.DO_LOWER_CASE = False # Define se o tokenizador irá converter os tokens para minúsculo.

    # ============================  
    def getTokensPalavrasEmbeddingsTexto(self, 
                                         embeddings_texto, 
                                         tokens_texto_mcl: list[str],
                                         tokens_texto_concatenado_pln: str,
                                         pln: PLN,
                                         dic_excecao: dict = {"":0,}) -> dict:
        '''
        Escolhe o melhor tokenizador de palavra para o texto de acordo com o modelo de linguagem.
        
        De um texto preparado(tokenizado) ou não, retorna os embeddings das palavras do texto. 
        Retorna 5 listas, os tokens(palavras), as postagging, tokens OOV, e os embeddings dos tokens igualando a quantidade de tokens do spaCy com a tokenização do MCL de acordo com a estratégia. 
        Utiliza duas estratégias para realizar o pooling de tokens que forma uma palavra.
            - Estratégia MEAN para calcular a média dos embeddings dos tokens que formam uma palavra.
            - Estratégia MAX para calcular o valor máximo dos embeddings dos tokens que formam uma palavra.
            
        Parâmetros:
           `texto` - Um texto a ser recuperado os embeddings das palavras do modelo de linguagem
           `embeddings_texto` - Os embeddings do texto gerados pelo método getEmbeddingsTexto
           `tokens_texto_mcl` - Os tokens do texto gerados pelo método getEmbeddingsTexto
           `tokens_texto_concatenado_pln` - Os tokens do texto concatenado gerados pela ferramenta de PLN.
           `pln` - Uma instância da classe PLN para realizar a tokenização e POS-Tagging do texto.
           `dic_excecao` - Um dicionário de tokens de exceções e seus deslocamentos para considerar mais ou menos tokens do modelo de linguagem em relação ao spaCy. Valor positivo para considerar mais tokens e negativo para considerar menos tokens. Exemplo exceção negativo: {"1°": -1}, a ferramenta de PLN separa o token "1°" em "1" e "°", portanto é necessário reduzir 1 token pois o MCL gera somente um. Exemplo exceção positiva: {"1°": 1}, a ferramenta de PLN não separa o token "1°", mas o MCL separa em dois "1" e "°" portanto é necessário agrupar em 1 token.
               
        Retorna um dicionário com as seguintes chaves: 
           `tokens_texto` - Uma lista com os tokens do texto gerados pelo método.
           `pos_texto_pln` - Uma lista com as postagging dos tokens gerados pela ferramenta de PLN.
           `tokens_oov_texto_mcl` - Uma lista com os tokens OOV do mcl.
           `palavra_embeddings_MEAN` - Uma lista dos embeddings de palavras com a média dos embeddings(Estratégia MEAN) dos tokens que formam a palavra.
           `palavra_embeddings_MAX` - Uma lista dos embeddings de palavras com o máximo dos embeddings(Estratégia MAX) dos tokens que formam a palavra.
        ''' 
        
        # Tokenização SentencePiece (Separador, _) T5
        return self.getTokensPalavrasEmbeddingsTextoSentencePiece(embeddings_texto=embeddings_texto,
                                                                  lista_tokens_texto_mcl=tokens_texto_mcl,
                                                                  tokens_texto_concatenado_pln=tokens_texto_concatenado_pln,
                                                                  pln=pln,
                                                                  dic_excecao=dic_excecao)
    
                            