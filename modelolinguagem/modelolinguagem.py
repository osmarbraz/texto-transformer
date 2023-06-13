# Import das bibliotecas.
import logging  # Biblioteca de logging

# Biblioteca dos modelos de linguagem
from modelo.modeloarguments import ModeloArgumentos
from spacynlp.spacymodulo import *

from medidor.medidorenum import *
from medidor.mensurador import Mensurador
from modelo.transformer import Transformer

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

# Índice dos campos da camada
LISTATIPOCAMADA_ID = 0
LISTATIPOCAMADA_CAMADA = 1
LISTATIPOCAMADA_OPERACAO = 2
LISTATIPOCAMADA_NOME = 3

# BERT Large possui 25 camadas(1a camada com os tokens de entrada e 24 camadas dos transformers)
# BERT Large possui 13 camadas(1a camada com os tokens de entrada e 24 camadas dos transformers)
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


# Definição dos parâmetros do Modelo para os cálculos das Medidas
modelo_argumentos = ModeloArgumentos(
                                    max_seq_len=512,
                                    pretrained_model_name_or_path='neuralmind/bert-base-portuguese-cased', 
                                    modelo_spacy='pt_core_news_lg',
                                    versao_spacy='3.4.4',
                                    do_lower_case=False,        # default True
                                    output_attentions=False,    # default False
                                    output_hidden_states=True,  # default False  /Retornar os embeddings das camadas ocultas  
                                    camadas_embeddings = 2,     # 0-Primeira/1-Penúltima/2-Ùltima/3-Soma 4 últimas/4-Concat 4 últiamas/5-Todas
                                    estrategia_pooling=0,       # 0 - MEAN estratégia média / 1 - MAX  estratégia maior
                                    palavra_relevante=0         # 0 - Considera todas as palavras das sentenças / 1 - Desconsidera as stopwords / 2 - Considera somente as palavras substantivas
                                    )

class ModeloLinguagem:
    
    ''' 
    Carrega e cria um modelo de Linguagem, que pode ser usado para gerar embeddings de tokens, palavras, sentenças e textos.
     
    Parâmetros:
    `pretrained_model_name_or_path' - Se for um caminho de arquivo no disco, carrega o modelo a partir desse caminho. Se não for um caminho, ele primeiro tenta fazer o download de um modelo pré-treinado do modelo de linguagem. Se isso falhar, tenta construir um modelo do repositório de modelos do Huggingface com esse nome.
    ''' 
    
    # Construtor da classe
    def __init__(self, pretrained_model_name_or_path, camadas_embeddings=2):
        # Parâmetro recebido para o modelo de linguagem
        modelo_argumentos.pretrained_model_name_or_path = pretrained_model_name_or_path
                
        # Carrega o modelo de linguagem da classe transformador
        self.transformer_model = Transformer(modelo_args=modelo_argumentos)
    
        # Recupera o modelo.
        self.model = self.transformer_model.get_auto_model()
    
        # Recupera o tokenizador.     
        self.tokenizer = self.transformer_model.get_tokenizer()
        
        # Carrega o spaCy
        self.verificaCarregamentoSpacy()
        
        # Especifica de qual camada utilizar os embeddings
        logging.info("Utilizando embeddings do modelo de:", listaTipoCamadas[modelo_argumentos.camadas_embeddings]) 
        
        # Especifica camadas para recuperar os embeddings
        modelo_argumentos.camadas_embeddings = camadas_embeddings
        
        # Define que camadas de embeddings a ser utilizada
        self.TipoCamadas = listaTipoCamadas[modelo_argumentos.camadas_embeddings]
        
        # Constroi um mensurador
        self.mensurador = Mensurador(modelo_args=modelo_argumentos, 
                                     transformer_model=self.transformer_model, 
                                     nlp=self.nlp)
    
    def verificaCarregamentoSpacy(self):
        ''' 
        Verifica se é necessário carregar o spacy.
        Utilizado para as estratégias de palavras relevantes CLEAN e NOUN.
        ''' 
        
        if modelo_argumentos.palavra_relevante != PalavrasRelevantes.ALL.value:
            # Carrega o modelo spacy
            print("Carregando o spaCy")
            self.nlp = carregaSpacy(modelo_argumentos)
            
        else:
            print("spaCy não carregado!")
            self.nlp = None
    
    def defineEstrategiaPooling(self, estrategiaPooling):
        ''' 
        Define a estratégia de pooling para os parâmetros do modelo.

        Parâmetros:
        `estrategiaPooling` - Estratégia de pooling das camadas do BERT.
        ''' 
        
        if estrategiaPooling == EstrategiasPooling.MAX.name:
            modelo_argumentos.estrategia_pooling = EstrategiasPooling.MAX.value
            
        else:
            if estrategiaPooling == EstrategiasPooling.MEAN.name:
                modelo_argumentos.estrategia_pooling = EstrategiasPooling.MEAN.value
            else:
                logging.info("Não foi especificado uma estratégia de pooling válida.") 

    def definePalavraRelevante(self, palavraRelevante):
        ''' 
        Define a estratégia de palavra relavante para os parâmetros do modelo.
        
        Parâmetros:        
        `palavraRelevante` - Estratégia de relevância das palavras do texto.               
        ''' 
        
        if palavraRelevante == PalavrasRelevantes.CLEAN.name:
            modelo_argumentos.palavra_relevante = PalavrasRelevantes.CLEAN.value
            verificaCarregamentoSpacy()
            
        else:
            if palavraRelevante == PalavrasRelevantes.NOUN.name:
                modelo_argumentos.palavra_relevante = PalavrasRelevantes.NOUN.value
                verificaCarregamentoSpacy()
                
            else:
                if palavraRelevante == PalavrasRelevantes.ALL.name:
                    modelo_argumentos.palavra_relevante = PalavrasRelevantes.ALL.value
                    
                else:
                    logging.info("Não foi especificado uma estratégia de relevância de palavras do texto válida.") 

    def getMedidasTexto(self, texto, estrategiaPooling='MEAN', palavraRelevante='ALL'):
        ''' 
        Retorna as medidas de (in)coerência Ccos, Ceuc, Cman do texto.
        
        Parâmetros:
        `texto` - Um texto a ser medido.           
        `estrategiaPooling` - Estratégia de pooling das camadas do BERT.
        `palavraRelevante` - Estratégia de relevância das palavras do texto.            
        
        Retorno:
        `Ccos` - Medida de coerência Ccos do do texto.            
        `Ceuc` - Medida de incoerência Ceuc do do texto.            
        `Cman` - Medida de incoerência Cman do do texto.            
        ''' 

        self.defineEstrategiaPooling(estrategiaPooling)
        self.definePalavraRelevante(palavraRelevante)

        self.Ccos, self.Ceuc, self.Cman = self.mensurador.getMedidasComparacaoTexto(texto,                                                            camada=self.TipoCamadas, 
                                                                    tipoTexto='o')
          
        return self.Ccos, self.Ceuc, self.Cman
    
    def getMedidasTextoCosseno(self, 
                               texto, 
                               estrategiaPooling='MEAN', 
                               palavraRelevante='ALL'):
        ''' 
        Retorna a medida de coerência do texto utilizando a medida de similaridade de cosseno.
        
        Parâmetros:
        `texto` - Um texto a ser medido a coerência.           
        `estrategiaPooling` - Estratégia de pooling das camadas do BERT. 
        `palavraRelevante` - Estratégia de relevância das palavras do texto.            
        
        Retorno:
        `Ccos` - Medida de coerência Ccos do do texto.            
        '''         
        
        self.defineEstrategiaPooling(estrategiaPooling)
        self.definePalavraRelevante(palavraRelevante)

        self.Ccos, self.Ceuc, self.Cman = self.mensurador.getMedidasComparacaoTexto(texto, 
                                                                    camada=self.TipoCamadas, 
                                                                    tipoTexto='o')
          
        return self.Ccos
    
    def getMedidasTextoEuclediana(self, texto, estrategiaPooling='MEAN', palavraRelevante='ALL'):
        ''' 
        Retorna a medida de incoerência do texto utilizando a medida de distância de Euclidiana.
                 
        Parâmetros:
        `texto` - Um texto a ser medido a coerência.           
        `estrategiaPooling` - Estratégia de pooling das camadas do BERT.
        `palavraRelevante` - Estratégia de relevância das palavras do texto.            
        
        Retorno:
        `Ceuc` - Medida de incoerência Ceuc do do texto.            
        ''' 
        
        self.defineEstrategiaPooling(estrategiaPooling)
        self.definePalavraRelevante(palavraRelevante)

        self.Ccos, self.Ceuc, self.Cman = self.mensurador.getMedidasComparacaoTexto(texto,
                                                                    camada=self.TipoCamadas, 
                                                                    tipoTexto='o')
          
        return self.Ceuc        
    
   

   def getMedidasTextoManhattan(self, texto, estrategiaPooling='MEAN', palavraRelevante='ALL'):
        ''' 
        Retorna a medida de incoerência do texto utilizando a medida de distância de Manhattan.
                 
        Parâmetros:
        `texto` - Um texto a ser medido a coerência.           
        `estrategiaPooling` - Estratégia de pooling das camadas do BERT.
        `palavraRelevante` - Estratégia de relevância das palavras do texto.            
        
        Retorno:
        `Cman` - Medida de incoerência Cman do do texto.            
        ''' 
        
        self.defineEstrategiaPooling(estrategiaPooling)
        self.definePalavraRelevante(palavraRelevante)
        
        self.Ccos, self.Ceuc, self.Cman = self.mensurador.getMedidasComparacaoTexto(texto, 
                                                                    camada=self.TipoCamadas, 
                                                                    tipoTexto='o')
          
        return self.Cman                


    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_transformer_model(self):
        return self.transformer_model
        
    def get_mensurador(self):
        return self.mensurador        
        
        
