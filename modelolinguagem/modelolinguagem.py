# Import das bibliotecas.
import logging  # Biblioteca de logging

# Biblioteca dos modelos de linguagem
from modelo.modeloarguments import ModeloArgumentos
from nlp.nlpmodulo import NLP
from util.utilconstantes import *
from mensurador.mensuradorenum import *
from mensurador.mensurador import Mensurador
from modelo.transformer import Transformer

logger = logging.getLogger(__name__)

# Definição dos parâmetros do Modelo para os cálculos das Medidas
modelo_argumentos = ModeloArgumentos(
        max_seq_len=512,
        pretrained_model_name_or_path="neuralmind/bert-base-portuguese-cased", # Nome do modelo de linguagem pré-treinado Transformer
        modelo_spacy="pt_core_news_lg", # Nome do modelo de linguagem da ferramenta de NLP
        do_lower_case=False,            # default True
        output_attentions=False,        # default False
        output_hidden_states=True,      # default False  /Retornar os embeddings das camadas ocultas  
        camadas_embeddings = 2,         # 0-Primeira/1-Penúltima/2-Ùltima/3-Soma 4 últimas/4-Concat 4 últiamas/5-Todas
        estrategia_pooling=0,           # 0 - MEAN estratégia média / 1 - MAX  estratégia maior
        palavra_relevante=0             # 0 - Considera todas as palavras das sentenças / 1 - Desconsidera as stopwords / 2 - Considera somente as palavras substantivas
        )

class ModeloLinguagem:
    
    ''' 
    Carrega e cria um modelo de Linguagem, que pode ser usado para gerar embeddings de tokens, palavras, sentenças e textos.
     
    Parâmetros:
    `pretrained_model_name_or_path` - Se for um caminho de arquivo no disco, carrega o modelo a partir desse caminho. Se não for um caminho, ele primeiro faz o download do repositório de modelos do Huggingface com esse nome. Valor default: 'neuralmind/bert-base-portuguese-cased'.                  
    `modelo_spacy` - Nome do modelo a ser instalado e carregado pela ferramenta de nlp spaCy. Valor default 'pt_core_news_lg'.                       
    `camadas_embeddings` - Especifica de qual camada ou camadas será recuperado os embeddings do transformer. Valor defaul '2'. Valores possíveis: 0-Primeira/1-Penúltima/2-Ùltima/3-Soma 4 últimas/4-Concat 4 últiamas/5-Todas.       
    ''' 
    
    # Construtor da classe
    def __init__(self, pretrained_model_name_or_path="neuralmind/bert-base-portuguese-cased", 
                       modelo_spacy="pt_core_news_lg",
                       camadas_embeddings=2):
                       
        # Parâmetro recebido para o modelo de linguagem
        modelo_argumentos.pretrained_model_name_or_path = pretrained_model_name_or_path
               
        # Parâmetro recebido para o modelo da ferramenta de nlp
        modelo_argumentos.modelo_spacy = modelo_spacy
                
        # Carrega o modelo de linguagem da classe transformador
        self.transformer_model = Transformer(modelo_args=modelo_argumentos)
    
        # Recupera o modelo.
        self.model = self.transformer_model.get_auto_model()
    
        # Recupera o tokenizador.     
        self.tokenizer = self.transformer_model.get_tokenizer()
        
        # Especifica de qual camada utilizar os embeddings        
        logger.info("Utilizando embeddings do modelo da {} camada(s).".format(listaTipoCamadas[modelo_argumentos.camadas_embeddings][3]))
                    
        # Especifica camadas para recuperar os embeddings
        modelo_argumentos.camadas_embeddings = camadas_embeddings
      
        # Carrega o spaCy
        self.nlp = NLP(modelo_args=modelo_argumentos)
                        
        # Constroi um mensurador
        self.mensurador = Mensurador(modelo_args=modelo_argumentos, 
                                     transformer_model=self.transformer_model, 
                                     nlp=self.nlp)        
    
        logger.info("Classe ModeloLinguagem carregada: {}.".format(modelo_argumentos))
    
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
                logger.info("Não foi especificado uma estratégia de pooling válida.") 

    def definePalavraRelevante(self, palavraRelevante):
        ''' 
        Define a estratégia de palavra relavante para os parâmetros do modelo.
        
        Parâmetros:        
        `palavraRelevante` - Estratégia de relevância das palavras do texto.               
        ''' 
        
        if palavraRelevante == PalavrasRelevantes.CLEAN.name:
            modelo_argumentos.palavra_relevante = PalavrasRelevantes.CLEAN.value
            
        else:
            if palavraRelevante == PalavrasRelevantes.NOUN.name:
                modelo_argumentos.palavra_relevante = PalavrasRelevantes.NOUN.value
                
            else:
                if palavraRelevante == PalavrasRelevantes.ALL.name:
                    modelo_argumentos.palavra_relevante = PalavrasRelevantes.ALL.value                    
                else:
                    logger.info("Não foi especificado uma estratégia de relevância de palavras do texto válida.") 

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
        
        self.Ccos, self.Ceuc, self.Cman = self.mensurador.getMedidasComparacaoTexto(texto, 
                                                                                    camada=modelo_argumentos.camadas_embeddings, 
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
                                                                    camada=modelo_argumentos.camadas_embeddings, 
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
                                                                    camada=modelo_argumentos.camadas_embeddings, 
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
                                                                    camada=modelo_argumentos.camadas_embeddings, 
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
        
    def get_nlp(self):
        return self.nlp          
        
        
