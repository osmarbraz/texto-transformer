# Import das bibliotecas.
import logging  # Biblioteca de logging

# Biblioteca dos modelos de linguagem
from modelo.modeloarguments import ModeloArgumentos
from modelo.modelomodulo import *
from spacynlp.spacymodulo import *

from medidor.medidorenum import *

# Definição dos parâmetros do Modelo para os cálculos das Medidas
model_args = ModeloArgumentos(
                                    max_seq_len=512,
                                    pretrained_model_name_or_path='neuralmind/bert-base-portuguese-cased', 
                                    modelo_spacy='pt_core_news_lg',
                                    versao_spacy='3.4.4',
                                    do_lower_case=False, # default True
                                    output_attentions=False, # default False
                                    output_hidden_states=True, # default False    
                                    estrategia_pooling=0, # 0 - MEAN estratégia média / 1 - MAX  estratégia maior
                                    palavra_relevante=0 # 0 - Considera todas as palavras das sentenças / 1 - Desconsidera as stopwords / 2 - Considera somente as palavras substantivas
                                    )

class ModeloLinguagem:
    
    ''' 
    Carrega e cria um modelo de Linguagem, que pode ser usado para gerar embeddings de tokens, palavras, sentenças e documentos.
     
    Parâmetros:
    `pretrained_model_name_or_path' - Se for um caminho de arquivo no disco, carrega o modelo a partir desse caminho. Se não for um caminho, ele primeiro tenta fazer o download de um modelo pré-treinado do modelo de linguagem. Se isso falhar, tenta construir um modelo do repositório de modelos do Huggingface com esse nome.
    ''' 
    
    # Construtor da classe
    def __init__(self, pretrained_model_name_or_path):
        # Parâmetro recebido para o modelo de linguagem
        model_args.pretrained_model_name_or_path = pretrained_model_name_or_path
        
        # Carrega o modelo e tokenizador do modelo de linguagem
        self.model, self.tokenizer = carregaModeloLinguagem(model_args)
        
        self.verificaCarregamentoSpacy()
    
    def verificaCarregamentoSpacy(self):
        ''' 
        Verifica se é necessário carregar o spacy.
        Utilizado para as estratégias de palavras relevantes CLEAN e NOUN.
        ''' 
        
        if model_args.palavra_relevante != PalavrasRelevantes.ALL.value:
            # Carrega o modelo spacy
            self.nlp = carregaSpacy(model_args)
            
        else:
            self.nlp = None
    
    def defineEstrategiaPooling(self, estrategiaPooling):
        ''' 
        Define a estratégia de pooling para os parâmetros do modelo.

        Parâmetros:
        `estrategiaPooling` - Estratégia de pooling das camadas do BERT.
        ''' 
        
        if estrategiaPooling == EstrategiasPooling.MAX.name:
            model_args.estrategia_pooling = EstrategiasPooling.MAX.value
            
        else:
            if estrategiaPooling == EstrategiasPooling.MEAN.name:
                model_args.estrategia_pooling = EstrategiasPooling.MEAN.value
            else:
                logging.info("Não foi especificado uma estratégia de pooling válida.") 

    def definePalavraRelevante(self, palavraRelevante):
        ''' 
        Define a estratégia de palavra relavante para os parâmetros do modelo.
        
        Parâmetros:        
        `palavraRelevante` - Estratégia de relevância das palavras do texto.               
        ''' 
        
        if palavraRelevante == PalavrasRelevantes.CLEAN.name:
            model_args.palavra_relevante = PalavrasRelevantes.CLEAN.value
            verificaCarregamentoSpacy()
            
        else:
            if palavraRelevante == PalavrasRelevantes.NOUN.name:
                model_args.palavra_relevante = PalavrasRelevantes.NOUN.value
                verificaCarregamentoSpacy()
                
            else:
                if palavraRelevante == PalavrasRelevantes.ALL.name:
                    model_args.palavra_relevante = PalavrasRelevantes.ALL.value
                    
                else:
                    logging.info("Não foi especificado uma estratégia de relevância de palavras do texto válida.") 

    def getMedidaCoerencia(self, texto, estrategiaPooling='MEAN', palavraRelevante='ALL'):
        ''' 
        Retorna as medidas de (in)coerência Ccos, Ceuc, Cman do texto.
        
        Parâmetros:
        `texto` - Um texto a ser medido a coerência.           
        `estrategiaPooling` - Estratégia de pooling das camadas do BERT.
        `palavraRelevante` - Estratégia de relevância das palavras do texto.            
        
        Retorno:
        `Ccos` - Medida de coerência Ccos do do texto.            
        `Ceuc` - Medida de incoerência Ceuc do do texto.            
        `Cman` - Medida de incoerência Cman do do texto.            
        ''' 

        self.defineEstrategiaPooling(estrategiaPooling)
        self.definePalavraRelevante(palavraRelevante)

        self.Ccos, self.Ceuc, self.Cman = getMedidasCoerenciaDocumento(texto, 
                                                                       modelo=self.model, 
                                                                       tokenizador=self.tokenizer, 
                                                                       nlp=self.nlp, 
                                                                       camada=listaTipoCamadas[4], 
                                                                       tipoDocumento='o', 
                                                                       estrategia_pooling=model_args.estrategia_pooling, 
                                                                       palavra_relevante=model_args.palavra_relevante)
          
        return self.Ccos, self.Ceuc, self.Cman
    
    def getMedidaCoerenciaCosseno(self, texto, estrategiaPooling='MEAN', palavraRelevante='ALL'):
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

        self.Ccos, self.Ceuc, self.Cman = getMedidasCoerenciaDocumento(texto, 
                                                                       modelo=self.model, 
                                                                       tokenizador=self.tokenizer, 
                                                                       nlp=self.nlp, 
                                                                       camada=listaTipoCamadas[4], 
                                                                       tipoDocumento='o', 
                                                                       estrategia_pooling=model_args.estrategia_pooling, 
                                                                       palavra_relevante=model_args.palavra_relevante)
          
        return self.Ccos
    
    def getMedidaCoerenciaEuclediana(self, texto, estrategiaPooling='MEAN', palavraRelevante='ALL'):
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

        self.Ccos, self.Ceuc, self.Cman = getMedidasCoerenciaDocumento(texto, 
                                                                       modelo=self.model, 
                                                                       tokenizador=self.tokenizer, 
                                                                       nlp=self.nlp, 
                                                                       camada=listaTipoCamadas[4], 
                                                                       tipoDocumento='o', 
                                                                       estrategia_pooling=model_args.estrategia_pooling, 
                                                                       palavra_relevante=model_args.palavra_relevante)
          
        return self.Ceuc        
    
    def getMedidaCoerenciaManhattan(self, texto, estrategiaPooling='MEAN', palavraRelevante='ALL'):
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
        
        self.Ccos, self.Ceuc, self.Cman = getMedidasCoerenciaDocumento(texto, 
                                                                       modelo=self.model, 
                                                                       tokenizador=self.tokenizer, 
                                                                       nlp=self.nlp, 
                                                                       camada=listaTipoCamadas[4], 
                                                                       tipoDocumento='o', 
                                                                       estrategia_pooling=model_args.estrategia_pooling, 
                                                                       palavra_relevante=model_args.palavra_relevante)
          
        return self.Cman                
