# Import das bibliotecas.

# Biblioteca de logging
import logging 

# Biblioteca próprias
from textotransformer.pln.pln import PLN
from textotransformer.mensurador.mensurador import Mensurador
from textotransformer.modelo.transformer import Transformer
from textotransformer.modelo.modeloarguments import ModeloArgumentos
from textotransformer.modelo.modeloenum import LISTATIPOCAMADA_NOME, EstrategiasPooling, listaTipoCamadas
from textotransformer.modelo.modeloenum import EstrategiasPooling
from textotransformer.mensurador.mensuradorenum import PalavrasRelevantes 

logger = logging.getLogger(__name__)

# Definição dos parâmetros do Modelo para os cálculos das Medidas
modelo_argumentos = ModeloArgumentos(
        max_seq_len=512,
        pretrained_model_name_or_path="neuralmind/bert-base-portuguese-cased", # Nome do modelo de linguagem pré-treinado Transformer
        modelo_spacy="pt_core_news_lg", # Nome do modelo de linguagem da ferramenta de PLN
        do_lower_case=False,            # default True
        output_attentions=False,        # default False
        output_hidden_states=True,      # default False  /Retornar os embeddings das camadas ocultas  
        camadas_embeddings = 2,         # 0-Primeira/1-Penúltima/2-Ùltima/3-Soma 4 últimas/4-Concat 4 últiamas/5-Todas
        estrategia_pooling=0,           # 0 - MEAN estratégia média / 1 - MAX  estratégia maior
        palavra_relevante=0             # 0 - Considera todas as palavras das sentenças / 1 - Desconsidera as stopwords / 2 - Considera somente as palavras substantivas
        )

class TextoTransformer:
    
    ''' 
    Carrega e cria um modelo de Linguagem, que pode ser usado para gerar embeddings de tokens, palavras, sentenças e textos.
     
    Parâmetros:
    `pretrained_model_name_or_path` - Se for um caminho de arquivo no disco, carrega o modelo a partir desse caminho. Se não for um caminho, ele primeiro faz o download do repositório de modelos do Huggingface com esse nome. Valor default: 'neuralmind/bert-base-portuguese-cased'.                  
    `modelo_spacy` - Nome do modelo a ser instalado e carregado pela ferramenta de pln spaCy. Valor default 'pt_core_news_lg'.                       
    `camadas_embeddings` - Especifica de qual camada ou camadas será recuperado os embeddings do transformer. Valor defaul '2'. Valores possíveis: 0-Primeira/1-Penúltima/2-Ùltima/3-Soma 4 últimas/4-Concat 4 últiamas/5-Todas.       
    ''' 
    
    # Construtor da classe
    def __init__(self, pretrained_model_name_or_path="neuralmind/bert-base-portuguese-cased", 
                       modelo_spacy="pt_core_news_lg",
                       camadas_embeddings=2):
                       
        # Parâmetro recebido para o modelo de linguagem
        modelo_argumentos.pretrained_model_name_or_path = pretrained_model_name_or_path
               
        # Parâmetro recebido para o modelo da ferramenta de pln
        modelo_argumentos.modelo_spacy = modelo_spacy
                
        # Carrega o modelo de linguagem da classe transformador
        self.transformer_model = Transformer(modelo_args=modelo_argumentos)
    
        # Recupera o modelo.
        self.model = self.transformer_model.get_auto_model()
    
        # Recupera o tokenizador.     
        self.tokenizer = self.transformer_model.get_tokenizer()
        
        # Especifica de qual camada utilizar os embeddings        
        logger.info("Utilizando embeddings do modelo da {} camada(s).".format(listaTipoCamadas[modelo_argumentos.camadas_embeddings][LISTATIPOCAMADA_NOME]))
                    
        # Especifica camadas para recuperar os embeddings
        modelo_argumentos.camadas_embeddings = camadas_embeddings
      
        # Carrega o spaCy
        self.pln = PLN(modelo_args=modelo_argumentos)
                        
        # Constroi um mensurador
        self.mensurador = Mensurador(modelo_args=modelo_argumentos, 
                                     transformer_model=self.transformer_model, 
                                     pln=self.pln)        
    
        logger.info("Classe ModeloLinguagem carregada: {}.".format(modelo_argumentos))
    
    # ============================
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
    
    # ============================
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

    # ============================
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
    
    # ============================
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
    
    # ============================
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
       
    # ============================
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
    
    # ============================
    def tokenize(self, texto):
        return self.get_transformer_model().tokenize(texto)
    
    # ============================
    def getEmbeddings(self, texto):
        '''
        De um texto preparado(tokenizado) ou não, retorna token_embeddings, input_ids, attention_mask, token_type_ids, 
        tokens_texto, texto_original  e all_layer_embeddings em um dicionário.
        
        Facilita acesso a classe Transformer.
    
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings do modelo de linguagem
    
        Retorna um dicionário com:            
            token_embeddings uma lista com os embeddings da última camada
            input_ids uma lista com os textos indexados.            
            attention_mask uma lista com os as máscaras de atenção
            token_type_ids uma lista com os tipos dos tokens.            
            tokens_texto uma lista com os textos tokenizados com os tokens especiais.
            texto_original uma lista com os textos originais.
            all_layer_embeddings uma lista com os embeddings de todas as camadas.
        '''
        return self.get_transformer_model().getEmbeddings(texto)

    # ============================
    def getEmbeddingsPalavras(self, 
                              texto):
        
        '''
        De um texto preparado(tokenizado) ou não, retorna os embeddings das palavras do texto. 
        Retorna um dicionário 5 listas, os tokens(palavras), as postagging, tokens OOV, e os embeddings dos tokens igualando a quantidade de tokens do spaCy com a tokenização do MCL de acordo com a estratégia. 
        Utiliza duas estratégias para realizar o pooling de tokens que forma uma palavra.
            - Estratégia MEAN para calcular a média dos embeddings dos tokens que formam uma palavra.
            - Estratégia MAX para calcular o valor máximo dos embeddings dos tokens que formam uma palavra.
            
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings das palavras do modelo de linguagem
    
        Retorna um dicionário com:    
            tokens_texto uma lista com os tokens(palavras) realizados pelo método.
            tokens_texto_mcl uma lista com os tokens e tokens especiais realizados pelo mcl.
            tokens_oov_texto_mcl uma lista com os tokens OOV(com ##) do mcl.
            tokens_texto_pln uma lista com os tokens realizados pela ferramenta de pln(spaCy).
            pos_texto_pln uma lista com as postagging dos tokens realizados pela ferramenta de pln(spaCy).            
            embeddings_MEAN uma lista com os embeddings com a estratégia MEAN
            embeddings_MAX uma lista com os embeddings com a estratégia MAX
        '''
        
        # Tokeniza o texto
        texto_embeddings = self.get_transformer_model().getEmbeddings(texto)

        # Acumula a saída do método
        saida = {}
        saida.update({'tokens_texto': [], 
                      'tokens_texto_mcl' : [],
                      'tokens_oov_texto_mcl': [],                      
                      'tokens_texto_pln' : [],
                      'pos_texto_pln': [],
                      'embeddings_MEAN': [],        
                      'embeddings_MAX': []
                     }
        )

        # Percorre os textos da lista.
        for i, sentenca in enumerate(texto_embeddings['tokens_texto_mcl']):
            # Recupera o texto tokenizado pela ferramenta de pln
            lista_tokens_texto_pln = self.get_pln().getTokensTexto(texto[i])
            # Recupera os embeddings do texto  
            embeddings_texto = texto_embeddings['token_embeddings'][i][0:len(texto_embeddings['tokens_texto_mcl'][i])]
            #print(len(embeddings))
            # Recupera a lista de tokens do tokenizado pelo MCL sem CLS e SEP
            tokens_texto_mcl = texto_embeddings['tokens_texto_mcl'][i][1:-1]
            #print(tokens_texto)
            # Concatena os tokens gerandos pela ferramenta de pln
            tokens_texto_concatenado = " ".join(lista_tokens_texto_pln)
            #print(tokens_texto_concatenado)
            lista_tokens_texto, lista_pos_texto_pln, lista_tokens_oov_texto_mcl, lista_embeddings_MEAN, lista_embeddings_MAX = self.get_transformer_model().getTokensEmbeddingsPOSTexto(
                                                    embeddings_texto,
                                                    tokens_texto_mcl,
                                                    tokens_texto_concatenado,
                                                    self.get_pln())

            #Acumula a saída do método 
            saida['tokens_texto'].append(lista_tokens_texto)
            saida['tokens_texto_mcl'].append(tokens_texto_mcl)
            saida['tokens_oov_texto_mcl'].append(lista_tokens_oov_texto_mcl)            
            saida['tokens_texto_pln'].append(lista_tokens_texto_pln)
            saida['pos_texto_pln'].append(lista_pos_texto_pln)            
            saida['embeddings_MEAN'].append(lista_embeddings_MEAN)
            saida['embeddings_MAX'].append(lista_embeddings_MAX)

        return saida

    # ============================
    def get_model(self):
        return self.model

    # ============================
    def get_tokenizer(self):
        return self.tokenizer

    # ============================
    def get_transformer_model(self):
        return self.transformer_model

    # ============================    
    def get_mensurador(self):
        return self.mensurador        
        
    # ============================        
    def get_pln(self):
        return self.pln          
        