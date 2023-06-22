# Import das bibliotecas.

# Biblioteca de logging
import logging 
# Biblioteca de tipos
from typing import List, Union
# Biblioteca de aprendizado de máquina
import torch 
import numpy as np
# Biblioteca barra de progresso
from tqdm import trange

# Biblioteca próprias
from textotransformer.pln.pln import PLN
from textotransformer.mensurador.mensurador import Mensurador
from textotransformer.modelo.transformer import Transformer
from textotransformer.modelo.modeloarguments import ModeloArgumentos
from textotransformer.modelo.modeloenum import LISTATIPOCAMADA_NOME
from textotransformer.modelo.modeloenum import EstrategiasPooling
from textotransformer.modelo.modeloenum import listaTipoCamadas
from textotransformer.mensurador.mensuradorenum import PalavrasRelevantes
from textotransformer.util.utiltexto import encontrarIndiceSubLista

logger = logging.getLogger(__name__)

# Definição dos parâmetros do Modelo para os cálculos das Medidas
modelo_argumentos = ModeloArgumentos(
        max_seq_len=512,
        pretrained_model_name_or_path="neuralmind/bert-base-portuguese-cased", # Nome do modelo de linguagem pré-treinado Transformer
        modelo_spacy="pt_core_news_lg", # Nome do modelo de linguagem da ferramenta de PLN
        do_lower_case=False,            # default True
        output_attentions=False,        # default False
        output_hidden_states=True,      # default False  /Retornar os embeddings das camadas ocultas  
        camadas_embeddings=2,           # 0-Primeira/1-Penúltima/2-Ùltima/3-Soma 4 últimas/4-Concat 4 últiamas/5-Todas
        estrategia_pooling=0,           # 0 - MEAN estratégia média / 1 - MAX  estratégia maior
        palavra_relevante=0             # 0 - Considera todas as palavras das sentenças / 1 - Desconsidera as stopwords / 2 - Considera somente as palavras substantivas
        )

class TextoTransformer:
    
    ''' 
    Carrega e cria um objeto da classe TextoTransformer para manipular um modelo de linguagem baseado e transformer.
    Manipula embeddings de tokens, palavras, sentenças e textos.
     
    Parâmetros:
    `pretrained_model_name_or_path` - Se for um caminho de arquivo no disco, carrega o modelo a partir desse caminho. Se não for um caminho, ele primeiro faz o download do repositório de modelos do Huggingface com esse nome. Valor default: 'neuralmind/bert-base-portuguese-cased'.                  
    `modelo_spacy` - Nome do modelo a ser instalado e carregado pela ferramenta de pln spaCy. Valor default 'pt_core_news_lg'.                       
    `camadas_embeddings` - Especifica de qual camada ou camadas será recuperado os embeddings do transformer. Valor defaul '2'. Valores possíveis: 0-Primeira/1-Penúltima/2-Ùltima/3-Soma 4 últimas/4-Concat 4 últiamas/5-Todas.       
    `device` - Dispositivo (como 'cuda' / 'cpu') que deve ser usado para computação. Se none, verifica se uma GPU pode ser usada.
    ''' 
    
    # Construtor da classe
    def __init__(self, pretrained_model_name_or_path: str ="neuralmind/bert-base-portuguese-cased", 
                       modelo_spacy: str ="pt_core_news_lg",
                       camadas_embeddings: int = 2,
                       device = None):
                       
        # Parâmetro recebido para o modelo de linguagem
        modelo_argumentos.pretrained_model_name_or_path = pretrained_model_name_or_path
               
        # Parâmetro recebido para o modelo da ferramenta de pln
        modelo_argumentos.modelo_spacy = modelo_spacy
                
        # Carrega o modelo de linguagem da classe transformador
        self.transformer = Transformer(modelo_args=modelo_argumentos)
    
        # Recupera o modelo de linguagem.
        self.model = self.transformer.getAutoMmodel()
    
        # Recupera o tokenizador.     
        self.tokenizer = self.transformer.getTokenizer()
        
        # Especifica de qual camada utilizar os embeddings                
        logger.info("Utilizando embeddings do modelo da {} camada(s).".format(listaTipoCamadas[modelo_argumentos.camadas_embeddings][LISTATIPOCAMADA_NOME]))
                    
        # Especifica camadas para recuperar os embeddings
        modelo_argumentos.camadas_embeddings = camadas_embeddings
      
        # Carrega o spaCy
        self.pln = PLN(modelo_args=modelo_argumentos)
                        
        # Constroi um mensurador
        self.mensurador = Mensurador(modelo_args=modelo_argumentos, 
                                     transformer=self.transformer, 
                                     pln=self.pln)        
    
        # Verifica se é possível usar GPU
        if device is None:
            if torch.cuda.is_available():    
                device = "cuda"
                logger.info("Existem {} GPU(s) disponíveis.".format(torch.cuda.device_count()))
                logger.info("Iremos usar a GPU: {}.".format(torch.cuda.get_device_name(0)))

            else:                
                device = "cpu"
                logger.info("Sem GPU disponível, usando CPU.")
            
            # Diz ao PyTorch para usar o dispositvo (GPU ou CPU)
            self._target_device = torch.device(device)
        else:
            # Usa o device informado
            self._target_device = torch.device(device)

        # Mensagem de carregamento da classe
        logger.info("Classe TextoTransformer carregada: {}.".format(modelo_argumentos))
    
    # ============================   
    def __repr__(self):
        '''
        Retorna uma string com descrição do objeto.
        '''
        
        return "Classe ({}) com Transformer {} carregada com o modelo {} e NLP {} carregado com o modelo {} ".format(self.__class__.__name__, 
                                                                                                            self.getTransformer().auto_model.__class__.__name__,
                                                                                                            modelo_argumentos.pretrained_model_name_or_path,
                                                                                                            self.getPln().model_pln.__class__.__name__,
                                                                                                            modelo_argumentos.modelo_spacy)
    
    # ============================
    def _defineEstrategiaPooling(self, estrategia_pooling: Union[int, EstrategiasPooling] = EstrategiasPooling.MEAN):
        ''' 
        Define a estratégia de pooling para os parâmetros do modelo.

        Parâmetros:
        `estrategia_pooling` - Um número de 0 a 1 com a estratégia de pooling das camadas do modelo contextualizado. Valor defaul '0'. Valores possíveis: 0 - MEAN estratégia média / 1 - MAX  estratégia maior.
        ''' 
        
        # Verifica o tipo de dado do parâmetro 'estrategia_pooling'
        if isinstance(estrategia_pooling, int):
          if estrategia_pooling == 0:
            estrategia_pooling = EstrategiasPooling.MEAN
          else:
            if estrategia_pooling == 1:
              estrategia_pooling = EstrategiasPooling.MAX
            else:
              estrategia_pooling = None
              logger.info("Não foi especificado um valor inteiro para a estratégia de pooling.") 
        
        # Atribui para os parâmetros do modelo
        if estrategia_pooling == EstrategiasPooling.MEAN:
            modelo_argumentos.estrategia_pooling = EstrategiasPooling.MEAN.value
        else:
            if estrategia_pooling == EstrategiasPooling.MAX:
                modelo_argumentos.estrategia_pooling = EstrategiasPooling.MAX.value            
            else:
                logger.info("Não foi especificado uma estratégia de pooling válida.") 
    
    # ============================
    def _definePalavraRelevante(self, palavra_relevante: Union[int, PalavrasRelevantes] = PalavrasRelevantes.ALL):
        ''' 
        Define a estratégia de palavra relavante para os parâmetros do modelo.
        
        Parâmetros:        
        `palavra_relevante` - Um número de 0 a 2 que indica a estratégia de relevância das palavras do texto. Valor defaul '0'. Valores possíveis: 0 - Considera todas as palavras das sentenças / 1 - Desconsidera as stopwords / 2 - Considera somente as palavras substantivas.
        ''' 
        
        # Verifica o tipo de dado do parâmetro 'palavra_relevante'
        if isinstance(palavra_relevante, int):
          if palavra_relevante == 0:
            palavra_relevante = PalavrasRelevantes.ALL
          else:
            if palavra_relevante == 1:
              palavra_relevante = PalavrasRelevantes.CLEAN
            else:
                if palavra_relevante == 2:
                    palavra_relevante = PalavrasRelevantes.NOUN
                else:
                    palavra_relevante = None
                    logger.info("Não foi especificado um valor inteiro para a estratégia de relevância de palavra.") 
        
        # Atribui para os parâmetros do modelo
        if palavra_relevante == PalavrasRelevantes.ALL:
            modelo_argumentos.palavra_relevante = PalavrasRelevantes.ALL.value            
        else:
            if palavra_relevante == PalavrasRelevantes.CLEAN:
                modelo_argumentos.palavra_relevante = PalavrasRelevantes.CLEAN.value                
            else:
                if palavra_relevante == PalavrasRelevantes.NOUN:
                    modelo_argumentos.palavra_relevante = PalavrasRelevantes.NOUN.value                    
                else:
                    logger.info("Não foi especificado uma estratégia de relevância de palavras do texto válida.") 

    # ============================
    def getMedidasTexto(self, texto: str, 
                        estrategia_pooling: EstrategiasPooling = EstrategiasPooling.MEAN, 
                        palavra_relevante: PalavrasRelevantes = PalavrasRelevantes.ALL):
        ''' 
        Retorna as medidas de (in)coerência Ccos, Ceuc, Cman do texto.
        
        Parâmetros:
        `texto` - Um texto a ser medido.           
        `estrategia_pooling` - Estratégia de pooling das camadas do BERT.
        `palavra_relevante` - Estratégia de relevância das palavras do texto.            
        
        Retorno um dicionário com:
        `cos` - Medida de cos do do texto.
        `euc` - Medida de euc do do texto.
        `man` - Medida de man do do texto.
        ''' 

        self._defineEstrategiaPooling(estrategia_pooling)
        self._definePalavraRelevante(palavra_relevante)
        
        saida = self.mensurador.getMedidasComparacaoTexto(texto, 
                                                          camada=modelo_argumentos.camadas_embeddings, 
                                                          tipo_texto='o')
          
        return saida
    
    # ============================
    def getMedidasTextoCosseno(self, texto: str, 
                               estrategia_pooling: EstrategiasPooling = EstrategiasPooling.MEAN, 
                               palavra_relevante: PalavrasRelevantes = PalavrasRelevantes.ALL):
        ''' 
        Retorna a medida do texto utilizando a medida de similaridade de cosseno.
        
        Parâmetros:
        `texto` - Um texto a ser medido a coerência.           
        `estrategia_pooling` - Estratégia de pooling das camadas do BERT. 
        `palavra_relevante` - Estratégia de relevância das palavras do texto.            
        
        Retorno:
        `cos` - Medida de cos do do texto.            
        '''         
        
        self._defineEstrategiaPooling(estrategia_pooling)
        self._definePalavraRelevante(palavra_relevante)
        
        saida = self.mensurador.getMedidasComparacaoTexto(texto, 
                                                          camada=modelo_argumentos.camadas_embeddings, 
                                                          tipo_texto='o')
          
        return saida['cos']
    
    # ============================
    def getMedidasTextoEuclediana(self, texto: str, 
                                  estrategia_pooling: EstrategiasPooling = EstrategiasPooling.MEAN, 
                                  palavra_relevante: PalavrasRelevantes = PalavrasRelevantes.ALL):
        ''' 
        Retorna a medida do texto utilizando a medida de distância de Euclidiana.
                 
        Parâmetros:
        `texto` - Um texto a ser mensurado.           
        `estrategia_pooling` - Estratégia de pooling das camadas do BERT.
        `palavra_relevante` - Estratégia de relevância das palavras do texto.            
        
        Retorno:
        `ceu` - Medida euc do texto.            
        ''' 
        
        self._defineEstrategiaPooling(estrategia_pooling)
        self._definePalavraRelevante(palavra_relevante)

        saida = self.mensurador.getMedidasComparacaoTexto(texto,
                                                          camada=modelo_argumentos.camadas_embeddings, 
                                                          tipo_texto='o')
          
        return saida['euc']      
       
    # ============================
    def getMedidasTextoManhattan(self, texto: str, 
                                 estrategia_pooling: EstrategiasPooling = EstrategiasPooling.MEAN, 
                                 palavra_relevante: PalavrasRelevantes = PalavrasRelevantes.ALL):
        ''' 
        Retorna a medida do texto utilizando a medida de distância de Manhattan.
                 
        Parâmetros:
        `texto` - Um texto a ser mensurado.           
        `estrategia_pooling` - Estratégia de pooling das camadas do BERT.
        `palavra_relevante` - Estratégia de relevância das palavras do texto.            
        
        Retorno:
        `man` - Medida  Cman do do texto.            
        ''' 
        
        self._defineEstrategiaPooling(estrategia_pooling)
        self._definePalavraRelevante(palavra_relevante)
        
        saida = self.mensurador.getMedidasComparacaoTexto(texto, 
                                                          camada=modelo_argumentos.camadas_embeddings, 
                                                          tipo_texto='o')
          
        return saida['man']
    
    # ============================
    def tokenize(self, texto: Union[str, List[str]]):
        '''
        Tokeniza um texto para submeter ao modelo de linguagem. 
        Retorna um dicionário listas de mesmo tamanho para garantir o processamento em lote.
        Use a quantidade de tokens para saber até onde deve ser recuperado em uma lista de saída.
        Ou use attention_mask diferente de 1 para saber que posições devem ser utilizadas na lista.

        Facilita acesso a classe Transformer.    

        Parâmetros:
        `texto` - Texto a ser tokenizado para o modelo de linguagem.
         
        Retorna um dicionário com:
            tokens_texto_mcl uma lista com os textos tokenizados com os tokens especiais.
            input_ids uma lista com os ids dos tokens de entrada mapeados em seus índices do vocabuário.
            token_type_ids uma lista com os tipos dos tokens.
            attention_mask uma lista com os as máscaras de atenção indicando com '1' os tokens  pertencentes à sentença.
        '''
        return self.getTransformer().tokenize(texto)
    
    # ============================
    def getCodificacaoCompleta(self, texto: Union[str, List[str]],
                               tamanho_lote: int = 32, 
                               mostra_barra_progresso: bool = False,
                               convert_to_numpy: bool = False,
                               device: str = None):

        '''
        Retorna um dicionário com as informações completas da codificação do texto utilizando o modelo de linguagem.
    
        Parâmetros:
         `texto` - Um texto a ser recuperado os embeddings do modelo de linguagem
         `tamanho_lote` - o tamanho do lote usado para o computação
         `mostra_barra_progresso` - Mostra uma barra de progresso ao codificar o texto.
         `convert_to_numpy` - Se verdadeiro, a saída é uma lista de vetores numpy. Caso contrário, é uma lista de tensores pytorch.        
         `device` - Qual torch.device usar para o computação.
    
        Retorno um dicionário com:            
            token_embeddings uma lista com os embeddings da última camada
            input_ids uma lista com os textos indexados.            
            attention_mask uma lista com os as máscaras de atenção
            token_type_ids uma lista com os tipos dos tokens.            
            tokens_texto_mcl uma lista com os textos tokenizados com os tokens especiais.
            texto_original uma lista com os textos originais.
            all_layer_embeddings uma lista com os embeddings de todas as camadas.
        '''
        #Coloca o modelo e modo avaliação
        self.model.eval()

        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'): 
            #Colocar uma texto individual em uma lista com comprimento 1
            texto = [texto]
            entrada_eh_string = True

        # Se não foi especificado um dispositivo, use-o defaul
        if device is None:
            device = self._target_device

        # Adiciona um dispositivo ao modelo
        self.model.to(device)

        # Dicionário com a saída
        saida = {}
        saida.update({'token_embeddings': [],                        
                      'input_ids': [],
                      'attention_mask': [],
                      'token_type_ids': [],        
                      'tokens_texto_mcl': [],
                      'texto_original': [],
                      'all_layer_embeddings': []
                      }
                     )

        # Percorre os lotes
        for start_index in trange(0, 
                                  len(texto), 
                                  tamanho_lote, 
                                  desc="Lotes", 
                                  disable=not mostra_barra_progresso):
            
            # Recupera um lote
            lote_textos = texto[start_index:start_index+tamanho_lote]

            # Tokeniza o lote usando o modelo
            lote_textos_tokenizados = self.getTransformer().tokenize(lote_textos)

            # Adiciona ao device gpu ou cpu
            lote_textos_tokenizados = self.getTransformer().batchToDevice(lote_textos_tokenizados, device)
            
            # Recupera os embeddings do modelo
            with torch.no_grad():

                # Recupera a saída da rede
                output_rede = self.getTransformer().getSaidaRede(lote_textos_tokenizados)

                # Lista para os embeddings do texto
                embeddings = []

                # Percorre todas as saídas(textos) do lote
                #for token_embeddings, attention_mask in zip(output_rede['token_embeddings'], output_rede['attention_mask']):                    
                for token_embeddings, input_ids, attention_mask,token_type_ids, tokens_texto_mcl,texto_original,all_layer_embeddings in zip(output_rede['token_embeddings'], 
                                              output_rede['input_ids'],
                                              output_rede['attention_mask'],
                                              output_rede['token_type_ids'],
                                              output_rede['tokens_texto_mcl'],
                                              output_rede['texto_original'],
                                              output_rede['all_layer_embeddings']):

                    ultimo_mask_id = len(attention_mask)-1
                    
                    # Localiza o último token de "attention_mask" igual a 1
                    while ultimo_mask_id > 0 and attention_mask[ultimo_mask_id].item() == 0:                        
                      ultimo_mask_id -= 1                        
                    
                    # Recupera os embeddings do primeiro(0) até o último token que "attention_mask" seja 1                        

                    # Concatena a lista dos embeddings do texto a lista já existente
                    saida['token_embeddings'].append(token_embeddings[0:ultimo_mask_id+1])
                    saida['input_ids'].append(input_ids[0:ultimo_mask_id+1])
                    saida['attention_mask'].append(attention_mask[0:ultimo_mask_id+1])                    
                    saida['token_type_ids'].append(token_type_ids[0:ultimo_mask_id+1])
                    saida['tokens_texto_mcl'].append(tokens_texto_mcl[0:ultimo_mask_id+1])
                    saida['texto_original'].append(texto_original)
                    saida['all_layer_embeddings'].append(all_layer_embeddings[0:ultimo_mask_id+1])
        
        if convert_to_numpy:
          saida['token_embeddings'] = np.array([emb.numpy() for emb in saida['token_embeddings']],dtype=object)

        # Se é uma string remove a lista de lista
        if entrada_eh_string:
            saida['token_embeddings'] = saida['token_embeddings'][0]
            saida['input_ids'] = saida['input_ids'][0]
            saida['attention_mask'] = saida['attention_mask'][0]
            saida['token_type_ids'] = saida['token_type_ids'][0]
            saida['tokens_texto_mcl'] = saida['tokens_texto_mcl'][0]
            saida['texto_original'] = saida['texto_original'][0]
            saida['all_layer_embeddings'] = saida['all_layer_embeddings'][0]

        return saida

    # ============================
    def getCodificacao(self, texto: Union[str, List[str]],
                       tamanho_lote: int = 32,
                       mostra_barra_progresso: bool = False,
                       convert_to_numpy: bool = False,
                       device: str = None):

        '''
        Retorna os embeddings da codificação de um texto utilizando o modelo de linguagem.
    
        Parâmetros:
         `texto` - Um texto a ser recuperado os embeddings do modelo de linguagem
         `tamanho_lote` - o tamanho do lote usado para o computação
         `mostra_barra_progresso` - Mostra uma barra de progresso ao codificar o texto.
         `convert_to_numpy` - Se verdadeiro, a saída é uma lista de vetores numpy. Caso contrário, é uma lista de tensores pytorch.        
         `device` - Qual torch.device usar para o computação.
    
        Retorno:
         Uma lista de tensores é com os embeddings do texto. Se convert_to_numpy, uma matriz numpy é retornada.
        '''
        #Coloca o modelo e modo avaliação
        self.model.eval()

        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'): 
            #Colocar uma texto individual em uma lista com comprimento 1
            texto = [texto]
            entrada_eh_string = True

        # Se não foi especificado um dispositivo, use-o defaul
        if device is None:
            device = self._target_device

        # Adiciona um dispositivo ao modelo
        self.model.to(device)

        # Lista com embeddings de saída
        all_embeddings = []

        # Ordena o texto pelo comprimento decrescente
        indice_tamanho_ordenado = np.argsort([-self._tamanho_texto(sen) for sen in texto])        
        # Ordena o texto pelo comprimento decrescente
        textos_ordenados = [texto[idx] for idx in indice_tamanho_ordenado]
        
        # Percorre os lotes
        for start_index in trange(0, 
                                  len(texto), 
                                  tamanho_lote, 
                                  desc="Lotes", 
                                  disable=not mostra_barra_progresso):
            # Recupera um lote
            lote_textos = textos_ordenados[start_index:start_index+tamanho_lote]

            # Tokeniza o lote
            lote_textos_tokenizados = self.getTransformer().tokenize(lote_textos)
            # Adiciona ao device gpu ou cpu
            lote_textos_tokenizados = self.getTransformer().batchToDevice(lote_textos_tokenizados, device)
            
            # Recupera os embeddings do modelo
            with torch.no_grad():

                # Recupera a saída da rede
                output_rede = self.getTransformer().getSaidaRede(lote_textos_tokenizados)

                embeddings = []
                # Percorre todas as saídas(textos) do lote
                for token_emb, attention in zip(output_rede['token_embeddings'], output_rede['attention_mask']):                    
                    ultimo_mask_id = len(attention)-1
                    
                    # Localiza o último token de "attention_mask" igual a 1
                    while ultimo_mask_id > 0 and attention[ultimo_mask_id].item() == 0:                        
                      ultimo_mask_id -= 1                        
                    
                    # Adiciona os embeddings do primeiro(0) até o último token que "attention_mask" seja 1                        
                    embeddings.append(token_emb[0:ultimo_mask_id+1])
                                   
                # Concatena a nova lista a lista já existente
                all_embeddings.extend(embeddings)
        
        # Reorganiza a lista
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(indice_tamanho_ordenado)]

        if convert_to_numpy:
          all_embeddings = np.array([emb.numpy() for emb in all_embeddings],dtype=object)

        # Se é uma string remove a lista de lista
        if entrada_eh_string:
          all_embeddings = all_embeddings[0]

        return all_embeddings

    # ============================
    def getSaidaRede(self, texto: Union[str, dict]):
        '''
        De um texto preparado(tokenizado) ou não, retorna token_embeddings, input_ids, attention_mask, token_type_ids, 
        tokens_texto, texto_original  e all_layer_embeddings em um dicionário.
        
        Facilita acesso ao método "getSaidaRede" da classe Transformer.
    
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
        
        return self.getTransformer().getSaidaRede(texto)

   # ============================
    def getEmbeddingTexto(self, texto: Union[str, List[str]], 
                          estrategia_pooling: Union[int, EstrategiasPooling] = EstrategiasPooling.MEAN):
        '''
        Recebe um texto (string ou uma lista de strings) e retorna os embeddings consolidados dos tokens do texto utilizando estratégia pooling especificada(MEAN, MAX).
            
        Parâmetros:
        `texto` - Um texto é uma string ou uma lista de strings.
        `estrategia_pooling` - Valor default MEAN. Uma estratégia de pooling,(EstrategiasPooling.MEAN, EstrategiasPooling.MAX). Pode ser utilizado os valores inteiros 0 para MEAN e 1 para MAX.
    
        Retorno: 
        Os embeddings consolidados do texto se o parâmetro texto é uma string, caso contrário uma lista com os embeddings consolidados se o parâmetro é lista de string.
        '''

        # Verifica o tipo da estratégia de pooling
        if isinstance(estrategia_pooling, int):
          if estrategia_pooling == 0:
            estrategia_pooling = EstrategiasPooling.MEAN
          else:
            if estrategia_pooling == 1:
              estrategia_pooling = EstrategiasPooling.MAX
            else:
              estrategia_pooling = None
              logger.info("Não foi especificado um valor inteiro para a estratégia de pooling.") 

        # Retorna os embeddings de acordo com a estratégia
        if estrategia_pooling == EstrategiasPooling.MEAN:
            return self.getCodificacaoTexto(texto)['texto_embeddings_MEAN']
        else:
            if estrategia_pooling == EstrategiasPooling.MAX:
                return self.getCodificacaoTexto(texto)['texto_embeddings_MAX']
            else:              
                logger.info("Não foi especificado uma estratégia de pooling válida.") 
                return None  

    # ============================
    def getCodificacaoTexto(self, texto: Union[str, List[str]],
                            tamanho_lote: int = 32, 
                            mostra_barra_progresso: bool = False,                     
                            convert_to_numpy: bool = False,         
                            device: str = None):
        '''        
        De um texto preparado(tokenizado) ou não, retorna a codificação dos textos consolidados dos tokens do textos utilizando estratégia pooling MEAN e MAX.
    
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings dos textos consolidados dos tokens com a estratégia MEAN e MAX utilizando o modelo de linguagem
    
        Retorno:
        Uma lista com os embeddings consolidados dos textos utilizando embeddings da última camada do transformer.
        '''

        # Se o texto é uma string, coloca em uma lista de comprimento 1
        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'):             
            texto = [texto]
            entrada_eh_string = True

        # Recupera os embeddings do texto
        texto_embeddings = self.getCodificacaoCompleta(texto,
                                                       tamanho_lote,
                                                       mostra_barra_progresso,
                                                       convert_to_numpy,
                                                       device)

        # Acumula a saída do método
        saida = {}
        saida.update({'texto_original' : [],            # Lista com os textos originais
                      'tokens_texto_mcl' : [],          # Lista com os tokens dos textos originais
                      'texto_embeddings_MEAN': [],      # Lista de lista média dos embeddings dos tokens que da sentença.
                      'texto_embeddings_MAX': [],       # Lista de lista máximo dos embeddings dos tokens que da sentença.
                      #'all_layer_embeddings': []
                     }
        )

        # Percorre os textos da lista.
        for i, texto in enumerate(texto_embeddings['texto_original']):       

            # Recupera a lista de embeddings gerados pelo MCL sem CLS e SEP 
            embeddings_texto = texto_embeddings['token_embeddings'][i][1:-1]
           
            # Recupera a lista de tokens do tokenizado pelo MCL sem CLS e SEP
            tokens_texto_mcl = texto_embeddings['tokens_texto_mcl'][i][1:-1]
            
            # Calcula a média dos embeddings dos tokens das sentenças do texto
            embedding_documento_media = torch.mean(embeddings_texto, dim=0)

            # Calcula a média dos embeddings dos tokens das sentenças do texto
            embedding_documento_maximo, linha = torch.max(embeddings_texto, dim=0)
            
            #Acumula a saída do método 
            #Se é uma string uma lista com comprimento 1
            if entrada_eh_string:
                saida['texto_original'] = texto_embeddings['texto_original'][i]
                saida['tokens_texto_mcl'] =  tokens_texto_mcl
                saida['texto_embeddings_MEAN'] = embedding_documento_media
                saida['texto_embeddings_MAX'] = embedding_documento_maximo
            else:
                saida['texto_original'].append(texto_embeddings['texto_original'][i])
                saida['tokens_texto_mcl'].append(tokens_texto_mcl)
                saida['texto_embeddings_MEAN'].append(embedding_documento_media)
                saida['texto_embeddings_MAX'].append(embedding_documento_maximo)

        return saida
    
    # ============================
    def getEmbeddingSentenca(self, texto: Union[str, List[str]], 
                             estrategia_pooling: Union[int, EstrategiasPooling] = EstrategiasPooling.MEAN):
        '''
        Recebe um texto (string ou uma lista de strings) e retorna os embeddings consolidados dos tokens das sentenças do texto utilizando estratégia pooling especificada(MEAN, MAX).          
        O texto ou a lista de textos é sentenciado utilizando a ferramenta de PLN. 
        Os embeddings dos tokens gerados pelo modelo de linguagem serão consolidados em sentenças geradas pela sentenciação da ferramenta de pln.
            
        Parâmetros:
        `texto` - Um texto é uma string ou uma lista de strings.
        `estrategia_pooling` - Valor default MEAN. Uma estratégia de pooling,(EstrategiasPooling.MEAN, EstrategiasPooling.MAX). Pode ser utilizado os valores inteiros 0 para MEAN e 1 para MAX.
    
        Retorno: 
        Uma lista com os embeddings consolidados das sentenças se o parâmetro texto é uma string, caso contrário uma lista com a lista dos embeddings consolidados das sentenças se o parâmetro é lista de string.
        '''

        # Verifica o tipo da estratégia de pooling
        if isinstance(estrategia_pooling, int):
          if estrategia_pooling == 0:
            estrategia_pooling = EstrategiasPooling.MEAN
          else:
            if estrategia_pooling == 1:
              estrategia_pooling = EstrategiasPooling.MAX
            else:
              estrategia_pooling = None
              logger.info("Não foi especificado um valor inteiro para a estratégia de pooling.") 

        # Retorna os embeddings de acordo com a estratégia
        if estrategia_pooling == EstrategiasPooling.MEAN:
            return self.getCodificacaoSentenca(texto)['sentenca_embeddings_MEAN']
        else:
            if estrategia_pooling == EstrategiasPooling.MAX:
                return self.getCodificacaoSentenca(texto)['sentenca_embeddings_MAX']
            else:              
                logger.info("Não foi especificado uma estratégia de pooling válida.") 
                return None

    # ============================    
    def getCodificacaoSentenca(self, texto: Union[str, List[str]],
                               tamanho_lote: int = 32, 
                               mostra_barra_progresso: bool = False,                     
                               convert_to_numpy: bool = False,         
                               device: str = None):      
        '''        
        De um texto preparado(tokenizado) ou não, retorna a codificação das sentenças consolidados dos tokens do textos utilizando estratégia pooling MEAN e MAX.
    
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings das sentenças consolidados dos tokens com a estratégia MEAN e MAX utilizando o modelo de linguagem
    
        Retorno:
        Uma lista com os embeddings consolidados das sentenças utilizando embeddings da última camada do transformer.
        '''

        # Se o texto é uma string, coloca em uma lista de comprimento 1
        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'):             
            texto = [texto]
            entrada_eh_string = True
        
        # Recupera os embeddings do texto
        texto_embeddings = self.getCodificacaoCompleta(texto,
                                               tamanho_lote,
                                               mostra_barra_progresso,
                                               convert_to_numpy,
                                               device)

        # Acumula a saída do método
        saida = {}
        saida.update({'texto_original' : [],               # Lista com os textos originais
                      'tokens_texto_mcl' : [],             # Lista com os tokens dos textos originais
                      'sentencas_texto' : [],              # Lista com as sentenças do texto
                      'sentenca_embeddings_MEAN': [],      # Lista de lista média dos embeddings dos tokens que da sentença.
                      'sentenca_embeddings_MAX': [],       # Lista de lista máximo dos embeddings dos tokens que da sentença.
                     }
        )

        # Percorre os textos da lista.
        for i, texto in enumerate(texto_embeddings['texto_original']):       

            # Recupera a lista de embeddings gerados pelo MCL sem CLS e SEP 
            embeddings_texto = texto_embeddings['token_embeddings'][i][1:-1]

            # Recupera a lista de tokens do tokenizado pelo MCL sem CLS e SEP
            tokens_texto_mcl = texto_embeddings['tokens_texto_mcl'][i][1:-1]
                        
            # Recupera as sentenças do texto
            lista_sentencas_texto = self.getPln().getListaSentencasTexto(texto_embeddings['texto_original'][i])

            # Lista de embeddings das sentenças do texto    
            lista_embeddings_tokens_sentencas_texto_media = []
            lista_embeddings_tokens_sentencas_texto_maximo = []
            
            # Percorre as sentenças do texto
            for j, sentenca in enumerate(lista_sentencas_texto):

                # Tokeniza a sentença
                sentenca_tokenizada =  self.transformer.getTextoTokenizado(sentenca)
                
                # Remove os tokens de início e fim da sentença
                sentenca_tokenizada.remove('[CLS]')
                sentenca_tokenizada.remove('[SEP]')    
                #print(len(sentenca_tokenizada))

                # Localiza os índices dos tokens da sentença no texto
                inicio, fim = encontrarIndiceSubLista(tokens_texto_mcl, sentenca_tokenizada)

                # Recupera os embeddings dos tokens da sentença a partir dos embeddings do texto
                embedding_sentenca = embeddings_texto[inicio:fim + 1]

                # Calcula a média dos embeddings dos tokens das sentenças do texto
                embedding_sentenca_media = torch.mean(embedding_sentenca, dim=0)

                # Calcula a média dos embeddings dos tokens das sentenças do texto
                embedding_sentenca_maximo, linha = torch.max(embedding_sentenca, dim=0)

                # Guarda os tokens e os embeddings das sentenças do texto da média e do máximo
                lista_embeddings_tokens_sentencas_texto_media.append(embedding_sentenca_media)
                lista_embeddings_tokens_sentencas_texto_maximo.append(embedding_sentenca_maximo)
            
            # Acumula a saída do método             
            saida['texto_original'].append(texto_embeddings['texto_original'][i])
            saida['tokens_texto_mcl'].append(tokens_texto_mcl)
            saida['sentencas_texto'].append(lista_sentencas_texto)
            saida['sentenca_embeddings_MEAN'].append(lista_embeddings_tokens_sentencas_texto_media)
            saida['sentenca_embeddings_MAX'].append(lista_embeddings_tokens_sentencas_texto_maximo)

        # Se é uma string uma lista com comprimento 1
        if entrada_eh_string:
          saida['texto_original'] = saida['texto_original'][0]
          saida['tokens_texto_mcl'] =  saida['tokens_texto_mcl'][0]
          saida['sentencas_texto'] = saida['sentencas_texto'][0]
          saida['sentenca_embeddings_MEAN'] = saida['sentenca_embeddings_MEAN'][0]
          saida['sentenca_embeddings_MAX'] = saida['sentenca_embeddings_MAX'][0]

        return saida
   
    # ============================
    def getEmbeddingPalavra(self, texto: Union[str, List[str]], 
                             estrategia_pooling: Union[int, EstrategiasPooling] = EstrategiasPooling.MEAN):
        '''
        Recebe um texto (string ou uma lista de strings) e retorna os embeddings consolidados dos tokens das palavras do texto utilizando estratégia pooling especificada(MEAN, MAX).          
        O texto ou a lista de textos será tokenizado utilizando a ferramenta de PLN. 
        Os embeddings dos tokens gerados pelo modelo de linguagem serão consolidados em palavras geradas pela tokenização da ferramenta de pln.
            
        Parâmetros:
        `texto` - Um texto é uma string ou uma lista de strings.
        `estrategia_pooling` - Valor default MEAN. Uma estratégia de pooling,(EstrategiasPooling.MEAN, EstrategiasPooling.MAX). Pode ser utilizado os valores inteiros 0 para MEAN e 1 para MAX.
    
        Retorno: Uma lista com os embeddings consolidados das palavras se o parâmetro texto é uma string, caso contrário uma lista com a lista dos embeddings consolidados das palavras se o parâmetro é lista de string.
        '''

        # Verifica o tipo da estratégia de pooling
        if isinstance(estrategia_pooling, int):
          if estrategia_pooling == 0:
            estrategia_pooling = EstrategiasPooling.MEAN
          else:
            if estrategia_pooling == 1:
              estrategia_pooling = EstrategiasPooling.MAX
            else:
              estrategia_pooling = None
              logger.info("Não foi especificado um valor inteiro para a estratégia de pooling.") 

        # Retorna os embeddings de acordo com a estratégia
        if estrategia_pooling == EstrategiasPooling.MEAN:
            return self.getCodificacaoPalavra(texto)['embeddings_MEAN']
        else:
            if estrategia_pooling == EstrategiasPooling.MAX:
                return self.getCodificacaoPalavra(texto)['embeddings_MAX']
            else:              
                logger.info("Não foi especificado uma estratégia de pooling válida.") 
                return None
            
    # ============================
    def getCodificacaoPalavra(self, texto: Union[str, List[str]],
                              tamanho_lote: int = 32, 
                              mostra_barra_progresso: bool = False,
                              convert_to_numpy: bool = False,
                              device: str = None):      
        
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
            embeddings_MEAN uma lista com os embeddings com a estratégia MEAN.
            embeddings_MAX uma lista com os embeddings com a estratégia MAX.
        '''
        
        # Se o texto é uma string, coloca em uma lista de comprimento 1
        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'):             
            texto = [texto]
            entrada_eh_string = True

        # Recupera os embeddings do texto
        texto_embeddings = self.getCodificacaoCompleta(texto,
                                                       tamanho_lote,
                                                       mostra_barra_progresso,
                                                       convert_to_numpy,
                                                       device)
        
        # Acumula a saída do método
        saida = {}
        saida.update({'texto_original' : [],
                      'tokens_texto': [], 
                      'tokens_texto_mcl' : [],
                      'tokens_oov_texto_mcl': [],                      
                      'tokens_texto_pln' : [],
                      'pos_texto_pln': [],
                      'embeddings_MEAN': [],        
                      'embeddings_MAX': []
                     }
        )

        # Percorre os textos da lista.
        for i, texto in enumerate(texto_embeddings['texto_original']):
            
            # Recupera o texto tokenizado pela ferramenta de pln do texto original
            lista_tokens_texto_pln = self.getPln().getTokensTexto(texto_embeddings['texto_original'][i])
            
            # Recupera a lista de embeddings gerados pelo MCL sem CLS e SEP 
            embeddings_texto = texto_embeddings['token_embeddings'][i][1:-1]
            
            # Recupera a lista de tokens do tokenizado pelo MCL sem CLS e SEP
            tokens_texto_mcl = texto_embeddings['tokens_texto_mcl'][i][1:-1]
            
            # Concatena os tokens gerandos pela ferramenta de pln
            tokens_texto_concatenado = " ".join(lista_tokens_texto_pln)

            # Recupera os embeddings e tokens de palavra            
            saidaEmbeddingPalavra = self.getTransformer().getTokensEmbeddingsPOSTexto(embeddings_texto,
                                                                                      tokens_texto_mcl,
                                                                                      tokens_texto_concatenado,
                                                                                      self.getPln())

            # Acumula a saída do método 
            saida['texto_original'].append(texto_embeddings['texto_original'][i])
            saida['tokens_texto'].append(saidaEmbeddingPalavra['tokens_texto'])
            saida['tokens_texto_mcl'].append(tokens_texto_mcl)
            saida['tokens_oov_texto_mcl'].append(saidaEmbeddingPalavra['tokens_oov_texto_mcl'])            
            saida['tokens_texto_pln'].append(lista_tokens_texto_pln)
            saida['pos_texto_pln'].append(saidaEmbeddingPalavra['pos_texto_pln'])            
            saida['embeddings_MEAN'].append(saidaEmbeddingPalavra['embeddings_MEAN'])
            saida['embeddings_MAX'].append(saidaEmbeddingPalavra['embeddings_MAX'])
        
         # Se é uma string uma lista com comprimento 1
        if entrada_eh_string:
            saida['texto_original'] = saida['texto_original'][0]
            saida['tokens_texto'] = saida['tokens_texto'][0]
            saida['tokens_texto_mcl'] = saida['tokens_texto_mcl'][0]
            saida['tokens_oov_texto_mcl'] = saida['tokens_oov_texto_mcl'][0]
            saida['tokens_texto_pln'] = saida['tokens_texto_pln'][0]
            saida['pos_texto_pln'] = saida['pos_texto_pln'][0]
            saida['embeddings_MEAN'] = saida['embeddings_MEAN'][0]
            saida['embeddings_MAX'] = saida['embeddings_MAX'][0]

        return saida
    
    # ============================
    def getEmbeddingToken(self, texto: Union[str, List[str]]):
        '''
        Recebe um texto (string ou uma lista de strings) e retorna os embeddings dos tokens gerados pelo tokenizador modelo de linguagem.                  
                    
        Parâmetros:
        `texto` - Um texto é uma string ou uma lista de strings.
    
        Retorno: Uma lista com os embeddings dos tokens se o parâmetro texto é uma string, caso contrário uma lista com a lista dos embeddings dos tokens se o parâmetro é lista de string.
        '''

        return self.getCodificacaoToken(texto)['token_embeddings']

    # ============================    
    def getCodificacaoToken(self, texto: Union[str, List[str]],
                            tamanho_lote: int = 32, 
                            mostra_barra_progresso: bool = False,                     
                            convert_to_numpy: bool = False,         
                            device: str = None):
        '''        
        De um texto preparado(tokenizado) ou não, retorna os embeddings dos tokens do texto utilizando estratégia pooling MEAN.
    
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings do modelo de linguagem
    
        Retorna uma lista com os embeddings da última camada.
        '''

        # Se o texto é uma string, coloca em uma lista de comprimento 1
        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'):             
            texto = [texto]
            entrada_eh_string = True

        # Recupera os embeddings do texto
        texto_embeddings = self.getCodificacaoCompleta(texto,
                                                       tamanho_lote,
                                                       mostra_barra_progresso,
                                                       convert_to_numpy,
                                                       device)
        
        # Acumula a saída do método
        saida = {}
        saida.update({'texto_original' : [],
                      'tokens_texto_mcl' : [],                      
                      'token_embeddings': [],        
                      #'all_layer_embeddings': []
                     }
        )

        # Percorre os textos da lista.
        for i, texto in enumerate(texto_embeddings['texto_original']):    

            # Recupera a lista de embeddings gerados pelo MCL sem CLS e SEP 
            lista_token_embeddings = texto_embeddings['token_embeddings'][i][1:-1]

            # Recupera os embeddings do texto  
            #lista_all_layer_embeddings = texto_embeddings['all_layer_embeddings'][i][1:-1]
            
            # Recupera a lista de tokens do tokenizado pelo MCL sem CLS e SEP
            tokens_texto_mcl = texto_embeddings['tokens_texto_mcl'][i][1:-1]

            # Acumula a saída do método 
            #Se é uma string uma lista com comprimento 1
            if entrada_eh_string:
                saida['texto_original'] = texto_embeddings['texto_original'][i]
                saida['tokens_texto_mcl'] = tokens_texto_mcl
                saida['token_embeddings'] = lista_token_embeddings
            else:
                saida['texto_original'].append(texto_embeddings['texto_original'][i])
                saida['tokens_texto_mcl'].append(tokens_texto_mcl)
                saida['token_embeddings'].append(lista_token_embeddings)

        return saida

    # ============================    
    def _tamanho_texto(self, texto: Union[List[int], List[List[int]]]):
        '''                
        Função de ajuda para obter o comprimento do texto de entrada. O texto pode ser 
        uma lista de ints (o que significa um único texto como entrada) ou uma tupla de 
        lista de ints (representando várias entradas de texto para o modelo).

        Parâmetros:
        `texto` - Um texto a ser recuperado o tamanho.

        Retorno:
          Um inteiro com tamanho do texto.
        '''

        if isinstance(texto, dict):              # caso seja um dic de listas
            return len(next(iter(texto.values())))
        else:
            if not hasattr(texto, '__len__'):      # Objeto não tem o len()
                return 1
            else:
                if len(texto) == 0 or isinstance(texto[0], int):  #String vazia ou lista de ints
                    return len(texto)
                else:
                    return sum([len(t) for t in texto])      ##Soma do comprimento de strings individuais
    
    # ============================
    def getModel(self):
        return self.model

    # ============================
    def getTokenizer(self):
        return self.tokenizer

    # ============================
    def getTransformer(self):
        return self.transformer

    # ============================    
    def getMensurador(self):
        return self.mensurador        
        
    # ============================        
    def getPln(self):
        return self.pln                  