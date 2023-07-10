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
# Biblioteca para o sorteio
from random import randint 

# Biblioteca próprias
from textotransformer.modelo.transformerfactory import TransformerFactory
from textotransformer.pln.pln import PLN
from textotransformer.mensurador.mensurador import Mensurador
from textotransformer.modelo.transformer import Transformer
from textotransformer.modelo.modeloargumentos import ModeloArgumentos
from textotransformer.modelo.modeloenum import EstrategiasPooling, GranularidadeTexto
from textotransformer.modelo.modeloenum import AbordagemExtracaoEmbeddingsCamadas
from textotransformer.mensurador.mensuradorenum import PalavraRelevante
from textotransformer.util.utiltexto import contaElemento, encontrarIndiceSubLista, tamanhoTexto

# Objeto de logger
logger = logging.getLogger(__name__)

# Definição dos parâmetros dos modelos do Texto-Transformer.
modelo_args = ModeloArgumentos(
    max_seq_len = 512,
    pretrained_model_name_or_path = "neuralmind/bert-base-portuguese-cased", # Nome do modelo de linguagem pré-treinado Transformer
    modelo_spacy = "pt_core_news_lg",             # Nome do modelo de linguagem da ferramenta de PLN
    do_lower_case = False,                        # default True
    output_attentions = False,                    # default False
    output_hidden_states = True,                  # default False  /Retornar os embeddings das camadas ocultas  
    abordagem_extracao_embeddings_camadas = 2,    # 0-Primeira/1-Penúltima/2-Ùltima/3-Soma 4 últimas/4-Concat 4 últimas/5-Soma todas
    estrategia_pooling = 0,                       # 0 - MEAN estratégia média / 1 - MAX  estratégia maior
    palavra_relevante = 0                         # 0 - Considera todas as palavras das sentenças / 1 - Desconsidera as stopwords / 2 - Considera somente as palavras substantivas
)

class TextoTransformer:
    
    ''' 
    A classe carrega e cria um objeto para manipular um modelo de linguagem baseado e transformer. 
    Permite recuperar e manipular embeddings recuperados de tokens, palavras, sentenças e textos.
     
    Parâmetros:
       `pretrained_model_name_or_path` - Se for um caminho de arquivo no disco, carrega o modelo a partir desse caminho. Se não for um caminho, ele primeiro faz o download do repositório de modelos do Huggingface com esse nome. Valor default: 'neuralmind/bert-base-portuguese-cased'.                  
       `modelo_spacy` - Nome do modelo spaCy a ser instalado e carregado pela ferramenta de pln spaCy. Valor default 'pt_core_news_lg'.
       `abordagem_extracao_embeddings_camadas` - Especifica a abordagem padrão para a extração dos embeddings das camadas do transformer. Valor default '2'. Valores possíveis: 0-Primeira/1-Penúltima/2-Ùltima/3-Soma 4 últimas/4-Concat 4 últimas/5-Todas.
       `do_lower_case` - Se True, converte todas as letras para minúsculas antes da tokenização. Valor default 'False'.
       `device` - Dispositivo (como 'cuda' / 'cpu') que deve ser usado para o processamento. Se `None`, verifica se uma GPU pode ser usada. Se a GPU estiver disponível será usada no processamento. Valor default 'None'.
       `tipo_modelo_pretreinado` - Tipo de modelo pré-treinado. Pode ser "simples" para criar AutoModel (default) ou "mascara" para criar AutoModelForMaskedLM.
    ''' 
    
    # Construtor da classe
    def __init__(self, pretrained_model_name_or_path: str = "neuralmind/bert-base-portuguese-cased", 
                       modelo_spacy: str = "pt_core_news_lg",
                       abordagem_extracao_embeddings_camadas: int = 2,
                       do_lower_case: bool = False,
                       device = None,
                       tipo_modelo_pretreinado: str = "simples"):
                       
        # Parâmetro recebido para o modelo de linguagem
        modelo_args.pretrained_model_name_or_path = pretrained_model_name_or_path
        logger.info("Especificado parâmetro \"pretrained_model_name_or_path\": {}.".format(pretrained_model_name_or_path))
               
        # Parâmetro recebido para o modelo da ferramenta de pln
        modelo_args.modelo_spacy = modelo_spacy
        logger.info("Especificado parâmetro \"modelo_spacy\": {}.".format(modelo_spacy))
        
        # Parâmetro recebido para o modelo do_lower_case
        modelo_args.do_lower_case = do_lower_case
        logger.info("Especificado parâmetro \"do_lower_case\": {}.".format(do_lower_case))
                
        # Retorna um objeto Transformer carregado com o modelo de linguagem especificado especificado nos parâmetros.
        self.transformer = TransformerFactory.getTransformer(modelo_args=modelo_args,
                                                             tipo_modelo_pretreinado=tipo_modelo_pretreinado)
        
        # Recupera o modelo de linguagem do objeto transformer.
        self.auto_model = self.transformer.getAutoModel()
    
        # Recupera o tokenizador do objeto transformer.     
        self.auto_tokenizer = self.transformer.getAutoTokenizer()
        
        # Especifica a abordagem para a extração dos embeddings das camadas do transformer.         
        logger.info("Utilizando abordagem para extração dos embeddings das camadas do transfomer \"{}\" camada(s).".format(AbordagemExtracaoEmbeddingsCamadas.converteInt(modelo_args.abordagem_extracao_embeddings_camadas).getStr()))
                    
        # Especifica camadas para recuperar os embeddings
        modelo_args.abordagem_extracao_embeddings_camadas = abordagem_extracao_embeddings_camadas
      
        # Carrega o spaCy
        self.pln = PLN(modelo_args=modelo_args)
        
        # Se não foi especificado um dispositivo
        if device is None:
            # Verifica se é possível usar GPU
            if torch.cuda.is_available():    
                device = "cuda"
                logger.info("Existem \"{}\" GPU(s) disponíveis.".format(torch.cuda.device_count()))
                logger.info("Iremos usar a GPU: \"{}\".".format(torch.cuda.get_device_name(0)))

            else:                
                device = "cpu"
                logger.info("Sem GPU disponível, usando CPU.")
            
            # Diz ao PyTorch para usar o dispositvo (GPU ou CPU)
            self._target_device = torch.device(device)
        else:
            # Usa o dipositivo informado
            logger.info("Usando dispositivo informado: \"{}\".".format(device))
            self._target_device = torch.device(device)
            
        # Instância o mensurador
        self.mensurador = Mensurador(modelo_args=modelo_args, 
                                     transformer=self.transformer, 
                                     pln=self.pln,
                                     device=self._target_device)

        # Mensagem de carregamento da classe
        logger.info("Classe \"{}\" carregada com os parâmetros: \"{}\".".format(self.__class__.__name__, modelo_args))
    
    # ============================   
    def __repr__(self):
        '''
        Retorna uma string com descrição do objeto.
        '''
        return "Classe (\"{}\") com o Transformer \"{}\" carregada com o modelo \"{}\" e NLP \"{}\" carregada com o modelo \"{}\" ".format(self.__class__.__name__,
                                                                                                                                           self.getTransformer().auto_model.__class__.__name__,
                                                                                                                                           modelo_args.pretrained_model_name_or_path,
                                                                                                                                           self.getPln().model_pln.__class__.__name__,
                                                                                                                                           modelo_args.modelo_spacy)
    
    # ============================
    def _defineEstrategiaPooling(self, estrategia_pooling: Union[int, EstrategiasPooling] = EstrategiasPooling.MEAN):
        ''' 
        Define a estratégia de pooling para os parâmetros do modelo.

        Parâmetros:
           `estrategia_pooling` - Um número de 0 a 1 com a estratégia de pooling das camadas do modelo contextualizado. Valor defaul '0'. Valores possíveis: 0 - MEAN estratégia média / 1 - MAX  estratégia maior.
        ''' 

        # Verifica o tipo de dado do parâmetro 'estrategia_pooling'
        if isinstance(estrategia_pooling, int):
            
            # Converte o parâmetro estrategia_pooling para um objeto da classe EstrategiasPooling
            estrategia_pooling = EstrategiasPooling.converteInt(estrategia_pooling)
        
        # Atribui para os parâmetros do modelo
        if estrategia_pooling == EstrategiasPooling.MEAN:
            modelo_args.estrategia_pooling = EstrategiasPooling.MEAN.value
        else:
            if estrategia_pooling == EstrategiasPooling.MAX:
                modelo_args.estrategia_pooling = EstrategiasPooling.MAX.value            
            else:
                logger.error("Não foi especificado uma estratégia de pooling válida.") 
    
    # ============================
    def _definePalavraRelevante(self, palavra_relevante: Union[int, PalavraRelevante] = PalavraRelevante.ALL):
        ''' 
        Define a estratégia de palavra relavante para os parâmetros do modelo.
        
        Parâmetros:        
           `palavra_relevante` - Um número de 0 a 2 que indica a estratégia de relevância das palavras do texto. Valor defaul '0'. Valores possíveis: 0 - Considera todas as palavras das sentenças / 1 - Desconsidera as stopwords / 2 - Considera somente as palavras substantivas.
        ''' 
        
        # Verifica o tipo de dado do parâmetro 'palavra_relevante'
        if isinstance(palavra_relevante, int):
            
            # Converte o parâmetro estrategia_pooling para um objeto da classe PalavraRelevante
            palavra_relevante = PalavraRelevante.converteInt(palavra_relevante)

        # Atribui para os parâmetros do modelo
        if palavra_relevante == PalavraRelevante.ALL:
            modelo_args.palavra_relevante = PalavraRelevante.ALL.value            
        elif palavra_relevante == PalavraRelevante.CLEAN:
            modelo_args.palavra_relevante = PalavraRelevante.CLEAN.value                
        elif palavra_relevante == PalavraRelevante.NOUN:
            modelo_args.palavra_relevante = PalavraRelevante.NOUN.value                    
        else:
            logger.error("Não foi especificado uma estratégia de relevância de palavras do texto válida.") 

    # ============================
    def getMedidasTexto(self, texto: str, 
                        estrategia_pooling: EstrategiasPooling = EstrategiasPooling.MEAN, 
                        palavra_relevante: Union[int, PalavraRelevante] = PalavraRelevante.ALL,
                        converte_para_numpy: bool = True) -> dict:
        ''' 
        Retorna as medidas de (in)coerência Ccos, Ceuc, Cman do texto.
        
        Parâmetros:
           `texto` - Um texto a ser medido.           
           `estrategia_pooling` - Estratégia de pooling das camadas do transformer.
           `palavra_relevante` - Estratégia de relevância das palavras do texto.
           `converte_para_numpy` - Se verdadeiro, a saída em vetores numpy. Caso contrário, é uma lista de tensores pytorch.
        
        Retorno um dicionário com:
           `cos` - Medida de cos do do texto.
           `euc` - Medida de euc do do texto.
           `man` - Medida de man do do texto.
        ''' 

        self._defineEstrategiaPooling(estrategia_pooling)
        self._definePalavraRelevante(palavra_relevante)
        
        saida = self.mensurador.getMedidasComparacaoTexto(texto=texto, 
                                                          abordagem_extracao_embeddings_camadas=modelo_args.abordagem_extracao_embeddings_camadas,
                                                          converte_para_numpy=converte_para_numpy)
          
        return saida
    
    # ============================
    def getMedidasTextoCosseno(self, texto: str, 
                               estrategia_pooling: Union[int, EstrategiasPooling] = EstrategiasPooling.MEAN, 
                               palavra_relevante: Union[int, PalavraRelevante] = PalavraRelevante.ALL,
                               converte_para_numpy: bool = True):
        ''' 
        Retorna a medida do texto utilizando a medida de similaridade de cosseno.
        
        Parâmetros:
           `texto` - Um texto a ser medido a coerência.           
           `estrategia_pooling` - Estratégia de pooling das camadas do transformer. 
           `palavra_relevante` - Estratégia de relevância das palavras do texto.            
           `converte_para_numpy` - Se verdadeiro, a saída em vetores numpy. Caso contrário, é uma lista de tensores pytorch.
        
        Retorno:
           `cos` - Medida de cos do do texto.            
        '''         

        self._defineEstrategiaPooling(estrategia_pooling)
        self._definePalavraRelevante(palavra_relevante)
        
        saida = self.mensurador.getMedidasComparacaoTexto(texto=texto, 
                                                          abordagem_extracao_embeddings_camadas=modelo_args.abordagem_extracao_embeddings_camadas,
                                                          converte_para_numpy=converte_para_numpy)
          
        return saida['cos']
    
    # ============================
    def getMedidasTextoEuclediana(self, texto: str, 
                                  estrategia_pooling: Union[int, EstrategiasPooling] = EstrategiasPooling.MEAN, 
                                  palavra_relevante: Union[int, PalavraRelevante] = PalavraRelevante.ALL,
                                  converte_para_numpy: bool = True):
        ''' 
        Retorna a medida do texto utilizando a medida de distância de Euclidiana.
                 
        Parâmetros:
           `texto` - Um texto a ser mensurado.           
           `estrategia_pooling` - Estratégia de pooling das camadas do transformer.
           `palavra_relevante` - Estratégia de relevância das palavras do texto.            
           `converte_para_numpy` - Se verdadeiro, a saída em vetores numpy. Caso contrário, é uma lista de tensores pytorch.
        
        Retorno:
           `ceu` - Medida euc do texto.            
        ''' 
        
        self._defineEstrategiaPooling(estrategia_pooling)
        self._definePalavraRelevante(palavra_relevante)

        saida = self.mensurador.getMedidasComparacaoTexto(texto=texto,
                                                          abordagem_extracao_embeddings_camadas=modelo_args.abordagem_extracao_embeddings_camadas,
                                                          converte_para_numpy=converte_para_numpy)
          
        return saida['euc']      
       
    # ============================
    def getMedidasTextoManhattan(self, texto: str, 
                                 estrategia_pooling: Union[int, EstrategiasPooling] = EstrategiasPooling.MEAN, 
                                 palavra_relevante: Union[int, PalavraRelevante] = PalavraRelevante.ALL,
                                 converte_para_numpy: bool = True):
        ''' 
        Retorna a medida do texto utilizando a medida de distância de Manhattan.
                 
        Parâmetros:
           `texto` - Um texto a ser mensurado.           
           `estrategia_pooling` - Estratégia de pooling das camadas do BERT.
           `palavra_relevante` - Estratégia de relevância das palavras do texto.            
           `converte_para_numpy` - Se verdadeiro, a saída em vetores numpy. Caso contrário, é uma lista de tensores pytorch.
        
        Retorno:
           `man` - Medida  Cman do do texto.            
        ''' 
        
        self._defineEstrategiaPooling(estrategia_pooling)
        self._definePalavraRelevante(palavra_relevante)
        
        saida = self.mensurador.getMedidasComparacaoTexto(texto = texto, 
                                                          abordagem_extracao_embeddings_camadas=modelo_args.abordagem_extracao_embeddings_camadas,
                                                          converte_para_numpy=converte_para_numpy)
          
        return saida['man']
    
    # ============================
    def tokenize(self, texto: Union[str, List[str]])-> dict:
        '''
        Tokeniza um texto para submeter ao modelo de linguagem. 
        Retorna um dicionário listas de mesmo tamanho para garantir o processamento em lote.
        Use a quantidade de tokens para saber até onde deve ser recuperado em uma lista de saída.
        Ou use attention_mask diferente de 1 para saber que posições devem ser utilizadas na lista.

        Facilita acesso a classe Transformer.    

        Parâmetros:
           `texto` - Texto a ser tokenizado para o modelo de linguagem.
         
        Retorna um dicionário com as seguintes chaves:
           `tokens_texto_mcl` - Uma lista com os textos tokenizados com os tokens especiais.
           `input_ids` - Uma lista com os ids dos tokens de entrada mapeados em seus índices do vocabuário.
           `token_type_ids` - Uma lista com os tipos dos tokens.
           `attention_mask` - Uma lista com os as máscaras de atenção indicando com '1' os tokens  pertencentes à sentença.
        '''
        return self.getTransformer().tokenize(texto)
        
    # ============================
    def getSaidaRede(self, texto: Union[str, dict]) :
        '''
        De um texto preparado(tokenizado) ou não, retorna token_embeddings, input_ids, attention_mask, token_type_ids, 
        tokens_texto, texto_original  e all_layer_embeddings em um dicionário.
        
        Facilita acesso ao método "getSaidaRede" da classe Transformer.
    
        Parâmetros:
           `texto` - Um texto a ser recuperado os embeddings do modelo de linguagem
    
        Retorna um dicionário com as seguintes chaves:
           `token_embeddings` - Uma lista com os embeddings da última camada.
           `input_ids` - Uma lista com os textos indexados.            
           `attention_mask` - Uma lista com os as máscaras de atenção
           `token_type_ids` - Uma lista com os tipos dos tokens.            
           `tokens_texto` - Uma lista com os textos tokenizados com os tokens especiais.
           `texto_original` - Uma lista com os textos originais.
           `all_layer_embeddings` - Uma lista com os embeddings de todas as camadas.
        '''
        
        return self.getTransformer().getSaidaRede(texto)

    # ============================
    def getSaidaRedeCamada(self, texto: Union[str, dict], 
                           abordagem_extracao_embeddings_camadas: Union[int, AbordagemExtracaoEmbeddingsCamadas] = AbordagemExtracaoEmbeddingsCamadas.ULTIMA_CAMADA) -> dict:
        '''
        Retorna os embeddings do texto de acordo com a abordagem de extração especificada.
        
        Parâmetros:
           `texto` - Texto a ser recuperado os embeddings.
           `abordagem_extracao_embeddings_camadas` - Camada de onde deve ser recupera os embeddings.

        Retorno:
           Os embeddings da camada para o texto.
        '''    

        return self.getTransformer().getSaidaRedeCamada(texto, 
                                                        abordagem_extracao_embeddings_camadas=abordagem_extracao_embeddings_camadas)

    # ============================
    def getCodificacaoCompleta(self, texto: Union[str, List[str]],
                               tamanho_lote: int = 32, 
                               mostra_barra_progresso: bool = False,
                               converte_para_numpy: bool = False,
                               device: str = None) -> dict:

        '''
        Retorna a codificação completa do texto utilizando o modelo de linguagem.
    
        Parâmetros:
           `texto` - Um texto a ser recuperado a codificação em embeddings do modelo de linguagem
           `tamanho_lote` - o tamanho do lote usado para o computação
           `mostra_barra_progresso` - Mostra uma barra de progresso ao codificar o texto.
           `converte_para_numpy` - Se verdadeiro, a saída é uma lista de vetores numpy. Caso contrário, é uma lista de tensores pytorch.
           `device` - Qual torch.device usar para a computação.
    
        Retorna um dicionário com as seguintes chaves:
           `token_embeddings` - Uma lista com os embeddings da última camada.
           `input_ids` - Uma lista com os textos indexados.
           `attention_mask` - Uma lista com os as máscaras de atenção
           `token_type_ids` - Uma lista com os tipos dos tokens.
           `tokens_texto_mcl` - Uma lista com os textos tokenizados com os tokens especiais.
           `texto_original` - Uma lista com os textos originais.
           `all_layer_embeddings` - Uma lista com os embeddings de todas as camadas.
        '''
        
        # Coloca o modelo em modo avaliação
        self.auto_model.eval()

        # Verifica se a entrada é uma string ou uma lista de strings
        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'): 
            #Colocar uma texto individual em uma lista com comprimento 1
            texto = [texto]
            entrada_eh_string = True

        # Padding é o preenchimento do texto para que fiquem do mesmo tamanho nos lotes.
        # A maioria do modelo preenche a direita(0), mas alguns preenchem a esquerda(1)
        #padding_side = 1
        #if isinstance(self.auto_model, XLNetModel):
        #     padding_side = 0
        padding_side = self.getTransformer().getLadoPreenchimento()

        # Se não foi especificado um dispositivo, use-o defaul
        if device is None:
            device = self._target_device

        # Adiciona um dispositivo ao modelo
        self.auto_model.to(device)

        # Dicionário com a saída
        saida = {}
        saida.update({'token_embeddings': [],                        
                      'input_ids': [],
                      'attention_mask': [],
                      'tokens_texto_mcl': [],
                      'texto_original': [],
                      'all_layer_embeddings': []
                      }
                     )

        # Ordena o texto pelo comprimento decrescente
        indice_tamanho_ordenado = np.argsort([-tamanhoTexto(sen) for sen in texto])        
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

            # Tokeniza o lote usando o modelo
            lote_textos_tokenizados = self.getTransformer().tokenize(lote_textos)

            # Adiciona ao device gpu ou cpu
            lote_textos_tokenizados = self.getTransformer().batchToDevice(lote_textos_tokenizados, device)
            
            # Roda o texto através do modelo de linguagem, e coleta todos os estados ocultos produzidos.
            with torch.no_grad():

                # Recupera a saída da rede
                output_rede = self.getTransformer().getSaidaRede(lote_textos_tokenizados)

                # Percorre todas as saídas(textos) do lote                
                for i, texto in enumerate(output_rede['texto_original']):      
                
                    # Recupera o último token que não foi preenchido
                    ultimo_mask_id = len(output_rede['attention_mask'][i])-1
                    
                    # Padding_side = 1 o preenchimento foi realizado no lado direito
                    if padding_side == 1:
                        # Localiza o último token de "attention_mask" igual a 1                    
                        while ultimo_mask_id > 0 and output_rede['attention_mask'][i][ultimo_mask_id].item() == 0:                        
                            ultimo_mask_id -= 1
                        
                        # Recupera os embeddings do primeiro(0) até o último token que "attention_mask" seja 1                        
                        # Concatena a lista dos embeddings do texto a lista já existente                     
                        saida['token_embeddings'].append(output_rede['token_embeddings'][i][0:ultimo_mask_id+1])
                        saida['input_ids'].append(output_rede['input_ids'][i][0:ultimo_mask_id+1])
                        saida['attention_mask'].append(output_rede['attention_mask'][i][0:ultimo_mask_id+1])                    
                        saida['tokens_texto_mcl'].append(output_rede['tokens_texto_mcl'][i][0:ultimo_mask_id+1])
                        saida['texto_original'].append(output_rede['texto_original'][i])
                        
                        # Percorre as camadas da segunda camada até o fim adicionando o lote especifico e descartando os tokens válidos
                        saida['all_layer_embeddings'].append([camada[i][0:ultimo_mask_id+1] for camada in output_rede['all_layer_embeddings'][1:]])
                        
                    else:   
                        # Preenchimento foi realizado no lado esquerdo
                        # Localiza o último token de "attention_mask" igual a 0
                        ultimo_mask_id = 0
                        quantidade_tokens =  len(output_rede['attention_mask'][i])
                        while ultimo_mask_id < quantidade_tokens and output_rede['attention_mask'][i][ultimo_mask_id].item() == 0:                        
                            ultimo_mask_id += 1

                        # Recupera os embeddings do primeiro(0) até o último token que "attention_mask" seja 1                        
                        # Concatena a lista dos embeddings do texto a lista já existente                     
                        saida['token_embeddings'].append(output_rede['token_embeddings'][i][ultimo_mask_id:])
                        saida['input_ids'].append(output_rede['input_ids'][i][ultimo_mask_id:])
                        saida['attention_mask'].append(output_rede['attention_mask'][i][ultimo_mask_id:])                    
                        saida['tokens_texto_mcl'].append(output_rede['tokens_texto_mcl'][i][ultimo_mask_id:])
                        saida['texto_original'].append(output_rede['texto_original'][i])
                        
                        # Percorre as camadas da segunda camada até o fim adicionando o lote especifico e descartando os tokens válidos
                        saida['all_layer_embeddings'].append([camada[i][ultimo_mask_id:] for camada in output_rede['all_layer_embeddings'][1:]])
                   
        # Reorganiza as listas
        saida['token_embeddings'] = [saida['token_embeddings'][idx] for idx in np.argsort(indice_tamanho_ordenado)]
        saida['input_ids'] = [saida['input_ids'][idx] for idx in np.argsort(indice_tamanho_ordenado)]
        saida['attention_mask'] = [saida['attention_mask'][idx] for idx in np.argsort(indice_tamanho_ordenado)]
        saida['tokens_texto_mcl'] = [saida['tokens_texto_mcl'][idx] for idx in np.argsort(indice_tamanho_ordenado)]
        saida['texto_original'] = [saida['texto_original'][idx] for idx in np.argsort(indice_tamanho_ordenado)]
        saida['all_layer_embeddings'] = [saida['all_layer_embeddings'][idx] for idx in np.argsort(indice_tamanho_ordenado)]
                
        # Se estiver usando GPU, copia os tensores para a memória do host.
        if torch.cuda.is_available():
            
            # Copiando os tensores para a memória do host.
            saida['token_embeddings'] = [emb.cpu() for emb in saida['token_embeddings']]
            
            # Copiando os tensores para a memória do host.
            saida['all_layer_embeddings'] = [[[emb.cpu() for emb in camada] for camada in sentenca] for sentenca in saida['all_layer_embeddings']]
            
        # Converte para numpy
        if converte_para_numpy:
            
            # Convertendo para numpy
            saida['token_embeddings'] = [np.array(emb.numpy(), dtype=object) for emb in saida['token_embeddings']]
                        
            # Convertendo para numpy
            saida['all_layer_embeddings'] = [[np.array([emb.numpy() for emb in camada], dtype=object) for camada in sentenca] for sentenca in saida['all_layer_embeddings']]
            # Caso contrário deixa como lista de tensores.            

        # Se é uma string remove a lista de lista
        if entrada_eh_string:
            saida['token_embeddings'] = saida['token_embeddings'][0]
            saida['input_ids'] = saida['input_ids'][0]
            saida['attention_mask'] = saida['attention_mask'][0]
            saida['tokens_texto_mcl'] = saida['tokens_texto_mcl'][0]
            saida['texto_original'] = saida['texto_original'][0]
            saida['all_layer_embeddings'] = saida['all_layer_embeddings'][0]

        return saida

    # ============================
    def getCodificacao(self, texto: Union[str, List[str]],
                       granularidade_texto: Union[int, GranularidadeTexto] = GranularidadeTexto.TOKEN,
                       tamanho_lote: int = 32,
                       mostra_barra_progresso: bool = False,
                       converte_para_numpy: bool = False,
                       device: str = None,
                       dic_excecao: dict = {"":0,}) -> dict:
       
        '''
        Retorna a codificação do texto utilizando o modelo de linguagem de acordo com o tipo codificação do texto.
            
        Parâmetros:
           `texto` - Um texto a ser recuperado a codificação em embeddings do modelo de linguagem.
           `tipo_codificação_texto` - O tipo de codificação do texto. Pode ser: texto, sentenca, palavra e token.         
           `tamanho_lote` - o tamanho do lote usado para o computação.
           `mostra_barra_progresso` - Mostra uma barra de progresso ao codificar o texto.
           `converte_para_numpy` - Se verdadeiro, a saída é uma lista de vetores numpy. Caso contrário, é uma lista de tensores pytorch.
           `device` - Qual torch.device usar para a computação.
           `dic_excecao` - Um dicionário de tokens de exceções e seus deslocamentos para considerar mais ou menos tokens do modelo de linguagem em relação ao spaCy. Valor positivo para considerar mais tokens e negativo para considerar menos tokens. Exemplo exceção negativo: {"1°": -1}, a ferramenta de PLN separa o token "1°" em "1" e "°", portanto é necessário reduzir 1 token pois o MCL gera somente um. Exemplo exceção positiva: {"1°": 1}, a ferramenta de PLN não separa o token "1°", mas o MCL separa em dois "1" e "°" portanto é necessário agrupar em 1 token.
    
        Retorna um dicionário com as seguintes chaves:
           `token_embeddings` - Uma lista com os embeddings da última camada.
           `input_ids` - Uma lista com os textos indexados.            
           `attention_mask` - Uma lista com os as máscaras de atenção
           `token_type_ids` - Uma lista com os tipos dos tokens.            
           `tokens_texto_mcl` - Uma lista com os textos tokenizados com os tokens especiais.
           `texto_original` - Uma lista com os textos originais.
           `all_layer_embeddings` - Uma lista com os embeddings de todas as camadas.
        '''
        
        # Verifica o tipo de dado do parâmetro 'granularidade_texto'
        if isinstance(granularidade_texto, int):
            
            # Converte o parâmetro estrategia_pooling para um objeto da classe GranularidadeTexto
            granularidade_texto = GranularidadeTexto.converteInt(granularidade_texto)
        
        # Verifica qual granularidade de texto foi passada como parâmetro para a função e chama a função correspondente de codificação.        
        if granularidade_texto == GranularidadeTexto.TOKEN:
            return self.getCodificacaoToken(texto=texto, 
                                            tamanho_lote=tamanho_lote, 
                                            mostra_barra_progresso=mostra_barra_progresso, 
                                            converte_para_numpy=converte_para_numpy, 
                                            device=device)
            
        else:
            if granularidade_texto == GranularidadeTexto.PALAVRA:
                return self.getCodificacaoPalavra(texto=texto,
                                                  tamanho_lote=tamanho_lote,
                                                  mostra_barra_progresso=mostra_barra_progresso, 
                                                  converte_para_numpy=converte_para_numpy,
                                                  device=device,
                                                  dic_excecao=dic_excecao)
            
            else:
                if granularidade_texto == GranularidadeTexto.SENTENCA:
                    return self.getCodificacaoSentenca(texto=texto, 
                                                       tamanho_lote=tamanho_lote, 
                                                       mostra_barra_progresso=mostra_barra_progresso, 
                                                       converte_para_numpy=converte_para_numpy, 
                                                       device=device)                
                else:
                    if granularidade_texto == GranularidadeTexto.TEXTO:
                        return self.getCodificacaoTexto(texto=texto,
                                                        tamanho_lote=tamanho_lote, 
                                                        mostra_barra_progresso=mostra_barra_progresso, 
                                                        converte_para_numpy=converte_para_numpy,
                                                        device=device)
                    
                    else:
                        logger.error("Granularidade de texto inválida.")
                        return None

    # ============================
    def getEmbeddingTexto(self, texto: Union[str, List[str]],
                          tamanho_lote: int = 32, 
                          mostra_barra_progresso: bool = False,
                          converte_para_numpy: bool = False,
                          device: str = None,
                          estrategia_pooling: Union[int, EstrategiasPooling] = EstrategiasPooling.MEAN) -> list:
        '''
        De um texto (string ou uma lista de strings) retorna os embeddings do texto consolidados dos tokens utilizando estratégia pooling MEAN e MAX.         
        Utiliza duas estratégias para realizar o pooling de tokens que forma uma palavra.
            - Estratégia MEAN para calcular a média dos embeddings dos tokens que formam uma palavra.
            - Estratégia MAX para calcular o valor máximo dos embeddings dos tokens que formam uma palavra.
            
        Parâmetros:
           `texto` - Um texto a ser recuperado os embeddings dos textos consolidados dos tokens com a estratégia MEAN e MAX utilizando o modelo de linguagem
           `tamanho_lote` - o tamanho do lote usado para o computação
           `mostra_barra_progresso` - Mostra uma barra de progresso ao codificar o texto.
           `converte_para_numpy` - Se verdadeiro, a saída é uma lista de vetores numpy. Caso contrário, é uma lista de tensores pytorch.        
           `device` - Qual torch.device usar para a computação.
           `estrategia_pooling` - Valor default MEAN. Uma estratégia de pooling,(EstrategiasPooling.MEAN, EstrategiasPooling.MAX). Pode ser utilizado os valores inteiros 0 para MEAN e 1 para MAX.
    
        Retorno: 
           Os embeddings consolidados do texto se o parâmetro texto é uma string, caso contrário uma lista com os embeddings consolidados se o parâmetro é lista de string.
        '''

        # Verifica o tipo de dado do parâmetro 'estrategia_pooling'
        if isinstance(estrategia_pooling, int):
            
            # Converte o parâmetro estrategia_pooling para um objeto da classe EstrategiasPooling
            estrategia_pooling = EstrategiasPooling.converteInt(estrategia_pooling)


        # Retorna os embeddings de acordo com a estratégia
        if estrategia_pooling == EstrategiasPooling.MEAN:
            return self.getCodificacaoTexto(texto=texto,
                                            tamanho_lote=tamanho_lote,
                                            mostra_barra_progresso=mostra_barra_progresso,
                                            converte_para_numpy=converte_para_numpy,
                                            device=device)['texto_embeddings_MEAN']
        else:
            if estrategia_pooling == EstrategiasPooling.MAX:
                return self.getCodificacaoTexto(texto=texto,
                                                tamanho_lote=tamanho_lote,
                                                mostra_barra_progresso=mostra_barra_progresso,
                                                converte_para_numpy=converte_para_numpy,
                                                device=device)['texto_embeddings_MAX']
            else:              
                logger.error("Não foi especificado uma estratégia de pooling válida.") 
                return None  

    # ============================
    def getCodificacaoTexto(self, texto: Union[str, List[str]],
                            tamanho_lote: int = 32, 
                            mostra_barra_progresso: bool = False,                     
                            converte_para_numpy: bool = False,
                            device: str = None) -> dict:
        '''                
        De um texto (string ou uma lista de strings) retorna a codificação do texto consolidados dos tokens utilizando estratégia pooling MEAN e MAX.         
        Utiliza duas estratégias para realizar o pooling de tokens que forma uma palavra.
            - Estratégia MEAN para calcular a média dos embeddings dos tokens que formam uma palavra.
            - Estratégia MAX para calcular o valor máximo dos embeddings dos tokens que formam uma palavra.
    
        Parâmetros:         
           `texto` - Um texto a ser recuperado os embeddings dos textos consolidados dos tokens com a estratégia MEAN e MAX utilizando o modelo de linguagem
           `tamanho_lote` - o tamanho do lote usado para o computação
           `mostra_barra_progresso` - Mostra uma barra de progresso ao codificar o texto.
           `converte_para_numpy` - Se verdadeiro, a saída é uma lista de vetores numpy. Caso contrário, é uma lista de tensores pytorch.        
           `device` - Qual torch.device usar para a computação.
    
        Retorna um dicionário com as seguintes chaves:
           `texto_original` - Uma lista com os textos originais.
           `tokens_texto_mcl` - Uma lista com os textos tokenizados com os tokens especiais.        
           `texto_embeddings_MEAN` - Uma lista da média dos embeddings dos tokens do texto.
           `texto_embeddings_MAX` - Uma lista do máximo dos embeddings dos tokens do texto.
        '''

        # Se o texto é uma string, coloca em uma lista de comprimento 1
        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'):             
            texto = [texto]
            entrada_eh_string = True

        # Recupera os embeddings do texto
        texto_embeddings = self.getCodificacaoCompleta(texto=texto,
                                                       tamanho_lote=tamanho_lote,
                                                       mostra_barra_progresso=mostra_barra_progresso,
                                                       converte_para_numpy=converte_para_numpy,
                                                       device=device)

        # Acumula a saída do método
        saida = {}
        saida.update({'texto_original' : [],            # Lista com os textos originais
                      'tokens_texto_mcl' : [],          # Lista com os tokens dos textos originais
                      'texto_embeddings_MEAN': [],      # Lista de lista da média dos embeddings dos tokens do texto
                      'texto_embeddings_MAX': [],       # Lista de lista do máximo dos embeddings dos tokens do texto.
                     }
                    )

        # Percorre os textos da lista.
        for i, texto in enumerate(texto_embeddings['texto_original']):       

            # Recupera a lista de embeddings gerados pelo MCL sem CLS e SEP 
            embeddings_texto = texto_embeddings['token_embeddings'][i][self.getTransformer().getPosicaoTokenInicio():self.getTransformer().getPosicaoTokenFinal()]
           
            # Recupera a lista de tokens do tokenizado pelo MCL sem CLS e SEP
            tokens_texto_mcl = texto_embeddings['tokens_texto_mcl'][i][self.getTransformer().getPosicaoTokenInicio():self.getTransformer().getPosicaoTokenFinal()]
            
            if isinstance(embeddings_texto, torch.Tensor): 
                # Calcula a média dos embeddings dos tokens das sentenças do texto
                embedding_documento_media = torch.mean(embeddings_texto, dim=0)
                # Calcula o máximo dos embeddings dos tokens das sentenças do texto
                embedding_documento_maximo, linha = torch.max(embeddings_texto, dim=0)
            else:
                # Calcula a média dos embeddings dos tokens das sentenças do texto
                embedding_documento_media = np.mean(embeddings_texto, axis=0)
                # Calcula o máximo dos embeddings dos tokens das sentenças do texto
                embedding_documento_maximo = np.max(embeddings_texto, axis=0)
                                
            # Acumula a saída do método 
            saida['texto_original'].append(texto_embeddings['texto_original'][i])
            saida['tokens_texto_mcl'].append(tokens_texto_mcl)
            saida['texto_embeddings_MEAN'].append(embedding_documento_media)
            saida['texto_embeddings_MAX'].append(embedding_documento_maximo)
            
        #Se é uma string uma lista com comprimento 1
        if entrada_eh_string:
            saida['texto_original'] = saida['texto_original'][0]
            saida['tokens_texto_mcl'] = saida['tokens_texto_mcl'][0]
            saida['texto_embeddings_MEAN'] = saida['texto_embeddings_MEAN'][0]
            saida['texto_embeddings_MAX'] = saida['texto_embeddings_MAX'][0]

        return saida
    
    # ============================
    def getEmbeddingSentenca(self, texto: Union[str, List[str]], 
                             tamanho_lote: int = 32, 
                             mostra_barra_progresso: bool = False,
                             converte_para_numpy: bool = False,
                             device: str = None,
                             estrategia_pooling: Union[int, EstrategiasPooling] = EstrategiasPooling.MEAN) -> list:
        '''        
        De um texto (string ou uma lista de strings) retorna os embeddings das sentenças do texto consolidados dos tokens utilizando estratégia pooling MEAN e MAX. 
        O texto ou a lista de textos é sentenciado utilizando a ferramenta de PLN. 
        Utiliza duas estratégias para realizar o pooling de tokens que forma uma palavra.
            - Estratégia MEAN para calcular a média dos embeddings dos tokens que formam uma palavra.
            - Estratégia MAX para calcular o valor máximo dos embeddings dos tokens que formam uma palavra.
            
        Parâmetros:           
           `texto` - Um texto a ser recuperado os embeddings das sentenças consolidados dos tokens com a estratégia MEAN e MAX utilizando o modelo de linguagem
           `tamanho_lote` - o tamanho do lote usado para o computação
           `mostra_barra_progresso` - Mostra uma barra de progresso ao codificar o texto.
           `converte_para_numpy` - Se verdadeiro, a saída é uma lista de vetores numpy. Caso contrário, é uma lista de tensores pytorch.        
           `device` - Qual torch.device usar para a computação.
           `estrategia_pooling` - Valor default MEAN. Uma estratégia de pooling,(EstrategiasPooling.MEAN, EstrategiasPooling.MAX). Pode ser utilizado os valores inteiros 0 para MEAN e 1 para MAX.
    
        Retorno: 
           Uma lista com os embeddings consolidados das sentenças se o parâmetro texto é uma string, caso contrário uma lista com a lista dos embeddings consolidados das sentenças se o parâmetro é lista de string.        
        '''

        # Verifica o tipo de dado do parâmetro 'estrategia_pooling'
        if isinstance(estrategia_pooling, int):
            
            # Converte o parâmetro estrategia_pooling para um objeto da classe EstrategiasPooling
            estrategia_pooling = EstrategiasPooling.converteInt(estrategia_pooling)


        # Retorna os embeddings de acordo com a estratégia
        if estrategia_pooling == EstrategiasPooling.MEAN:
            return self.getCodificacaoSentenca(texto=texto,
                                               tamanho_lote=tamanho_lote,
                                               mostra_barra_progresso=mostra_barra_progresso,
                                               converte_para_numpy=converte_para_numpy,
                                               device=device)['sentenca_embeddings_MEAN']
        else:
            if estrategia_pooling == EstrategiasPooling.MAX:
                return self.getCodificacaoSentenca(texto=texto,
                                                   tamanho_lote=tamanho_lote,
                                                   mostra_barra_progresso=mostra_barra_progresso,
                                                   converte_para_numpy=converte_para_numpy,
                                                   device=device)['sentenca_embeddings_MAX']
            else:              
                logger.error("Não foi especificado uma estratégia de pooling válida.") 
                return None

    # ============================    
    def getCodificacaoSentenca(self, texto: Union[str, List[str]],
                               tamanho_lote: int = 32, 
                               mostra_barra_progresso: bool = False,
                               converte_para_numpy: bool = False,
                               device: str = None) -> dict:      
        '''        
        De um texto (string ou uma lista de strings) retorna a codificação das sentenças do texto consolidados dos tokens utilizando estratégia pooling MEAN e MAX. 
        O texto ou a lista de textos é sentenciado utilizando a ferramenta de PLN. 
        Utiliza duas estratégias para realizar o pooling de tokens que forma uma palavra.
            - Estratégia MEAN para calcular a média dos embeddings dos tokens que formam uma palavra.
            - Estratégia MAX para calcular o valor máximo dos embeddings dos tokens que formam uma palavra.
    
        Parâmetros:
           `texto` - Um texto a ser recuperado os embeddings das sentenças consolidados dos tokens com a estratégia MEAN e MAX utilizando o modelo de linguagem
           `tamanho_lote` - o tamanho do lote usado para o computação
           `mostra_barra_progresso` - Mostra uma barra de progresso ao codificar o texto.
           `converte_para_numpy` - Se verdadeiro, a saída é uma lista de vetores numpy. Caso contrário, é uma lista de tensores pytorch.        
           `device` - Qual torch.device usar para a computação.
    
        Retorna um dicionário com as seguintes chaves:
           `texto_original` - Uma lista com os textos originais.        
           `tokens_texto_mcl` - Uma lista com os textos tokenizados com os tokens especiais.                
           `sentencas_texto` - Uma lista com as sentenças do texto.
           `sentenca_embeddings_MEAN` - Uma lista da média dos embeddings dos tokens da sentença.
           `sentenca_embeddings_MAX` - Uma lista do máximo dos embeddings dos tokens da sentença.        
        '''

        # Se o texto é uma string, coloca em uma lista de comprimento 1
        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'):             
            texto = [texto]
            entrada_eh_string = True
        
        # Recupera os embeddings do texto
        texto_embeddings = self.getCodificacaoCompleta(texto,
                                                       tamanho_lote=tamanho_lote,
                                                       mostra_barra_progresso=mostra_barra_progresso,
                                                       converte_para_numpy=converte_para_numpy,
                                                       device=device)

        # Acumula a saída do método
        saida = {}
        saida.update({'texto_original' : [],               # Lista com os textos originais
                      'tokens_texto_mcl' : [],             # Lista com os tokens dos textos originais
                      'sentencas_texto' : [],               # Lista com as sentenças do texto
                      'sentenca_embeddings_MEAN': [],      # Lista de lista média dos embeddings dos tokens da sentença.
                      'sentenca_embeddings_MAX': [],       # Lista de lista máximo dos embeddings dos tokens da sentença.
                     }
        )

        # Percorre os textos da lista.
        for i, texto in enumerate(texto_embeddings['texto_original']):       

            # Recupera a lista de embeddings gerados pelo MCL sem os tokens de classificação e separação.
            embeddings_texto = texto_embeddings['token_embeddings'][i][self.getTransformer().getPosicaoTokenInicio():self.getTransformer().getPosicaoTokenFinal()]

            # Recupera a lista de tokens do tokenizado pelo MCL sem os tokens de classificação e separação.
            tokens_texto_mcl = texto_embeddings['tokens_texto_mcl'][i][self.getTransformer().getPosicaoTokenInicio():self.getTransformer().getPosicaoTokenFinal()]
            
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
                sentenca_tokenizada = self.transformer.removeTokensEspeciais(sentenca_tokenizada)
                #print(len(sentenca_tokenizada))
                
                # Se for do tipo Roberta, GTP2 Model, adiciona o token de separação no início da sentença
                if j != 0:
                    if self.getTransformer().getPrimeiroTokenSemSeparador():
                        sentenca_tokenizada = self.getTransformer().trataListaTokensEspeciais(sentenca_tokenizada)
                        
                # Localiza os índices dos tokens da sentença no texto
                inicio, fim = encontrarIndiceSubLista(tokens_texto_mcl, sentenca_tokenizada)
                #print("inicio:", inicio, "  fim:", fim)
                if inicio == -1 or fim == -1:
                    logger.error("Não encontrei a sentença: {} dentro de {}.".format(sentenca_tokenizada, tokens_texto_mcl))

                # Recupera os embeddings dos tokens da sentença a partir dos embeddings do texto
                embedding_sentenca = embeddings_texto[inicio:fim + 1]
                #print("len(embedding_sentenca):", len(embedding_sentenca))

                if isinstance(embedding_sentenca, torch.Tensor): 
                    # Calcula a média dos embeddings dos tokens das sentenças do texto
                    embedding_sentenca_media = torch.mean(embedding_sentenca, dim=0)
                    # Calcula a média dos embeddings dos tokens das sentenças do texto
                    embedding_sentenca_maximo, linha = torch.max(embedding_sentenca, dim=0)
                else:
                    # Calcula a média dos embeddings dos tokens das sentenças do texto
                    embedding_sentenca_media = np.mean(embedding_sentenca, axis=0)
                    # Calcula o máximo dos embeddings dos tokens das sentenças do texto
                    embedding_sentenca_maximo = np.max(embedding_sentenca, axis=0)

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
          saida['tokens_texto_mcl'] = saida['tokens_texto_mcl'][0]
          saida['sentencas_texto'] = saida['sentencas_texto'][0]
          saida['sentenca_embeddings_MEAN'] = saida['sentenca_embeddings_MEAN'][0]
          saida['sentenca_embeddings_MAX'] = saida['sentenca_embeddings_MAX'][0]

        return saida
   
    # ============================
    def getEmbeddingPalavra(self, texto: Union[str, List[str]], 
                            tamanho_lote: int = 32, 
                            mostra_barra_progresso: bool = False,
                            converte_para_numpy: bool = False,
                            device: str = None, 
                            estrategia_pooling: Union[int, EstrategiasPooling] = EstrategiasPooling.MEAN) -> list:
        
        '''
        De um texto (string ou uma lista de strings) retorna os embeddings das palavras do texto, igualando a quantidade de tokens do spaCy com a tokenização do MCL de acordo com a estratégia. 
        As palavras são tokenizadas utilizando a ferramenta de PLN. 
        Utiliza duas estratégias para realizar o pooling de tokens que forma uma palavra.
            - Estratégia MEAN para calcular a média dos embeddings dos tokens que formam uma palavra.
            - Estratégia MAX para calcular o valor máximo dos embeddings dos tokens que formam uma palavra.
            
        Parâmetros:
           `texto` - Um texto a ser recuperado os embeddings das palavras consolidados dos tokens com a estratégia MEAN e MAX utilizando o modelo de linguagem
           `tamanho_lote` - o tamanho do lote usado para o computação
           `mostra_barra_progresso` - Mostra uma barra de progresso ao codificar o texto.
           `converte_para_numpy` - Se verdadeiro, a saída é uma lista de vetores numpy. Caso contrário, é uma lista de tensores pytorch.        
           `device` - Qual torch.device usar para o computação.
           `estrategia_pooling` - Valor default MEAN. Uma estratégia de pooling,(EstrategiasPooling.MEAN, EstrategiasPooling.MAX). Pode ser utilizado os valores inteiros 0 para MEAN e 1 para MAX.
    
        Retorno: 
           Uma lista com os embeddings consolidados das palavras se o parâmetro texto é uma string, caso contrário uma lista com a lista dos embeddings consolidados das palavras se o parâmetro é lista de string.
        '''

        # Verifica o tipo de dado do parâmetro 'estrategia_pooling'
        if isinstance(estrategia_pooling, int):
            
            # Converte o parâmetro estrategia_pooling para um objeto da classe EstrategiasPooling
            estrategia_pooling = EstrategiasPooling.converteInt(estrategia_pooling)

        # Retorna os embeddings de acordo com a estratégia
        if estrategia_pooling == EstrategiasPooling.MEAN:
            return self.getCodificacaoPalavra(texto=texto,
                                              tamanho_lote=tamanho_lote,
                                              mostra_barra_progresso=mostra_barra_progresso,
                                              converte_para_numpy=converte_para_numpy,
                                              device=device)['palavra_embeddings_MEAN']
        else:
            if estrategia_pooling == EstrategiasPooling.MAX:
                return self.getCodificacaoPalavra(texto=texto,
                                                  tamanho_lote=tamanho_lote,
                                                  mostra_barra_progresso=mostra_barra_progresso,
                                                  converte_para_numpy=converte_para_numpy,
                                                  device=device)['palavra_embeddings_MAX']
            else:              
                logger.error("Não foi especificado uma estratégia de pooling válida.") 
                return None
            
    # ============================
    def getCodificacaoPalavra(self, texto: Union[str, List[str]],
                              tamanho_lote: int = 32, 
                              mostra_barra_progresso: bool = False,
                              converte_para_numpy: bool = False,
                              device: str = None,
                              dic_excecao: dict = {"":0,}) -> dict:      
        
        '''                
        De um texto (string ou uma lista de strings) retorna a codificação das palavras do texto, igualando a quantidade de tokens do spaCy com a tokenização do MCL de acordo com a estratégia. 
        As palavras são tokenizadas utilizando a ferramenta de PLN. 
        Utiliza duas estratégias para realizar o pooling de tokens que forma uma palavra.
            - Estratégia MEAN para calcular a média dos embeddings dos tokens que formam uma palavra.
            - Estratégia MAX para calcular o valor máximo dos embeddings dos tokens que formam uma palavra.
            
        Parâmetros:
           `texto` - Um texto a ser recuperado os embeddings das palavras consolidados dos tokens com a estratégia MEAN e MAX utilizando o modelo de linguagem.
           `tamanho_lote` - o tamanho do lote usado para o computação.
           `mostra_barra_progresso` - Mostra uma barra de progresso ao codificar o texto.
           `converte_para_numpy` - Se verdadeiro, a saída é uma lista de vetores numpy. Caso contrário, é uma lista de tensores pytorch.
           `device` - Qual torch.device usar para o computação.
           `dic_excecao` - Um dicionário de tokens de exceções e seus deslocamentos para considerar mais ou menos tokens do modelo de linguagem em relação ao spaCy. Valor positivo para considerar mais tokens e negativo para considerar menos tokens. Exemplo exceção negativo: {"1°": -1}, a ferramenta de PLN separa o token "1°" em "1" e "°", portanto é necessário reduzir 1 token pois o MCL gera somente um. Exemplo exceção positiva: {"1°": 1}, a ferramenta de PLN não separa o token "1°", mas o MCL separa em dois "1" e "°" portanto é necessário agrupar em 1 token.
    
        Retorna um dicionário com as seguintes chaves:
           `texto_original` - Uma lista com os textos originais.  
           `tokens_texto` - Uma lista com os tokens(palavras) realizados pelo método.
           `tokens_texto_mcl` - Uma lista com os tokens e tokens especiais realizados pelo MCL.
           `tokens_oov_texto_mcl` - Uma lista com os tokens OOV(com ##) do MCL.
           `tokens_texto_pln` - Uma lista com os tokens realizados pela ferramenta de PLN(spaCy).
           `pos_texto_pln` - Uma lista com as postagging dos tokens realizados pela ferramenta de PLN(spaCy).            
           `palavra_embeddings_MEAN` - Uma lista com os embeddings das palavras com a estratégia MEAN.
           `palavra_embeddings_MAX` - Uma lista com os embeddings das palavras com a estratégia MAX.
        '''
                
        # Se o texto é uma string, coloca em uma lista de comprimento 1
        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'):             
            texto = [texto]
            entrada_eh_string = True

        # Recupera os embeddings do texto
        texto_embeddings = self.getCodificacaoCompleta(texto=texto,
                                                       tamanho_lote=tamanho_lote,
                                                       mostra_barra_progresso=mostra_barra_progresso,
                                                       converte_para_numpy=converte_para_numpy,
                                                       device=device)
        
        # Acumula a saída do método
        saida = {}
        saida.update({'texto_original' : [],
                      'tokens_texto': [], 
                      'tokens_texto_mcl' : [],
                      'tokens_oov_texto_mcl': [],                      
                      'tokens_texto_pln' : [],
                      'pos_texto_pln': [],
                      'palavra_embeddings_MEAN': [],        
                      'palavra_embeddings_MAX': []
                     }
        )

        # Percorre os textos da lista.
        for i, texto in enumerate(texto_embeddings['texto_original']):
            
            # Recupera o texto tokenizado pela ferramenta de pln do texto original
            lista_tokens_texto_pln = self.getPln().getTokensTexto(texto_embeddings['texto_original'][i])
            
            # Recupera a lista de embeddings gerados pelo MCL sem os tokens de classificação e separação.
            embeddings_texto = texto_embeddings['token_embeddings'][i][self.getTransformer().getPosicaoTokenInicio():self.getTransformer().getPosicaoTokenFinal()]
            
            # Recupera a lista de tokens do tokenizado pelo MCL sem os tokens de classificação e separação.
            tokens_texto_mcl = texto_embeddings['tokens_texto_mcl'][i][self.getTransformer().getPosicaoTokenInicio():self.getTransformer().getPosicaoTokenFinal()]
            
            # Concatena os tokens gerandos pela ferramenta de pln
            tokens_texto_concatenado_pln = " ".join(lista_tokens_texto_pln)
            # print("tokens_texto_concatenado:", tokens_texto_concatenado)

            # Recupera os embeddings dos tokens das palavras
            saida_embedding_palavra = self.getTransformer().getTokensPalavrasEmbeddingsTexto(embeddings_texto=embeddings_texto,
                                                                                             tokens_texto_mcl=tokens_texto_mcl,
                                                                                             tokens_texto_concatenado_pln=tokens_texto_concatenado_pln,
                                                                                             pln=self.getPln(),
                                                                                             dic_excecao=dic_excecao)

            # Acumula a saída do método 
            saida['texto_original'].append(texto_embeddings['texto_original'][i])
            saida['tokens_texto'].append(saida_embedding_palavra['tokens_texto'])
            saida['tokens_texto_mcl'].append(tokens_texto_mcl)
            saida['tokens_oov_texto_mcl'].append(saida_embedding_palavra['tokens_oov_texto_mcl'])            
            saida['tokens_texto_pln'].append(lista_tokens_texto_pln)
            saida['pos_texto_pln'].append(saida_embedding_palavra['pos_texto_pln'])
            # Lista dos embeddings de palavras com a média dos embeddings dos tokens que formam a palavra
            saida['palavra_embeddings_MEAN'].append(saida_embedding_palavra['palavra_embeddings_MEAN'])
            # Lista dos embeddings de palavras com o máximo dos embeddings dos tokens que formam a palavra
            saida['palavra_embeddings_MAX'].append(saida_embedding_palavra['palavra_embeddings_MAX'])
        
        # Se é uma string uma lista com comprimento 1
        if entrada_eh_string:
            saida['texto_original'] = saida['texto_original'][0]
            saida['tokens_texto'] = saida['tokens_texto'][0]
            saida['tokens_texto_mcl'] = saida['tokens_texto_mcl'][0]
            saida['tokens_oov_texto_mcl'] = saida['tokens_oov_texto_mcl'][0]
            saida['tokens_texto_pln'] = saida['tokens_texto_pln'][0]
            saida['pos_texto_pln'] = saida['pos_texto_pln'][0]
            saida['palavra_embeddings_MEAN'] = saida['palavra_embeddings_MEAN'][0]
            saida['palavra_embeddings_MAX'] = saida['palavra_embeddings_MAX'][0]

        return saida
    
    # ============================
    def getEmbeddingToken(self, texto: Union[str, List[str]],
                          tamanho_lote: int = 32,
                          mostra_barra_progresso: bool = False,
                          converte_para_numpy: bool = False,
                          device: str = None) -> list:
        '''
        Recebe um texto (string ou uma lista de strings) e retorna os embeddings dos tokens gerados pelo tokenizador modelo de linguagem.                  
                    
        Parâmetros:
           `texto` - Um texto a ser recuperado os embeddings dos tokens utilizando o modelo de linguagem
           `tamanho_lote` - o tamanho do lote usado para o computação
           `mostra_barra_progresso` - Mostra uma barra de progresso ao codificar o texto.
           `converte_para_numpy` - Se verdadeiro, a saída é uma lista de vetores numpy. Caso contrário, é uma lista de tensores pytorch.           
           `device` - Qual torch.device usar para o computação.
    
        Retorno: 
           Uma lista com os embeddings dos tokens se o parâmetro texto é uma string, caso contrário uma lista com a lista dos embeddings dos tokens se o parâmetro é lista de string.
        '''

        return self.getCodificacaoToken(texto=texto,
                                        tamanho_lote=tamanho_lote,
                                        mostra_barra_progresso=mostra_barra_progresso,
                                        converte_para_numpy=converte_para_numpy,
                                        device=device)['token_embeddings']

    # ============================    
    def getCodificacaoToken(self, texto: Union[str, List[str]],
                            tamanho_lote: int = 32, 
                            mostra_barra_progresso: bool = False,
                            converte_para_numpy: bool = False,
                            device: str = None) -> dict:
        '''        
        De um texto (string ou uma lista de strings) retorna a codificação dos tokens do texto utilizando o modelo de linguagem.
                
        Parâmetros:
           `texto` - Um texto a ser recuperado os embeddings dos tokens utilizando o modelo de linguagem
           `tamanho_lote` - o tamanho do lote usado para o computação
           `mostra_barra_progresso` - Mostra uma barra de progresso ao codificar o texto.
           `converte_para_numpy` - Se verdadeiro, a saída é uma lista de vetores numpy. Caso contrário, é uma lista de tensores pytorch.        
           `device` - Qual torch.device usar para o computação.
    
        Retorna um dicionário com as seguintes chaves:
           `texto_original` - Uma lista com os textos originais.  
           `tokens_texto_mcl` - Uma lista com os tokens e tokens especiais realizados pelo mcl.
           `input_ids` - Uma lista com os textos indexados.
           `token_embeddings` - Uma lista com os embeddings dos tokens.
        '''

        # Se o texto é uma string, coloca em uma lista de comprimento 1
        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'):             
            texto = [texto]
            entrada_eh_string = True

        # Recupera os embeddings do texto
        texto_embeddings = self.getCodificacaoCompleta(texto=texto,
                                                       tamanho_lote=tamanho_lote,
                                                       mostra_barra_progresso=mostra_barra_progresso,
                                                       converte_para_numpy=converte_para_numpy,
                                                       device=device)
        
        # Acumula a saída do método
        saida = {}
        saida.update({'texto_original' : [],
                      'tokens_texto_mcl' : [],
                      'input_ids' : [],
                      'token_embeddings': []
                     }
        )

        # Percorre os textos da lista.
        for i, texto in enumerate(texto_embeddings['texto_original']):    

            # Recupera a lista de embeddings gerados pelo MCL sem os tokens de classificação e separação.
            lista_token_embeddings = texto_embeddings['token_embeddings'][i][self.getTransformer().getPosicaoTokenInicio():self.getTransformer().getPosicaoTokenFinal()]

            # Recupera a lista de tokens do tokenizado pelo MCL sem os tokens de classificação e separação.
            lista_input_ids = texto_embeddings['input_ids'][i][self.getTransformer().getPosicaoTokenInicio():self.getTransformer().getPosicaoTokenFinal()]

            # Recupera a lista de tokens do tokenizado pelo MCL sem os tokens de classificação e separação.
            lista_tokens_texto_mcl = texto_embeddings['tokens_texto_mcl'][i][self.getTransformer().getPosicaoTokenInicio():self.getTransformer().getPosicaoTokenFinal()]

            # Acumula a saída do método             
            saida['texto_original'].append(texto_embeddings['texto_original'][i])
            saida['tokens_texto_mcl'].append(lista_tokens_texto_mcl)
            saida['input_ids'].append(lista_input_ids)
            # Converte o tensor de 2 dimensões(token x embeddings) para uma lista de token de embeddings
            saida['token_embeddings'].append([emb for emb in lista_token_embeddings])

        #Se é uma string uma lista com comprimento 1
        if entrada_eh_string:
            saida['texto_original'] = saida['texto_original'][0]
            saida['tokens_texto_mcl'] = saida['tokens_texto_mcl'][0]
            saida['input_ids'] = saida['input_ids'][0]
            saida['token_embeddings'] = saida['token_embeddings'][0]

        return saida
    
    # ============================
    def getTextoMascarado(self, texto: str,
                          texto_token: list[str],
                          texto_pos: list[str],
                          classe: list[str] =["VERB","NOUN","AUX"], 
                          qtde: int = 1):
        ''' 
        Gera o texto mascarado com [MAKS] para usar com MLM do BERT.
        Considera determinadas classes morfossintática das palavras e uma quantidade(qtde) de palavras a serem mascaradas.
            
        Parâmetros:
           `texto` - Texto a ser mascarada.
           `texto_token` - Lista com os tokens do texto.
           `texto_pos` - Lista com as POS dos tokens do texto.
           `classe` - Lista com as classes morfossintática das palavras a serem mascarada com [MASK].
           `qtde` - Quantidade de mascarada a serem realizadas nas palavras do texto.
                    Seleciona aleatoriamente a(s) palavra(s) a ser(em) mascarada(s) se a qtde 
                    for menor que quantidade de palavras das classes no texto.

        Retorno:    
           `texto_mascarado` - Texto mascarado.
           `palavra_mascarada` - Lista com as palavras substituidas pela máscara.
        '''
        
        # Somente modelos que possuem token de mascara.
        if self.getTransformer().getTokenMascara() != None:
        
            texto_mascarado = ""
            palavra_mascarada = ""

            # Verifica a quantidade de trocas a ser realizada
            if qtde != 0:

                # Conta o número de palavras das classes especificadas
                if len(classe) > 1:
                    # Se tem duas classes usa a primeira para contar se existe uma palavra
                    # Pega o primeiro para realizar a conta
                    classe_conta = [classe[0]]
                    conta_mascara = contaElemento(texto_pos, classe_conta)
                    
                    # Senão encontrar pega a segunda classe
                    if conta_mascara == 0:
                        #Pega a segunda classe
                        classe_conta = [classe[1]]
                        conta_mascara = contaElemento(texto_pos, classe_conta)

                        # Senão encontrar pega a terceira classe
                        if conta_mascara == 0:
                            #Pega a terceira classe
                            classe_conta = [classe[2]]
                            conta_mascara = contaElemento(texto_pos, classe_conta) 
                    
                        # Usa a classe para gerar o texto mascarado
                        classe = classe_conta
                    else:
                        conta_mascara = contaElemento(texto_pos, classe)
                
                    # Verifica se existe palavras das classes a serem mascaradas
                    if conta_mascara != 0:    
                        # Verifica a quantidade de trocas é menor que a quantidade palavras a serem trocadas encontradas
                        if qtde < conta_mascara:
                            # A quantidade de trocas é menor que a quantidade de palavras existentes
                            # Precisa sortear as posições que serão trocadas pela máscara dentro da quantidade
                                
                            roleta = []
                            # preenche a roleta com o indice das palavras as serem mascaradas
                            for i in range(conta_mascara):
                                roleta.append(i)

                            # Sorteia as posições das trocas
                            posicao = []
                            for i in range(qtde):
                                posicao_sorteio = randint(0, len(roleta)-1)
                                # Guarda o número sorteado
                                posicao.append(roleta[posicao_sorteio])
                                # Remove o elemento sorteado da roleta
                                del roleta[posicao_sorteio]
                            
                            # Conta o número das trocas realizadas
                            troca = 0

                            # Substitui o elemento pela máscara
                            for i, token in enumerate(texto_token):            
                                # Se a classe da palavra é a desejada
                                if texto_pos[i] in classe:
                                    # Verifica se a troca deve ser realizada para a posição
                                    if troca in posicao:      
                                        # Trocar palavra da classe por [MASK]
                                        texto_mascarado = texto_mascarado + self.getTransformer().getTokenMascara() + " "    
                                        # Guarda a palavra que foi mascarada
                                        palavra_mascarada = token                                  
                                    else:                  
                                        # Adiciona o token
                                        texto_mascarado = texto_mascarado + token + " "
                                        # Avança para a próxima troca
                                        troca = troca + 1
                                else:
                                    # Adiciona o token
                                    texto_mascarado = texto_mascarado + token + " "
                        else:        
                            # Trocar todas as palavras pela mascará, pois a quantidade
                            # de trocas é igual a quantidade de mascarás existentes na sentença

                            # Substitui o elemento da classe pela mascará
                            for i, token in enumerate(texto_token):
                                #print(token, sentenca_pos[i])        
                                # Se a classe da palavra é a desejada
                                if texto_pos[i] in classe:
                                    # Trocar palavra da classe por [MASK]
                                    texto_mascarado = texto_mascarado + self.getTransformer().getTokenMascara() + " "    
                                    # Guarda a palavra que foi mascarada
                                    palavra_mascarada = token 
                                else:
                                    texto_mascarado = texto_mascarado + token + " "
                else:
                    # Não existe palavras da classe especificada      
                    logger.erro("Não existe palavras da classe especificada.")
                    logger.erro("texto:",texto)
                    logger.erro("texto_pos:",texto_pos)
                    texto_mascarado = texto    
            else:
                # Quantidade trocas igual a 0
                logger.erro("Não foi especificado uma quantidade de trocas.")
                texto_mascarado = texto

            # Retira o espaço em branco do início e fim do texto
            texto_mascarado = texto_mascarado.strip(" ")

            return texto_mascarado, palavra_mascarada
        
        else:
            logger.error("O modelo \"{}\" não possui um token de máscara.".format(self.getModel().__class__.__name__))
            
            return None        
    
    # ============================
    def getPrevisaoPalavraTexto(self, texto,
                                top_k_predicao=5):
        ''' 
        Retorna uma lista com as k previsões para a palavra mascarada no texto.
            
        Parâmetros:
            `texto` - Texto mascarado.
            `top_k_predicao` - Quantidade de palavras a serem recuperadas mais próximas da máscara.

        Retorno:
            Lista com as k previsões para a palavra mascarada no texto.

        '''
        
        # Somente modelos que possuem token de mascara.
        if self.getTransformer().getTokenMascara() != None:
        
            # Divide as palavras em tokens        
            texto_tokenizado = self.getTransformer().getTextoTokenizado(texto)
            #print("texto_tokenizado:", texto_tokenizado)

            # Retorna o índice da mascara de atenção
            mascara_atencao_indice = texto_tokenizado.index(self.getTransformer().getTokenMascara())
            #print("mascara_atencao_indice:", mascara_atencao_indice)

            # Mapeia os tokens em seus índices do vocabulário
            tokens_indexados = self.getTokenizer().convert_tokens_to_ids(texto_tokenizado)
            #print("tokens_indexados:", tokens_indexados)

            # Define índices das sentenças A e B associados à 1ª e 2ª sentença 
            segmentos_ids = [0]*len(texto_tokenizado)
            
            # Converte as entradas de lista para tensores do torch
            tokens_tensores = torch.tensor([tokens_indexados])
            segmentos_tensores = torch.tensor([segmentos_ids])
            
            # Se existe GPU disponível.
            if torch.cuda.is_available():  
                # Se você tem uma GPU
                tokens_tensores = tokens_tensores.to('cuda')
                segmentos_tensores = segmentos_tensores.to('cuda')  

            # Realiza a predição dos tokens
            with torch.no_grad():
                ## Retorno de model quando ´output_hidden_states=True´ é setado:  
                ##outputs[0] = last_hidden_state, outputs[1] = pooler_output, outputs[2] = hidden_states
                outputs = self.getModel()(tokens_tensores, token_type_ids=segmentos_tensores)

                ## A predição é recuperada dos embeddings da última camada oculta do modelo        
                predicao = outputs[0]
                
            #print("shape:", predicao.shape)
            #print("mascara_atencao_indice:", mascara_atencao_indice)

            # Normaliza os pesos dos embeddings das predições e calcula sua probabilidade usando softmax.
            probabilidades = torch.nn.functional.softmax(predicao[0, mascara_atencao_indice], dim=-1)    
            # Probabilidade de cada uma das 29.794 palavras do vocabulário do BERT ser a palavra mascarada.
            #print("Tamanho vocabulário:", len(tokenizer.get_vocab())) #29.794
            
            # Retorna os k maiores elementos com as maiores probabilidades e sua posição(ordenada descrescentemente).
            top_k_predicao_pesos, top_k_predicao_indices = torch.topk(probabilidades, top_k_predicao, sorted=True)
            
            # Converte os ids para os tokens do vocabulário
            tokens_predicao = self.getTokenizer().convert_ids_to_tokens([ind.item() for ind in top_k_predicao_indices])

            # Retorna a predição e a probabilidade      
            return list(zip(tokens_predicao, top_k_predicao_pesos))[:top_k_predicao]            
        
        else:
            logger.error("O modelo \"{}\" não possui um token de máscara.".format(self.getModel().__class__.__name__))
            
            return None
    
    # ============================
    def getPerturbacaoPalavraTextoAleatoria(self, 
                                            texto, 
                                            texto_token, 
                                            texto_pos, 
                                            qtde=1, 
                                            top_k_predicao = 100):
        ''' 
        Gera as palavras da perturbação da máscara do texto.
        Considera determinadas classes morfossintática das palavras.
            
        Parâmetros:
           `texto` - Texto a ser mascarada.
           `texto_token` - Lista com os tokens do texto.
           `texto_pos` - Lista com as POS dos tokens do texto.
           `qtde` - Quantidade de mascarada a serem realizadas nas palavras do texto.
                    Seleciona aleatoriamente a(s) palavra(s) a ser(em) mascarada(s) se a qtde 
                    for menor que quantidade de palavras das classes no texto.          
           `top_k_predicao` - Quantidade de palavras a serem recuperadas mais próximas da máscara.

        Retorno:    
           `texto_mascarado` - Texto mascarado.
           `palavra_mascarada` - Palavra substituídas pela máscara.
           `token_predito` - Palavra prevista para a máscara.
           `token_peso` - Peso da palavra prevista.
           `posicao_sorteio` - Posição da palavra prevista na lista de previsões.
           `token_predito_marcado` - Token previsto marcado(##) para a máscara.
           `lista_previsoes` - Lista dos 'top_k_predicao' tokens preditos para a máscara.
        '''
        
        # Somente modelos que possuem token de mascara.
        if self.getTransformer().getTokenMascara() != None:

            #print("texto:", texto)
            texto_mascarado, palavra_mascarada = self.getTextoMascarado(texto, texto_token, texto_pos, classe=["VERB","NOUN","AUX"], qtde=1)
                    
            # Divide as palavras em tokens
            texto_tokenizado = self.getTransformer().getTextoTokenizado(texto)
            #print("texto_tokenizado:", texto_tokenizado)

            # Retorna o índice da mascara de atenção
            mascara_atencao_indice = texto_tokenizado.index(self.getTransformer().getTokenMascara())
            #print("mascara_atencao_indice:", mascara_atencao_indice)

            # Mapeia os tokens em seus índices do vocabulário
            tokens_indexados = self.getTokenizer().convert_tokens_to_ids(texto_tokenizado)
            #print("tokens_indexados:", tokens_indexados)
            
            # Converte as entradas de lista para tensores do torch
            tokens_tensores = torch.tensor([tokens_indexados])
            
            # Realiza a predição dos tokens
            with torch.no_grad():
                # Retorno de model quando ´output_hidden_states=True´ é setado:  
                #outputs[0] = last_hidden_state, outputs[1] = pooler_output, outputs[2] = hidden_states
                outputs = self.getModel()(tokens_tensores)

            # Recupera a predição com os embeddings da última camada oculta    
            predicao = outputs[0]
            
            # Normaliza os pesos das predições nos embeddings e calcula sua probabilidade
            probabilidades = torch.nn.functional.softmax(predicao[0, mascara_atencao_indice], dim=-1)    
            # Retorna os k maiores elementos de determinado tensor de entrada ao longo de uma determinada dimensão de forma ordenada descrescentemente.
            
            # Se existe mais de uma top_k_predição    
            if top_k_predicao != 1:

                # Recupera as top_k_predicao predições em ordem de orobabilidades
                top_k_predicao_pesos, top_k_predicao_indices = torch.topk(probabilidades, top_k_predicao, sorted=True)
                #print("top_k_predicao_pesos:",top_k_predicao_pesos)
                #print("top_k_predicao_indices:",top_k_predicao_indices)
                #print("len(top_k_predicao_indices):",len(top_k_predicao_indices))

                # Sorteia uma predição do intervalo
                posicao_sorteio = randint(0, top_k_predicao-1)    
                #print("posicao_sorteio:",posicao_sorteio)

                # Recupera as predições    
                # Mapeia os índices do vocabulário para os seus tokens
                token_predito = self.getTokenizer().convert_ids_to_tokens([top_k_predicao_indices[posicao_sorteio]])[0]
                # Recupera os pesos da predição
                token_peso = top_k_predicao_pesos[posicao_sorteio]
                #print((posicao_sorteio+1), "[MASK]: ", token_predito, " | peso:", float(token_peso))
                    
                # Se o token predito for igual a palavra que foi substituída pela máscara ou desconhecida ([UNK]) sorteia outra palavra
                while (palavra_mascarada.lower() == token_predito.lower()) or (token_predito == self.getTransformer().getTokenDesconhecido()):
                    # Sorteia uma predição do intervalo
                    posicao_sorteio = randint(0, top_k_predicao-1)    
                    #print("posicao_sorteio:",posicao_sorteio)

                    # Recupera as predições    
                    # Mapeia os índices do vocabulário para os seus tokens
                    token_predito = self.getTokenizer().convert_ids_to_tokens([top_k_predicao_indices[posicao_sorteio]])[0]
                    # Recupera os pesos da predição
                    token_peso = top_k_predicao_pesos[posicao_sorteio]
                    #print((posicao_sorteio+1), "[MASK]: ", token_predito, " | peso:", float(token_peso))
            
            else:
                # Se existe somente uma predição, esta não pode ser igual a palavra mascarada,
                # portanto é necessário aumentar a quantidade de top_k predições para gerar uma predição diferente 
                # da palavra mascarada.
                        
                # Recupera as top_k_predicao predições em ordem de orobabilidades
                top_k_predicao_pesos, top_k_predicao_indices = torch.topk(probabilidades, top_k_predicao, sorted=True)
                #print("top_k_predicao_pesos:",top_k_predicao_pesos)
                #print("top_k_predicao_indices:",top_k_predicao_indices)
                #print("len(top_k_predicao_indices):",len(top_k_predicao_indices))

                # Sorteia uma predição do intervalo
                posicao_sorteio = randint(0, top_k_predicao-1)    
                #print("posicao_sorteio:",posicao_sorteio)

                # Recupera as predições    
                # Mapeia os índices do vocabulário para os seus tokens
                token_predito = self.getTokenizer().convert_ids_to_tokens([top_k_predicao_indices[posicao_sorteio]])[0]
                # Recupera os pesos da predição
                token_peso = top_k_predicao_pesos[posicao_sorteio]
                #print((posicao_sorteio+1), "[MASK]: ", token_predito, " | peso:", float(token_peso))

                # Se o token predito for igual a palavra que foi substituída pela máscara ou desconhecida ([UNK]) sorteia outra palavra
                while (palavra_mascarada.lower() == token_predito.lower()) or (token_predito == self.getTransformer().getTokenDesconhecido()):
                    
                    # Incrementa a quantidade de predições para pegar uma palavra diferente
                    top_k_predicao = top_k_predicao + 1

                    # Recupera as top_k_predicao + 1 predições em ordem de orobabilidades
                    top_k_predicao_pesos, top_k_predicao_indices = torch.topk(probabilidades, top_k_predicao, sorted=True)
                    #print("top_k_predicao_pesos:",top_k_predicao_pesos)
                    #print("top_k_predicao_indices:",top_k_predicao_indices)
                    #print("len(top_k_predicao_indices):",len(top_k_predicao_indices))

                    # Sorteia uma predição do intervalo
                    posicao_sorteio = randint(0, top_k_predicao-1)    
                    #print("posicao_sorteio:",posicao_sorteio)

                    # Recupera as predições    
                    # Mapeia os índices do vocabulário para os seus tokens
                    token_predito = self.getTokenizer().convert_ids_to_tokens([top_k_predicao_indices[posicao_sorteio]])[0]
                    # Recupera os pesos da predição
                    token_peso = top_k_predicao_pesos[posicao_sorteio]
                    #print((posicao_sorteio+1), "[MASK]: ", token_predito, " | peso:", float(token_peso))

            token_predito_marcado = token_predito

            # Se o token tiver token separador
            if (self.getTransformer().getSeparadorSubToken() != None) and (sself.getTransformer().getSeparadorSubToken() in token_predito):
                # Remove os caracteres SEPARADOR_SUBTOKEN do token
                token_predito = token_predito.replace(self.getTransformer().getSeparadorSubToken(), "")                                        

            # Lista das predições
            lista_predicoes = []
            for i, indice_predicao in enumerate(top_k_predicao_indices):
                # Mapeia os índices do vocabulário para os seus tokens
                token_predito1 = self.getTokenizer().convert_ids_to_tokens([indice_predicao])[0]
                token_peso1 = top_k_predicao_pesos[i]
                lista_predicoes.append([(i+1), token_predito1, float(token_peso1)])        
            
            return texto_mascarado, palavra_mascarada, token_predito, token_peso, posicao_sorteio, token_predito_marcado, lista_predicoes
        
        else:
            logger.error("O modelo \"{}\" não possui um token de máscara.".format(self.getModel().__class__.__name__))
            
            return None        

    # ============================
    def getPerturbacaoTextoAleatorio(self, texto, 
                                     texto_token, 
                                     texto_pos, 
                                     classe=["VERB","NOUN","AUX"], 
                                     qtde=1, 
                                     top_k_predicao = 500):

        ''' 
        Gera um texto com a perturbação com seleção aleatória da palavra perturbada.
        Considera determinadas classes morfossintática das palavras.
            
        Parâmetros:
           `texto` - Texto a ser mascarado.
           `texto_token` - Lista com os tokens do texto.
           `texto_pos` - Lista com as POS dos tokens do texto.
           `classe` - Lista com as classes morfossintática das palavras a serem mascarada com [MASK].
           `qtde` - Quantidade de mascarada a serem realizadas nas palavras das sentenças.
                    Seleciona aleatoriamente a(s) palavra(s) a ser(em) mascarada(s) se a qtde 
                    for menor que quantidade de palavras das classes na sentença.
           `top_k_predicao` - Quantidade de palavras a serem recuperadas mais próximas da máscara.                

        Retorno:    
           `texto_perturbado` - Texto com a perturbação.
           `texto_mascarado` - Texto mascarado.
           `palavra_mascarada` - Palavra substituídas pela máscara.
           `token_predito` - Palavra prevista para a máscara.
           `lista_predicoes` - Lista dos tokens preditos para a máscara.                
        '''
        
        # Somente modelos que possuem token de mascara.
        if self.getTransformer().getTokenMascara() != None:

            # Recupera o texto mascarado e o token pervisto
            texto_mascarado, palavra_mascarada, token_predito, peso_predito, posicao_sorteio, lista_predicoes = self.getPerturbacaoPalavraTextoAleatorio(texto, texto_token, texto_pos, classe, qtde, top_k_predicao)
            
            # Se existir o token especial [MASK]
            if self.getTransformer().getTokenMascara() in texto_mascarado:
                
                # Substituir a mascará pelo token predito
                texto_perturbado = texto_mascarado.replace(self.getTransformer().getTokenMascara(), token_predito)
            
            return texto_perturbado, texto_mascarado, palavra_mascarada, token_predito, lista_predicoes
    
        else:
            logger.error("O modelo \"{}\" não possui um token de máscara.".format(self.getModel().__class__.__name__))
            
            return None
        
    # ============================  
    def getPerturbacaoPalavraTextoSequencial(self, texto, 
                                             texto_token, 
                                             texto_pos, 
                                             classe=["VERB","NOUN","AUX"], 
                                             qtde=1, 
                                             top_k_predicao = 500):
        ''' 
        Gera a palavras da perturbação do texto com seleção das top_k predições(em sequencia).        
        Considera determinadas classes morfossintática das palavras.
            
        Parâmetros:
           `texto` - Texto a ser mascarada.
           `texto_token` - Lista com os tokens do texto.
           `texto_pos` - Lista com as POS dos tokens do texto.
           `classe` - Lista com as classes morfossintática das palavras a serem mascarada com [MASK].
           `qtde` - Quantidade de mascarada a serem realizadas nas palavras do texto.
                Seleciona aleatoriamente a(s) palavra(s) a ser(em) mascarada(s) se a qtde 
                for menor que quantidade de palavras das classes na sentença.          
           `top_k_predicao` - Quantidade de palavras a serem recuperadas mais próximas da máscara.

        Retorno:    
           `texto_mascarado` - Texto mascarada.
           `palavra_mascarada` - Palavra substituídas pela máscara.
           `token_predito` - Palavra prevista para a máscara.
           `peso_predito` - Peso da palavra prevista.
           `lista_previsoes` - Lista dos 'top_k_predicao' tokens preditos para a máscara.
        '''
    
        # Somente modelos que possuem token de mascara.
        if self.getTransformer().getTokenMascara() != None:
            
            #print("Texto:", texto)
            texto_mascarado, palavra_mascarada = self.getTextoMascarado(texto,
                                                                        texto_token,
                                                                        texto_pos, 
                                                                        classe=classe, 
                                                                        qtde=qtde)

            # Divide as palavras em tokens
            texto_tokenizado = self.getTransformer().getTextoTokenizado(texto)
            #print("texto_tokenizado:", texto_tokenizado)

            # Retorna o índice da mascara de atenção
            mascara_atencao_indice = texto_tokenizado.index(self.getTransformer().getTokenMascara())
            #print("mascara_atencao_indice:", mascara_atencao_indice)

            # Mapeia os tokens em seus índices do vocabulário
            tokens_indexados =  self.getTokenizer().convert_tokens_to_ids(texto_tokenizado)
            #print("tokens_indexados:", tokens_indexados)
            
            # Converte as entradas de lista para tensores do torch
            tokens_tensores = torch.tensor([tokens_indexados])
            
            # Realiza a predição dos tokens
            with torch.no_grad():
                # Retorno de model quando ´output_hidden_states=True´ é setado:  
                #outputs[0] = last_hidden_state, outputs[1] = pooler_output, outputs[2] = hidden_states
                outputs =  self.getModel()(tokens_tensores)

            # Recupera a predição com os embeddings da última camada oculta    
            predicao = outputs[0]
            
            # Normaliza os pesos das predições nos embeddings e calcula sua probabilidade
            probabilidades = torch.nn.functional.softmax(predicao[0, mascara_atencao_indice], dim=-1)    
            
            # Retorna os k maiores elementos de determinado tensor de entrada ao longo de uma determinada 
            # dimensão de forma ordenada descrescentemente.    
            # Adiciona 20 elementos em topkpredicao para pular os tokens desconhecidos([UNK])
            MARGEM_UNK = 20
            top_k_predicao_pesos, top_k_predicao_indices = torch.topk(probabilidades, top_k_predicao + MARGEM_UNK, sorted=True)
            #print("top_k_predicao_pesos:",top_k_predicao_pesos)
            #print("top_k_predicao_indices:",top_k_predicao_indices)
            #print("len(top_k_predicao_indices):",len(top_k_predicao_indices))

            # Lista das predições
            lista_predicoes = []
            indice_token = 0
            for i, indice_predicao in enumerate(top_k_predicao_indices):

                # Mapeia os índices do vocabulário para os seus tokens
                token_predito =  self.getTokenizer().convert_ids_to_tokens([indice_predicao])[0]
                token_peso = top_k_predicao_pesos[i]

                # Pula o token se for desconhecido e existir tokens disponíveis
                if token_predito != self.getTransformer().getTokenDesconhecido() and indice_token < (top_k_predicao):
                
                    # Guarda o token original        
                    token_predito_marcado = token_predito
                
                    # Se o token tiver token separador
                    if (self.getTransformer().getSeparadorSubToken() != None) and (sself.getTransformer().getSeparadorSubToken() in token_predito):
                        # Remove os caracteres SEPARADOR_SUBTOKEN do token
                        token_predito = token_predito.replace(self.getTransformer().getSeparadorSubToken(), "")

                    # Guarda o token
                    lista_predicoes.append([indice_token, texto_mascarado, palavra_mascarada, token_predito, float(token_peso), token_predito_marcado])

                    # Incrementa para o próximo token
                    indice_token = indice_token + 1
            
            return lista_predicoes

        else:
            logger.error("O modelo \"{}\" não possui um token de máscara.".format(self.getModel().__class__.__name__))
            
            return None
        
    # ============================
    def getPerturbacaoTextoSequencial(self, texto,
                                      texto_token, 
                                      texto_pos, 
                                      classe=["VERB","NOUN","AUX"], 
                                      qtde=1, 
                                      top_k_predicao = 500):

        ''' 
        Gera o texto com a perturbação com seleção sequencial da palavra perturbada.
        Considera determinadas classes morfossintática das palavras.
            
        Parâmetros:
           `texto` - Texto a ser mascarada.
           `texto_token` - Lista com os tokens do texto.
           `texto_pos` - Lista com as POS dos tokens do texto.
           `classe` - Lista com as classes morfossintática das palavras a serem mascarada com [MASK].
           `qtde` - Quantidade de mascarada a serem realizadas nas palavras do texto.
                    Seleciona aleatoriamente a(s) palavra(s) a ser(em) mascarada(s) se a qtde 
                    for menor que quantidade de palavras das classes no texto.
           `top_k_predicao` - Quantidade de palavras a serem recuperadas mais próximas da máscara.                

        Retorno:    
            `texto_perturbada` - Texto com a perturbação.
            `texto_mascarada` - Texto mascarado.
            `palavra_mascarada` - Palavra substituídas pela máscara.
            `token_predito` - Palavra prevista para a máscara.
            `lista_predicoes` - Lista dos tokens preditos para a máscara.
                
        '''
        
        # Somente modelos que possuem token de mascara.
        if self.getTransformer().getTokenMascara() != None:

            # Recupera o texto mascarado e o token pervisto
            texto_mascarado, palavra_mascarada, token_predito, peso_predito, lista_predicoes = self.getPerturbacaoPalavraTextoSequencial(texto, texto_token, texto_pos, classe, qtde, top_k_predicao)
            
            # Se existir o token de mascará no texto
            if self.getTransformer().getTokenMascara() in texto_mascarado:
                
                # Substituir a mascará pelo token predito
                texto_perturbado = texto_mascarado.replace(self.getTransformer().getTokenMascara(), token_predito)
            
            return texto_perturbado, texto_mascarado, palavra_mascarada, token_predito, lista_predicoes
        
        else:
            logger.error("O modelo \"{}\" não possui um token de máscara.".format(self.getModel().__class__.__name__))
            
            return None
    
    # ============================
    def getInfoModel(self):
        '''
        Mostra informações do modelo.
        '''
        
        print("Modelo Huggingface  : {}.".format(self.auto_model.__class__.__name__))
        print("Modelo pré-treinado : {}.".format(self.auto_model.config._name_or_path))
        print("Parâmetros          : {:,}.".format(self.auto_model.num_parameters()))
        print("Tamanho embedding   : {:,}.".format(self.auto_model.config.hidden_size))
        print("Texto-Transformer   : {}.".format(self.getTransformer().__class__.__name__))
    
    # ============================
    def getModel(self):
        '''
        Recupera o modelo.
        '''
        
        return self.auto_model

    # ============================
    def getTokenizer(self):
        '''
        Recupera o tokenizador.
        '''
        
        return self.auto_tokenizer

    # ============================
    def getTransformer(self) -> Transformer:
        '''
        Recupera o transformer.
        '''
        
        return self.transformer

    # ============================
    def getMensurador(self) -> Mensurador:
        '''
        Recupera o mensurador.
        '''
        
        return self.mensurador        
        
    # ============================
    def getPln(self) -> PLN:
        '''
        Recupera o PLN.
        '''
        
        return self.pln