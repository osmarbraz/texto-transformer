# Import das bibliotecas.

# Biblioteca de logging
import logging  
# Biblioteca de aprendizado de máquina
from torch import nn 
import torch 
import numpy as np
from torch import Tensor, device
# Biblioteca de manipulação json
import json
# Biblioteca de tipos
from typing import List, Dict,  Union
# Biblioteca de manipulação sistema
import os
# Biblioteca do transformer hunggingface
from transformers import AutoModel, AutoTokenizer, AutoConfig
# Bibliteca das classes abstratas base
from abc import abstractmethod

# Bibliotecas próprias
from textotransformer.modelo.modeloargumentos import ModeloArgumentos
from textotransformer.modelo.modeloenum import AbordagemExtracaoEmbeddingsCamadas
from textotransformer.pln.pln import PLN

# Objeto de logger
logger = logging.getLogger(__name__)

# Constantes da classe
PALAVRA_FORA_DO_VOCABULARIO = 1
PALAVRA_DENTRO_DO_VOCABULARIO = 0

class Transformer(nn.Module):
    '''
    Classe base de Transformer que encapsula a classe AutoModel da Huggingface.
    Possui os métodos para gerar embeddings de token, palavra, sentença ou texto.
    '''

    def __init__(self, auto_model: AutoModel, 
                 auto_config: AutoConfig, 
                 auto_tokenizer: AutoTokenizer, 
                 modelo_args: ModeloArgumentos):
        '''
        Construtor da classe Transformer Base.

        Parâmetros:
            `auto_model` - Auto model modelo carregado.
            `auto_config` - Auto config carregado.
            `auto_tokenizer` - Auto tokenizer carregado.
            `modelo_args' - Argumentos passados para o modelo Huggingface Transformers.
        '''
        
        # Inicializa o construtor da superclasse
        super(Transformer, self).__init__()
      
        # Define os argumentos do modelo
        self.auto_model = auto_model
        
        # Define os argumentos do config modelo
        self.auto_config = auto_config
        
        # Define os argumentos do tokenizador modelo
        self.auto_tokenizer = auto_tokenizer
        
        # Define os argumentos do modelo
        self.modelo_args = modelo_args
        
    # ============================   
    def __repr__(self):
        '''
        Retorna uma string com descrição do objeto.
        '''

        return "Classe (\"{}\") carregada com o modelo \"{}\", m AutoConfig \"{}\", Transformer \"{}\" e tokenizador: \"{}\".".format(self.__class__.__name__,
                                                                                                                                      self.modelo_args.pretrained_model_name_or_path,
                                                                                                                                      self.auto_config.__class__.__name__,
                                                                                                                                      self.auto_model.__class__.__name__,
                                                                                                                                      self.auto_tokenizer.__class__.__name__)
    # ============================   
    @abstractmethod
    def defineTokensEspeciais(self):
        '''
        Define os tokens especiais e separadores considerando o modelo instânciado.
        
        # A maioria dos modelos a posição do token de início é 1 e o token separador é -1
        # Em alguns a posição do token de início é 0(não existe) e o token separador é -2 e o último <sep> é o token de classificação <CLS>
        '''
      
        # Sem um modelo especificado
        self.TOKEN_INICIO = None # Token de início de texto. Ex. [CLS].
        self.TOKEN_FIM = None # Token de fim. Ex. [SEP].
        self.TOKEN_SEPARADOR = None # Token separador de sentença. Ex. [SEP].
        self.TOKEN_CLASSIFICACAO = None # Token de classificação. Ex. [CLS].
        self.TOKEN_PADDING = None # Token de preenchimento. Ex. [PAD].
        self.PADDING_SIDE = 1 # Define o lado que será realizado o preenchimento das lista de tokens. 0: esquerda, 1: direita.
        self.TOKEN_MASCARA = None # Token de máscara. Ex. [MASK].
        self.TOKEN_DESCONHECIDO = None  # Token desconhecido. Ex. [UNK].
        self.SEPARADOR_SUBTOKEN = None # Separador de subtoken. Ex. ##, ou Ġ, </w>.
        self.POSICAO_TOKEN_INICIO = 0 # Posição primeiro do token válido do início da lista de tokens.
        self.POSICAO_TOKEN_FINAL = None # Posição último do token válido do final da lista de tokens. Valor "None" indica que é o último token.
        self.SERAPADOR_SUBTOKEN_REPETICAO = -1 # Repetição do separador subtoken. -1 - Sem separador subtoken, 0 - nos subtokens(menos primeiro), 1 - somente primeiro subtoken, 2 - somente último subtoken.
        self.SEPARADOR_SUBTOKEN_POSICAO = -1 # Posição do separador de subtoken. -1 - Sem separador de subtoken, 0 - no início do token,  1 - no fim do token.
        self.PRIMEIRO_TOKEN_SEM_SEPARADOR = False # Define se o primeiro token não terá separador de substoken. Ex. True - ['token1','Ġtoken2', 'Ġtoken3'] False - ['Ġtoken1','Ġtoken2', 'Ġtoken3'].
        self.DO_LOWER_CASE = False # Define se o tokenizador irá converter os tokens para minúsculo.
    
      
    # ============================ 
    def getTokenInicio(self) -> str:
        '''
        Recupera o token de início.

        Retorna:
           O token de início.
        '''
        
        return self.TOKEN_INICIO
    
    # ============================ 
    def getTokenFim(self) -> str:
        '''
        Recupera o token de fim.

        Retorna:
           O token de fim.
        '''
        
        return self.TOKEN_FIM
        
    # ============================ 
    def getPosicaoTokenInicio(self) -> int:
        '''
        Recupera a posição do token de início válido da lista de tokens.

        Retorna:
           Um inteiro com a posição do token de início válido da lista de tokens.
        '''
        
        return self.POSICAO_TOKEN_INICIO
    
    # ============================ 
    def getPosicaoTokenFinal(self) -> int:
        '''
        Recupera a posição do token de fim válido da lista de tokens.

        Retorna:
           Um inteiro com a posição do token de fim válido da lista de tokens.
        '''
        
        return self.POSICAO_TOKEN_FINAL
    
    # ============================   
    def getLadoPreenchimento(self) -> int:
        '''
        Recupera o lado de preenchimento da tag PAD.
        
        Retorna:
           0 - o preenchimento do token de pad a esquerda. 
           1 - o preenchimento do token de pad a direita. 
        '''

        return self.PADDING_SIDE
    
    # ============================   
    def getPrimeiroTokenSemSeparador(self) -> int:
        '''
        Recupera se o primeiro token do texto não possuo o token de separação.
        
        Retorna:
           True - O primeiro token não possui o token de separação.
           False - O primeiro token possui o token de separação.        
        '''

        return self.PRIMEIRO_TOKEN_SEM_SEPARADOR
    
    # ============================   
    def getTokenMascara(self) -> int:
        '''
        Recupera o token de máscara.
        
        Retorna:
          O token de máscara.
        '''

        return self.TOKEN_MASCARA    
    
    # ============================   
    def getTokenDesconhecido(self) -> int:
        '''
        Recupera o token desconhecido.
        
        Retorna:
          O token desconhecido.
        '''

        return self.TOKEN_DESCONHECIDO
    
    # ============================   
    def getSeparadorSubToken(self) -> int:
        '''
        Recupera o separador de subtoken.
        
        Retorna:
          O separador de subtoken.
        '''

        return self.SEPARADOR_SUBTOKEN
        
    # ============================      
    def getTextoTokenizado(self, texto : str,
                           addicionar_tokens_especiais: bool = True) -> List[str]:
        '''
        Retorna um texto tokenizado e concatenado com tokens especiais '[CLS]' no início e o token '[SEP]' no fim para ser submetido ao modelo de linguagem.
        
        Parâmetros:
           `texto` - Um texto a ser tokenizado.
        
        Retorno:
           `texto_tokenizado` - Texto tokenizado.
        '''

        # Tokeniza o texto
        saida = self.tokenize(texto, addicionar_tokens_especiais=addicionar_tokens_especiais)
        
        # Recupera o texto tokenizado da primeira posição, pois o texto vem em uma lista
        texto_tokenizado = saida['tokens_texto_mcl'][0]

        return texto_tokenizado

    # ============================    
    def removeTokensEspeciais(self, lista_tokens: List[str]) -> List[str]:
        '''
        Remove os tokens especiais de início, fim, separador e classificação  da lista de tokens.
        
        Parâmetros:
           `lista_tokens` - Uma lista de tokens.
        
        Retorno:
            Uma lista de tokens sem os tokens especiais.
        '''
        
        # Se possui token de início e faz parte da lista
        if self.TOKEN_INICIO != None and self.TOKEN_INICIO in lista_tokens:
             lista_tokens.remove(self.TOKEN_INICIO)
    
        # Se possui token de início e faz parte da lista
        if self.TOKEN_FIM != None and self.TOKEN_FIM in lista_tokens:
             lista_tokens.remove(self.TOKEN_FIM)

        # Se possui token de separação da lista
        if self.TOKEN_SEPARADOR != None and self.TOKEN_SEPARADOR in lista_tokens:
             lista_tokens.remove(self.TOKEN_SEPARADOR)
           
        # Se possui token de separação da lista
        if self.TOKEN_CLASSIFICACAO != None and self.TOKEN_CLASSIFICACAO in lista_tokens:
             lista_tokens.remove(self.TOKEN_CLASSIFICACAO)
        
        return lista_tokens

    # ============================ 
    def tokenize(self, texto: Union[str, List[str]],
                 addicionar_tokens_especiais: bool = True) -> dict:
        '''        
        Tokeniza um texto para submeter ao modelo de linguagem. 
        Retorna um dicionário listas de mesmo tamanho para garantir o processamento em lote.
        Use a quantidade de tokens para saber até onde deve ser recuperado em uma lista de saída.
        Ou use attention_mask diferente de 1 para saber que posições devem ser utilizadas na lista.

        Parâmetros:
           `texto` - Texto é uma string ou uma lista de strings a serem tokenizados para o modelo de linguagem.
           `addicionar_tokens_especiais` - Adiciona os tokens especiais de início e separação no texto.
                          
        Retorna um dicionário com as seguintes chaves:
           `tokens_texto_mcl` - Uma lista com os textos tokenizados com os tokens especiais.
           `input_ids` - Uma lsta com os ids dos tokens de entrada mapeados em seus índices do vocabuário.
           `token_type_ids` - Uma lista com os tipos dos tokens.
           `attention_mask` - Uma lista com os as máscaras de atenção indicando com '1' os tokens  pertencentes à sentença.
        '''
        
        # Dicionário com a saída do tokenizador
        saida = {}
        
        # Se o texto for uma string coloca em uma lista de listas para tokenizar
        if isinstance(texto, str):
            to_tokenize = [[texto]]
        else:
            # Se for uma lista de strings coloca em uma lista para tokenizar
            if isinstance(texto[0], str):
                to_tokenize = [texto]
            else:
                # Se for uma lista de listas de strings, não faz nada
                to_tokenize = texto
                
        # Remove os espaços em branco antes e depois de cada texto usando strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Se for para colocar para minúsculo usa Lowercase nos textos
        if self.DO_LOWER_CASE:
            # Convertendo todos os tokens para minúsculo
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        # Tokeniza o texto
        # Faz o mesmo que o método encode_plus com uma string e o mesmo que batch_encode_plus com uma lista de strings.
        saida.update(self.auto_tokenizer(*to_tokenize,  # Texto a ser codificado. O '*' remove a lista de listas de to_tokenize.
                                         add_special_tokens=addicionar_tokens_especiais, # Adiciona os tokens especiais '[CLS]' e '[SEP]'.
                                         padding=True, # Preenche o texto até max_length.
                                         truncation='longest_first',  # Trunca o texto no maior texto.
                                         return_tensors="pt",  # Retorna os dados como tensores pytorch.
                                         max_length=self.modelo_args.max_seq_len # Define o tamanho máximo para preencheer ou truncar.
                                        ) 
                    )
                        
        # Gera o texto tokenizado convertendo os ids para os respectivos tokens           
        saida['tokens_texto_mcl'] = [[self.auto_tokenizer.convert_ids_to_tokens(s.item()) for s in col] for col in saida['input_ids']]

        # Guarda o texto original        
        saida['texto_original'] = [[s for s in col] for col in to_tokenize][0]     
        
        # Verifica se existe algum texto maior que o limite de tokenização
        for tokens in saida['tokens_texto_mcl']:
            if len(tokens) >= 512:
                logger.info("Utilizando embeddings do modelo de:\"{}\".".format(AbordagemExtracaoEmbeddingsCamadas.converteInt(self.modelo_args.abordagem_extracao_embeddings_camadas).getStr()))
  
        return saida
        
    # ============================           
    def getSaidaRede(self, texto: dict) -> dict:
        '''
        De um texto preparado(tokenizado) retorna os embeddings dos tokens do texto. 
        O retorno é um dicionário com token_embeddings, input_ids, attention_mask, token_type_ids, 
        tokens_texto_mcl, texto_original  e all_layer_embeddings.
        
        Retorna os embeddings de todas as camadas de um texto.
    
        Parâmetros:
           `texto` - Um texto tokenizado a ser recuperado os embeddings do modelo de linguagem
    
        Retorna um dicionário com as seguintes chaves:
           `token_embeddings` - Uma lista com os embeddings da última camada.
           `input_ids` - Uma lista com os textos indexados.            
           `attention_mask` - Uma lista com os as máscaras de atenção
           `token_type_ids` - Uma lista com os tipos dos tokens.            
           `tokens_texto_mcl` - Uma lista com os textos tokenizados com os tokens especiais.
           `texto_origina`l - Uma lista com os textos originais.
           `all_layer_embeddings` - Uma lista com os embeddings de todas as camadas.
        '''
    
        # Recupera o texto preparado pelo tokenizador para envio ao modelo
        dic_texto_tokenizado = {'input_ids': texto['input_ids'],                                 
                                'attention_mask': texto['attention_mask']}
        
        # Se token_type_ids estiver no texto preparado copia para dicionário tokenizado
        # Alguns modelos como o Roberta não utilizam token_type_ids
        if 'token_type_ids' in texto:
            dic_texto_tokenizado['token_type_ids'] = texto['token_type_ids']

        # Roda o texto através do modelo, e coleta todos os estados ocultos produzidos.
        outputs = self.auto_model(**dic_texto_tokenizado, 
                                  return_dict=False)
        
        # A avaliação do modelo retorna um número de diferentes objetos com base em
        # como é configurado na chamada do método `from_pretrained` anterior. Nesse caso,
        # porque definimos `output_hidden_states = True`, o terceiro item será o
        # estados ocultos(hidden_states) de todas as camadas. Veja a documentação para mais detalhes:
        # https://huggingface.co/transformers/model_doc/bert.html#bertmodel

        # Retorno de model quando ´output_hidden_states=True´ é setado:    
        # outputs[0] = last_hidden_state, outputs[1] = pooler_output, outputs[2] = hidden_states
        
        # hidden_states é uma lista python, e cada elemento um tensor pytorch no formado <lote> x <qtde_tokens> x <768 ou 1024>.        
        # 0-texto_tokenizado, 1-input_ids, 2-attention_mask, 3-token_type_ids, 4-outputs(0=last_hidden_state,1=pooler_output,2=hidden_states)
        
        # Recupera a última camada de embeddings da saida do modelo
        last_hidden_state = outputs[0]

        # Adiciona os embeddings da última camada e os dados do texto preparado na saída
        saida = {}
        saida.update({'token_embeddings': last_hidden_state,  # Embeddings da última camada
                      'input_ids': texto['input_ids'],
                      'attention_mask': texto['attention_mask'],
                      'tokens_texto_mcl': texto['tokens_texto_mcl'],
                      'texto_original': texto['texto_original']
                      }
                    )

        # Se output_hidden_states == True existem embeddings nas camadas ocultas
        if self.auto_model.config.output_hidden_states:
            # 2 é o índice da saída com todos os embeddings em outputs
            all_layer_idx = 2
            if len(outputs) < 3: #Alguns modelos apenas geram last_hidden_states e all_hidden_states
                all_layer_idx = 1

            # Recupera todos as camadas do transformer
            # Tuplas com cada uma das camadas
            hidden_states = outputs[all_layer_idx]
            
            # Adiciona os embeddings de todas as camadas na saída
            saida.update({'all_layer_embeddings': hidden_states})

        return saida

    # ============================
    def getEmbeddingPrimeiraCamadaRede(self, saida_rede: dict) -> list:
        '''
        Retorna os embeddings extraído da primeira camada do transformer.

        Parâmtros:
           `saida_rede` - Um dicionário com a saída da rede.

        Retorno:
           Uma lista com os embeddings.
        '''
        
        # Retorna toda a primeira(0) camada da saida da rede.
        # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)          
        resultado = saida_rede['all_layer_embeddings'][0]
        # Retorno: (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
        #print('resultado=',resultado.size())

        return resultado

    # ============================
    def getEmbeddingPenultimaCamada(self, saida_rede: dict) -> list:
        '''
        Retorna os embeddings extraído da penúltima camada do transformer.
        
        Parâmtros:
           `saida_rede` - Um dicionário com a saída da rede.

        Retorno:
           Uma lista com os embeddings.
        '''
        # Retorna todas a penúltima(-2) camada
        # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
        resultado = saida_rede['all_layer_embeddings'][-2]
        # Retorno: (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
        #print('resultado=',resultado.size())

        return resultado

    # ============================
    def getEmbeddingUltimaCamada(self, saida_rede: dict) -> list:
        '''
        Retorna os embeddings extraído da última camada do transformer.
        
        Parâmtros:
           `saida_rede` - Um dicionário com a saída da rede.

        Retorno:
           Uma lista com os embeddings.
        '''
        # Retorna todas a última(-1) camada
        # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
        resultado = saida_rede['all_layer_embeddings'][-1]
        # Retorno: (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
        #print('resultado=',resultado.size())

        return resultado        

     # ============================
    def getEmbeddingSoma4UltimasCamadas(self, saida_rede: dict) -> list:
        '''        
        Retorna a soma dos embeddings extraído das 4 últimas camada do transformer.
     
        Parâmtros:
           `saida_rede` - Um dicionário com a saída da rede.

        Retorno:
           Uma lista com os embeddings.
        '''
        
        # Retorna todas as 4 últimas camadas
        # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
        embedding_camadas = saida_rede['all_layer_embeddings'][-4:]
        # Retorno: List das camadas(4) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  

        # Usa o método `stack` para criar uma nova dimensão no tensor 
        # com a concateção dos tensores dos embeddings.        
        # Entrada: List das camadas(4) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
        resultado_stack = torch.stack(embedding_camadas, dim=0)
        # Retorno: <4> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
        #print('resultado_stack=',resultado_stack.size())
      
        # Realiza a soma dos embeddings de todos os tokens para as camadas
        # Entrada: <4> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
        resultado = torch.sum(resultado_stack, dim=0)
        # Saida: <1(lote)> x <qtde_tokens> x <768 ou 1024>
        #print('resultado=',resultado.size())

        return resultado

    # ============================
    def getEmbeddingConcat4UltimasCamadas(self, saida_rede: dict) -> list:
        '''        
        Retorna a concatenação dos embeddings das 4 últimas camadas do transformer.
             
        Parâmtros:
           `saida_rede` - Um dicionário com a saída da rede.

        Retorno:
           Uma lista com os embeddings.
        '''
        
        # Cria uma lista com os tensores a serem concatenados
        # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
        # Lista com os tensores a serem concatenados
        lista_concatenada = []
        
        # Percorre os 4 últimos tensores da lista(camadas)
        for i in [-1, -2, -3, -4]:
            # Concatena da lista
            lista_concatenada.append(saida_rede['all_layer_embeddings'][i])
            
        # Retorno: Entrada: List das camadas(4) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
        #print('lista_concatenada=',len(lista_concatenada))

        # Realiza a concatenação dos embeddings de todos as camadas
        # Retorno: Entrada: List das camadas(4) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  
        resultado = torch.cat(lista_concatenada, dim=-1)
        # Retorno: Entrada: (<1(lote)> x <qtde_tokens> <3072 ou 4096>)  
        # print('resultado=',resultado.size())
      
        return resultado

    # ============================
    def getEmbeddingSomaTodasAsCamada(self, saida_rede: dict) -> list:
        '''
        Retorna a soma dos embeddings extraído de todas as camadas do transformer.
                   
        Parâmtros:
           `saida_rede` - Um dicionário com a saída da rede.

        Retorno:
           Uma lista com os embeddings.
        '''
      
        # Retorna todas as camadas descontando a primeira(0)
        # Entrada: List das camadas(13 ou 25) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
        embedding_camadas = saida_rede['all_layer_embeddings'][1:]
        # Retorno: List das camadas(12 ou 24) (<1(lote)> x <qtde_tokens> <768 ou 1024>)  

        # Usa o método `stack` para criar uma nova dimensão no tensor 
        # com a concateção dos tensores dos embeddings.        
        # Entrada: List das camadas(12 ou 24) (<1(lote)> x <qtde_tokens> x <768 ou 1024>)  
        resultado_stack = torch.stack(embedding_camadas, dim=0)
        # Retorno: <12 ou 24> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
        #print('resultado_stack=',resultado_stack.size())
      
        # Realiza a soma dos embeddings de todos os tokens para as camadas
        # Entrada: <12 ou 24> x <1(lote)> x <qtde_tokens> x <768 ou 1024>
        resultado = torch.sum(resultado_stack, dim=0)
        # Saida: <1(lote)> x <qtde_tokens> x <768 ou 1024>
        # print('resultado=',resultado.size())
      
        return resultado

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
                
        # Verifica o tipo de dado do parâmetro 'abordagem_extracao_embeddings_camadas'
        if isinstance(abordagem_extracao_embeddings_camadas, int):
            
            # Converte o parâmetro estrategia_pooling para um objeto da classe AbordagemExtracaoEmbeddingsCamadas
            abordagem_extracao_embeddings_camadas = AbordagemExtracaoEmbeddingsCamadas.converteInt(abordagem_extracao_embeddings_camadas)
        
        # Recupera todos os embeddings da rede('all_layer_embeddings')
        saida_rede = self.getSaidaRede(texto)
                
        # Embedding extraído usando a abordagem de extração
        embedding_extraido_abordagem = None

        # Chama o método que recupera os embeddings da camada especificada
        if abordagem_extracao_embeddings_camadas == AbordagemExtracaoEmbeddingsCamadas.PRIMEIRA_CAMADA:
            embedding_extraido_abordagem = self.getEmbeddingPrimeiraCamadaRede(saida_rede)
        elif abordagem_extracao_embeddings_camadas == AbordagemExtracaoEmbeddingsCamadas.PENULTIMA_CAMADA:
            embedding_extraido_abordagem = self.getEmbeddingPenultimaCamada(saida_rede)
        elif abordagem_extracao_embeddings_camadas == AbordagemExtracaoEmbeddingsCamadas.ULTIMA_CAMADA:
            embedding_extraido_abordagem = self.getEmbeddingUltimaCamada(saida_rede)
        elif abordagem_extracao_embeddings_camadas == AbordagemExtracaoEmbeddingsCamadas.SOMA_4_ULTIMAS_CAMADAS:
            embedding_extraido_abordagem = self.getEmbeddingSoma4UltimasCamadas(saida_rede)
        elif abordagem_extracao_embeddings_camadas == AbordagemExtracaoEmbeddingsCamadas.CONCAT_4_ULTIMAS_CAMADAS:
            embedding_extraido_abordagem = self.getEmbeddingConcat4UltimasCamadas(saida_rede)
        elif abordagem_extracao_embeddings_camadas == AbordagemExtracaoEmbeddingsCamadas.TODAS_AS_CAMADAS:
            embedding_extraido_abordagem = self.getEmbeddingSomaTodasAsCamada(saida_rede)
        else:                                
            logger.error("Não foi especificado uma abordagem de extração dos embeddings das camadas do transformer.") 
        
        # Verifica se foi realizado a extração
        if embedding_extraido_abordagem != None:
          # Atualiza a saída com os embeddings extraídos usando abordagem
          saida_rede.update({'embedding_extraido': embedding_extraido_abordagem,  # Embeddings extraídos usando abordagem de extração
                             'abordagem_extracao_embeddings_camadas': abordagem_extracao_embeddings_camadas})  # Tipo da abordagem da extração  dos embeddings
        else:
          logger.error("Não foi especificado uma abordagem de extração dos embeddings das camadas do transformer.") 
          saida_rede = None  

        return saida_rede

    # ============================
    @abstractmethod
    def getTokensPalavrasEmbeddingsTexto(self, 
                                         embeddings_texto, 
                                         tokens_texto_mcl: list[str],
                                         tokens_texto_concatenado_pln: str,
                                         pln: PLN,
                                         dic_excecao: dict = {"":0,}) -> dict:
        '''
        As subclasses devem escolher um dos métodos para retornar a tokenização de palavras:
        - getTokensPalavrasEmbeddingsTextoWordPiece
        - getTokensPalavrasEmbeddingsTextoSentencePiece
        - getTokensPalavrasEmbeddingsTextoBPE
        
        '''
        
        pass
    
    # ============================
    def _inicializaDicionarioExcecao(self, dic_excecao: dict = {"":0,}):
        '''
        Inicializa o dicionário utilizado pela tokenização de palavras.

        Parâmetros:           
           `dic_excecao` - Um dicionário de tokens de exceções e seus deslocamentos para considerar mais ou menos tokens do modelo de linguagem em relação ao spaCy. 
        '''
        
        self._dic_excecao = dic_excecao
        
    # ============================      
    def _getExcecaoDicMenor(self, token: str): 
        '''
        Retorna o deslocamento do token no texto para considerar mais ou menos tokens do MCL em relação a tokenização de PLN.

        Parâmetros:
           `token` - Um token a ser verificado se é uma exceção.

        Retorno:
           O deslocamento do token no texto para considerar menos tokens do MCL em relação a tokenização da ferramenta de PLN.
        '''
        
        valor = self._dic_excecao.get(token)
        if valor != None:
            return valor
        else:
            return 0

    # ============================
    def _procuraTokenDicionario(self, wi_pln: str, 
                                lista_tokens_texto_pln: list[str], 
                                pos_wi_pln: int):
        '''
        Retorna a posição do token de exceção no dicionário.

        Parâmetros:
           `wi_pln` - Um token a ser verificado no dicionário.
           `lista_tokens_texto_pln` - Lista dos tokens que podem fazer parte da exceção.
           `pos_wi_pln` - Posição do token a ser verificado na lista de tokens.

        Retorno:
           `wi_pln_pos_excecao' - O token concatenado encotrado no dicionário.
           `pos_pln_excecao' - O deslocamento da exceção no dicionário.
           `pos_excecao` - A posição do último token encontrado no dicionário.
        '''
        
        wi_pln_pos_excecao = ""
        pos_pln_excecao = 0
        
        # Indice de deslocamento da exceção
        pos_excecao = pos_wi_pln
        
        # Procura no dicionário a concatenação dos tokens seguintes
        while (pos_pln_excecao == 0) and (pos_excecao < len(lista_tokens_texto_pln)):
            # Recupera o token da palavra gerado pelo ferramenta de PLN
            wi_pln_pos = lista_tokens_texto_pln[pos_excecao]
            
            # Concatena o token com o anterior
            wi_pln_pos_excecao = wi_pln_pos_excecao + wi_pln_pos
            
            # Localiza o deslocamento da exceção no dicionário      
            pos_pln_excecao = self._getExcecaoDicMenor(wi_pln_pos_excecao)
            
            # Se não encontrou a exceção, incrementa o deslocamento
            pos_excecao = pos_excecao + 1
        
        return  wi_pln_pos_excecao, pos_pln_excecao, pos_excecao
    
    # ============================  
    def _verificaSituacaoListaPalavras(self, mensagem,
                                       tokens_texto_concatenado_pln,
                                       lista_tokens, 
                                       lista_tokens_texto_pln,
                                       lista_pos_texto_pln,
                                       lista_tokens_texto_mcl,
                                       lista_tokens_oov_mcl, 
                                       lista_palavra_embeddings_MEAN, 
                                       lista_palavra_embeddings_MAX):
        '''
        Verifica se as listas geradas pelo método de gerar embedding de palavras estão com o mesmo tamanho e conteúdo.
        
        Parâmetros:
           `mensagem` - Mensagem de erro a ser exibida.
           `tokens_texto_concatenado_pln` - Texto concatenado e tokenizado pelo PLN.
           `lista_tokens` - Lista dos tokens de palavras.
           `lista_tokens_texto_pln` - Lista dos tokens de palavras do texto tokenizado pelo PLN.
           `lista_pos_texto_pln` - Lista das POS das palavras do texto tokenizado pelo PLN.
           `lista_tokens_texto_mcl` - Lista dos tokens de palavras do texto tokenizado pelo MCL.
           `lista_tokens_oov_mcl` - Lista das palavras OOV do texto tokenizado pelo MCL.
           `lista_palavra_embeddings_MEAN` - Lista dos embeddings das palavras calculados pela média dos embeddings dos tokens.
           `lista_palavra_embeddings_MAX` - Lista dos embeddings das palavras calculados pelo máximo dos embeddings dos tokens.
        '''
        
        if (lista_tokens[0] != lista_tokens_texto_pln[0]) or (lista_tokens[-1] != lista_tokens_texto_pln[-1]):
            logger.error("Mensagem:                   :{}.".format(mensagem))
            logger.error("texto                       :{}.".format(tokens_texto_concatenado_pln))
            logger.error("lista_tokens              {:2d}:{}.".format(len(lista_tokens), lista_tokens))            
            logger.error("lista_tokens_texto_pln    {:2d}:{}.".format(len(lista_tokens_texto_pln), lista_tokens_texto_pln))
            logger.error("lista_pos_texto_pln       {:2d}:{}.".format(len(lista_pos_texto_pln),lista_pos_texto_pln))
            logger.error("lista_tokens_texto_mcl    {:2d}:{}.".format(len(lista_tokens_texto_mcl),lista_tokens_texto_mcl))            
            logger.error("lista_tokens_oov_mcl      {:2d}:{}.".format(len(lista_tokens_oov_mcl), lista_tokens_oov_mcl))
            logger.error("len(lista_embeddings_MEAN){:2d}:.".format(len(lista_palavra_embeddings_MEAN)))
            logger.error("len(lista_embeddings_MAX) {:2d}:.".format(len(lista_palavra_embeddings_MAX)))
    
    # ============================
    # getTokensPalavrasEmbeddingsTextoWordPiece
    # Gera os tokens, POS e embeddings de cada texto.
    
    def getTokensPalavrasEmbeddingsTextoWordPiece(self,
                                                  embeddings_texto,
                                                  lista_tokens_texto_mcl: list[str],
                                                  tokens_texto_concatenado_pln: str,
                                                  pln: PLN,
                                                  dic_excecao: dict = {"":0,}) -> dict:
        '''
        De um texto tokenizado pelo algoritmo WordPiece e seus embeddings retorna os embeddings das palavras segundo a ferramenta de PLN.
        
        Condidera os tratamentos de tokenização do MCL na ordem a seguir.  
        1 - Exceção 
            Procura o token e o próximo no dicionário.wi_pln_pos_excecao	
            1.1 - Exceção maior que 0(valores positivos) - Tokenização do MCL gera mais tokens que a PLN.
                Ex.: {"St.":2}, a ferramenta de pln tokeniza "St." em 1 token "St." e o MCL em dois tokens"St" e "." ou seja 2 tokens do MCL devem virar 1.

            1.2 - Exceção menor que 0(valores negativos) - Tokenização do MCL gera menos tokens que a PLN.
                Ex.: {"1°": -1}, a ferramenta de pln tokeniza "1°" em 2 tokens "1" e "°" e o MCL em um token "1°", ou seja 2 tokens do PLN devem virar 1.

        2 - Token PLN igual ao token MCL ou desconhecida(UNK) e que não possui separador no próximo subtoken do MCL
            Token do PLN é igual ao token do MCL adiciona diretamente na lista de tokens.
            
        3 - Palavra foi dividida(tokenizada), o próximo token MCL possui separador e não é o fim da lista de tokens do MCL
            3.1 - Palavra completa MCL é igual a palavra completa PLN
                Tokens da palavra adicionado a lista de tokens e embeddings dos tokens consolidados para gerar as palavras.
            3.2 - Palavra completa MCL diferente da palavra completa PLN
                Especificar exceção maior ou menor para tratar tokens de palavra não tokenizado em palavra.
                          
        Parâmetros:
           `texto` - Um texto a ser recuperado os embeddings das palavras do modelo de linguagem.
           `embeddings_texto` - Os embeddings do texto gerados pelo método getEmbeddingsTexto.
           `lista_tokens_texto_mcl` - Os tokens do texto gerados pelo método getEmbeddingsTexto.
           `tokens_texto_concatenado_pln` - Os tokens do texto concatenado gerados pela ferramenta de PLN.
           `pln` - Uma instância da classe PLN para realizar a tokenização e POS-Tagging do texto.
           `dic_excecao` - Um dicionário de tokens de exceções e seus deslocamentos para considerar mais ou menos tokens do modelo de linguagem em relação ao spaCy. Valor positivo para considerar mais tokens e negativo para considerar menos tokens. Exemplo exceção negativo: {"1°": -1}, a ferramenta de PLN separa o token "1°" em "1" e "°", portanto é necessário reduzir 1 token pois o MCL gera somente um. Exemplo exceção positiva: {"1°": 1}, a ferramenta de PLN não separa o token "1°", mas o MCL separa em dois "1" e "°" portanto é necessário agrupar em 1 token.
               
        Retorna um dicionário com as seguintes chaves: 
           `tokens_texto` - Uma lista com os tokens do texto gerados pelo método.
           `pos_texto_pln` - Uma lista com as postagging dos tokens gerados pela ferramenta de PLN.
           `tokens_oov_texto_mcl` - Uma lista com os tokens OOV do MCL.
           `palavra_embeddings_MEAN` - Uma lista dos embeddings de palavras com a média dos embeddings(Estratégia MEAN) dos tokens que formam a palavra.
           `palavra_embeddings_MAX` - Uma lista dos embeddings de palavras com o máximo dos embeddings(Estratégia MAX) dos tokens que formam a palavra.
        '''
        
        # Inicializa os dicionários de exceção
        self._inicializaDicionarioExcecao(dic_excecao=dic_excecao)
       
        # Guarda os tokens e embeddings de retorno
        lista_tokens = []
        lista_tokens_oov_mcl = []
        lista_palavra_embeddings_MEAN = []
        lista_palavra_embeddings_MAX = []
        
        # Gera a tokenização e POS-Tagging da sentença    
        lista_tokens_texto_pln, lista_pos_texto_pln = pln.getListaTokensPOSTexto(texto=tokens_texto_concatenado_pln)

        # print("\tokens_texto_concatenado    :",tokens_texto_concatenado)    
        # print("lista_tokens_texto_pln       :",lista_tokens_texto_pln)
        # print("len(lista_tokens_texto_pln)  :",len(lista_tokens_texto_pln))    
        # print("lista_pos_texto_pln          :",lista_pos_texto_pln)
        # print("len(lista_pos_texto_pln)     :",len(lista_pos_texto_pln))
        
        # embedding <qtde_tokens x 4096>        
        # print("embeddings_texto             :",embeddings_texto.shape)
        # print("lista_tokens_texto_mcl       :",lista_tokens_texto_mcl)
        # print("len(lista_tokens_texto_mcl)  :",len(lista_tokens_texto_mcl))

        # Seleciona os pares de palavra a serem avaliadas
        pos_wi_pln = 0 # Posição do token da palavra gerado pela ferramenta de PLN
        pos_wj_mcl = pos_wi_pln # Posição do token da palavra gerado pelo MCL

        # Enquanto o indíce da palavra pos_wj_mcl(2a palavra) não chegou ao final da quantidade de tokens do MCL
        while (pos_wj_mcl < len(lista_tokens_texto_mcl)) and (pos_wi_pln < len(lista_tokens_texto_pln)):  

            # Seleciona o token da sentença gerado pela ferramenta de PLN
            wi_pln = lista_tokens_texto_pln[pos_wi_pln] 
            # Recupera o token da palavra gerado pelo MCL
            wj_mcl = lista_tokens_texto_mcl[pos_wj_mcl] 
            # print("wi[",pos_wi_pln,"]=", wi_pln)
            # print("wj[",pos_wj_mcl,"]=", wj_mcl)
            
            # Procura o token e a concatenação dos seguintes no dicionário de exceções
            wi_pln_pos_excecao, pos_pln_excecao, pos_excecao = self._procuraTokenDicionario(wi_pln, 
                                                                                            lista_tokens_texto_pln, 
                                                                                            pos_wi_pln)
            
                        
            # Se existe uma exceção
            if pos_pln_excecao != 0:                
                #print("Exceção /pos_pln_excecao:", pos_pln_excecao, "/wi_pln_pos_excecao:", wi_pln_pos_excecao, "/pos_excecao:", pos_excecao)                
                # É uma exceção maior
                if pos_pln_excecao > 0:
                    #print("Adiciona 1 Exceção maior wi_pln:", wi_pln, "/ wi_pln_pos_excecao:", wi_pln_pos_excecao, "/pos_excecao:", pos_excecao)
                    lista_tokens.append(wi_pln)
                    # Marca como fora do vocabulário do MCL
                    lista_tokens_oov_mcl.append(PALAVRA_FORA_DO_VOCABULARIO)
                    # Verifica se tem mais de um token para pegar o intervalo
                    if wi_pln_pos_excecao != 1:
                        # Localiza o indíce final do token do MCL
                        indice_final_token = pos_wj_mcl + pos_pln_excecao                        
                        #print("Calcula a média de :", pos_wj_mcl , "até", indice_final_token)
                        
                        # Recupera os embeddings dos tokens do MCL da posição pos_wj até indice_final_token
                        embeddings_tokens_palavra = embeddings_texto[pos_wj_mcl:indice_final_token]
                        #print("embeddings_tokens_palavra:",embeddings_tokens_palavra.shape)
                        
                        if isinstance(embeddings_tokens_palavra, torch.Tensor): 
                            # calcular a média dos embeddings dos tokens do MCL da palavra
                            embedding_estrategia_MEAN = torch.mean(embeddings_tokens_palavra, dim=0)
                        else:
                            embedding_estrategia_MEAN = np.mean(embeddings_tokens_palavra, axis=0)
                            
                        #print("embedding_estrategia_MEAN:",embedding_estrategia_MEAN.shape)
                        lista_palavra_embeddings_MEAN.append(embedding_estrategia_MEAN)
                        
                        if isinstance(embeddings_tokens_palavra, torch.Tensor): 
                            # calcular o máximo dos embeddings dos tokens do MCL da palavra
                            embedding_estrategia_MAX, linha = torch.max(embeddings_tokens_palavra, dim=0)
                        else:
                            embedding_estrategia_MAX, linha = np.max(embeddings_tokens_palavra, axis=0)
                        
                        #print("embedding_estrategia_MAX:",embedding_estrategia_MAX.shape)
                        lista_palavra_embeddings_MAX.append(embedding_estrategia_MAX)
                    else:
                        # Adiciona o embedding do token a lista de embeddings
                        lista_palavra_embeddings_MEAN.append(embeddings_texto[wi_pln_pos_excecao])            
                        lista_palavra_embeddings_MAX.append(embeddings_texto[wi_pln_pos_excecao])
             
                    # Avança para o próximo token do PLN e MCL
                    pos_wi_pln = pos_wi_pln + 1
                    pos_wj_mcl = pos_wj_mcl + pos_pln_excecao
                    #print("Proxima:")            
                    #print("wi[",pos_wi_pln,"]=", lista_tokens_texto_pln[pos_wi_pln])
                    #print("wj[",pos_wi_mcl,"]=", lista_tokens_texto_mcl[pos_wi_mcl])
                else:
                    # É uma exceção menor
                    if (pos_pln_excecao < 0):
                        #print("Adiciona 2 Exceção menor wi_pln:", wi_pln, "/ wi_pln_pos_excecao:", wi_pln_pos_excecao, "/pos_excecao:", pos_excecao)
                        lista_tokens.append(wi_pln_pos_excecao)
                        # Marca como dentro do vocabulário do MCL
                        lista_tokens_oov_mcl.append(PALAVRA_DENTRO_DO_VOCABULARIO)
                        # 1 é o mínimo a ser considerado
                        if ((pos_pln_excecao * -1) == 1): 
                            # Adiciona o embedding do token a lista de embeddings
                            lista_palavra_embeddings_MEAN.append(embeddings_texto[pos_wj_mcl])
                            lista_palavra_embeddings_MAX.append(embeddings_texto[pos_wj_mcl])
                        else:
                            logger.error("Erro: Exceção menor WordPiece não pode ser maior que 1 token /wi_pln: \"{}\" wi_pln_pos_excecao: \"{}\" /pos_excecao: \"{}\".".format(wi_pln, wi_pln_pos_excecao, pos_excecao))
                            
                        # Avança para o próximo token do PLN e MCL nos deslocamentos de exceção
                        pos_wi_pln = pos_wi_pln + (pos_excecao - pos_wi_pln) # Avança os tokens do PLN considerando o deslocamento da exceção
                        pos_wj_mcl = pos_wj_mcl + (pos_pln_excecao * -1) # Avança os tokens do MCL considerando o deslocamento da exceção
                        #print("Proxima:")            
                        #print("wi[",pos_wi_pln,"]=", lista_tokens_texto_pln[pos_wi_pln])
                        #print("wj[",pos_wi_mcl,"]=", lista_tokens_texto_mcl[pos_wi_mcl])
            else:  
                # Tokens iguais ou desconhecido e não possui separador no próximo subtoken, adiciona diretamente na lista, pois o token não possui subtokens
                if ((wi_pln == wj_mcl) or (wj_mcl == self.TOKEN_DESCONHECIDO)) and (pos_wj_mcl+1 < len(lista_tokens_texto_mcl)) and (self.SEPARADOR_SUBTOKEN not in lista_tokens_texto_mcl[pos_wj_mcl+1]):
                    # Adiciona o token a lista de tokens
                    #print("Adiciona 3 (wi==wj or wj==TOKEN_DESCONHECIDO) e (próximo subtoken sem separador):", wi, wj)
                    lista_tokens.append(wi_pln)    
                    # Marca como dentro do vocabulário do MCL
                    lista_tokens_oov_mcl.append(PALAVRA_DENTRO_DO_VOCABULARIO)
                    # Adiciona o embedding do token a lista de embeddings
                    lista_palavra_embeddings_MEAN.append(embeddings_texto[pos_wj_mcl])
                    lista_palavra_embeddings_MAX.append(embeddings_texto[pos_wj_mcl])
                    #print("embedding1[pos_wj_mcl]:", embedding_texto[pos_wj_mcl].shape)
                    # Avança para o próximo token do PLN e MCL
                    pos_wi_pln = pos_wi_pln + 1
                    pos_wj_mcl = pos_wj_mcl + 1   
                  
                else:
                    # A palavra foi tokenizada pelo Wordpice com SEPARADOR_SUBTOKEN(Ex. ##) 
                    # ou diferente da ferramenta de PLN ou desconhecida
                    # Inicializa a palavra a ser completada com os subtokens do PLN e MCL       
                    #print("wi[",pos_wi_pln,"]=", wi_pln) # PLN
                    #print("wj[",pos_wj_mcl,"]=", wj_mcl) # MCL
                    palavra_completa_wi_pln = wi_pln
                    palavra_completa_wj_mcl = wj_mcl
                    # recupera o indíce do próximo do token do MCL
                    indice_proximo_token_wj_mcl = pos_wj_mcl + 1
                    # recupera o indíce do próximo do token do PLN
                    indice_proximo_token_wi_pln = pos_wi_pln + 1
                                        
                    # Concatena os subtokens até formar a palavra
                    while (indice_proximo_token_wj_mcl < len(lista_tokens_texto_mcl)) and (self.SEPARADOR_SUBTOKEN in lista_tokens_texto_mcl[indice_proximo_token_wj_mcl]):
                            
                        # Separa o subtoken
                        # Verifica se o subtoken possui o SEPARADOR_SUBTOKEN
                        if (self.SEPARADOR_SUBTOKEN != None) and (self.SEPARADOR_SUBTOKEN in lista_tokens_texto_mcl[indice_proximo_token_wj_mcl]):
                            # Remove os caracteres SEPARADOR_SUBTOKEN do subtoken
                            subtoken_palavra_mcl = lista_tokens_texto_mcl[indice_proximo_token_wj_mcl].replace(self.SEPARADOR_SUBTOKEN, "")                            
                        else:
                            # Recupera o subtoken
                            subtoken_palavra_mcl = lista_tokens_texto_mcl[indice_proximo_token_wj_mcl]
                  
                        # Concatena subtoken_palavra a palavra a ser completada
                        palavra_completa_wj_mcl = palavra_completa_wj_mcl + subtoken_palavra_mcl
                        #print("palavra_completa_wj:",palavra_completa_wj)
                                                
                        # Avança para o próximo token do MCL
                        indice_proximo_token_wj_mcl = indice_proximo_token_wj_mcl + 1
                    
                    # Monta a palavra do PLN se ela é diferente da palavra do MCL
                    if (palavra_completa_wj_mcl != wi_pln) or (palavra_completa_wj_mcl != self.TOKEN_DESCONHECIDO):
                        
                        # Concatena os subtokens até formar a palavra do PLN
                        while (indice_proximo_token_wi_pln < len(lista_tokens_texto_pln)) and (palavra_completa_wi_pln != palavra_completa_wj_mcl):
                            # Recupera o subtoken
                            subtoken_palavra_pln = lista_tokens_texto_pln[indice_proximo_token_wi_pln]                            
                            # Concatena subtoken_palavra a palavra a ser completada
                            palavra_completa_wi_pln = palavra_completa_wi_pln + subtoken_palavra_pln
                            # Avança para o próximo token do PLN
                            indice_proximo_token_wi_pln = indice_proximo_token_wi_pln + 1
                                                    
                    # print("\nMontei palavra_completa_wj_mcl:",palavra_completa_wj_mcl)
                    # print("Montei palavra_completa_wi_pln:",palavra_completa_wi_pln)
                    # print("Montei indice_proximo_token_wj_mcl:",indice_proximo_token_wj_mcl)
                    # print("Montei indice_proximo_token_wi_pln:",indice_proximo_token_wi_pln)
                    # print("Montei pos_wj_mcl:",pos_wj_mcl)                    
                    # print("Montei pos_wi_pln:",pos_wi_pln)
                    
                    # Verifica se a palavra é igual ao token do PLN ou se é desconhecida               
                    if (palavra_completa_wj_mcl == palavra_completa_wi_pln) or (palavra_completa_wj_mcl == self.TOKEN_DESCONHECIDO):
                        # Adiciona a palavra a lista
                        #print("Adiciona 4 palavra == wi or palavra_completa = TOKEN_DESCONHECIDO:",palavra_completa_wj_mcl, palavra_completa_wi_pln)
                        lista_tokens.append(palavra_completa_wj_mcl)
                                                
                        # Se a diferença do indice_proximo_token_wj_mcl e pos_wj_mcl for maior que 1 
                        # então a palavra é fora do vocabulário.
                        if indice_proximo_token_wj_mcl - pos_wj_mcl > 1:
                            # Marca como fora do vocabulário do MCL
                            lista_tokens_oov_mcl.append(PALAVRA_FORA_DO_VOCABULARIO)
                        else:
                            # Marca como dentro do vocabulário do MCL
                            lista_tokens_oov_mcl.append(PALAVRA_DENTRO_DO_VOCABULARIO)
                        
                        # Calcula a média dos tokens da palavra
                        #print("Calcula o máximo :", pos_wj_mcl , "até", indice_token)
                        embeddings_tokens_palavra = embeddings_texto[pos_wj_mcl:indice_proximo_token_wj_mcl]
                        #print("embeddings_tokens_palavra2:",embeddings_tokens_palavra)
                        #print("embeddings_tokens_palavra2:",embeddings_tokens_palavra.shape)
                        if isinstance(embeddings_tokens_palavra, torch.Tensor): 
                            # calcular a média dos embeddings dos tokens do MCL da palavra
                            embedding_estrategia_MEAN = torch.mean(embeddings_tokens_palavra, dim=0)
                            #print("embedding_estrategia_MEAN:",embedding_estrategia_MEAN)
                            #print("embedding_estrategia_MEAN.shape:",embedding_estrategia_MEAN.shape)      
                        else:
                            embedding_estrategia_MEAN = np.mean(embeddings_tokens_palavra, axis=0)      
                        # Adiciona a média dos embeddings dos tokens da palavra a lista de embeddings
                        lista_palavra_embeddings_MEAN.append(embedding_estrategia_MEAN)
                 
                        if isinstance(embeddings_tokens_palavra, torch.Tensor): 
                            # calcular o valor máximo dos embeddings dos tokens do MCL da palavra
                            embedding_estrategia_MAX, linha = torch.max(embeddings_tokens_palavra, dim=0)
                            #print("embedding_estrategia_MAX:",embedding_estrategia_MAX)
                            #print("embedding_estrategia_MAX.shape:",embedding_estrategia_MAX.shape)     
                        else:
                             embedding_estrategia_MAX = np.max(embeddings_tokens_palavra, axis=0)
                        # Adiciona o máximo dos embeddings dos tokens da palavra a lista de embeddings
                        lista_palavra_embeddings_MAX.append(embedding_estrategia_MAX)
                    else:
                        logger.error("A palavra tokenizada pelo PLN \"{}\" é diferente da palavra tokenizada pelo WordPiece do MCL \"{}\".".format(palavra_completa_wi_pln,palavra_completa_wj_mcl))
                        
                    # Avança para o próximo token da ferramenta de PLN
                    pos_wi_pln = indice_proximo_token_wi_pln
                    # Avança para o próximo token da ferramenta de MCL
                    pos_wj_mcl = indice_proximo_token_wj_mcl
        
        # Verificação se as listas estão com o mesmo tamanho        
        self._verificaSituacaoListaPalavras("getTokensPalavrasEmbeddingsTextoWordPiece.",
                                            tokens_texto_concatenado_pln,
                                            lista_tokens, 
                                            lista_tokens_texto_pln,
                                            lista_pos_texto_pln,
                                            lista_tokens_texto_mcl,
                                            lista_tokens_oov_mcl, 
                                            lista_palavra_embeddings_MEAN, 
                                            lista_palavra_embeddings_MAX)
       
        # Remove as variáveis que não serão mais utilizadas
        del embeddings_texto
        del lista_tokens_texto_mcl
        del lista_tokens_texto_pln

        # Retorna os tokens de palavras e os embeddings em um dicionário
        saida = {}
        saida.update({'tokens_texto' : lista_tokens,
                      'pos_texto_pln' : lista_pos_texto_pln,
                      'tokens_oov_texto_mcl' : lista_tokens_oov_mcl,
                      'palavra_embeddings_MEAN' : lista_palavra_embeddings_MEAN,
                      'palavra_embeddings_MAX' : lista_palavra_embeddings_MAX})

        return saida
    
    # ============================  
    # getTokensPalavrasEmbeddingsTextoSentencePiece(Albert)
    # Gera os tokens, POS e embeddings de cada texto.
    def getTokensPalavrasEmbeddingsTextoSentencePiece(self,
                                                      embeddings_texto, 
                                                      lista_tokens_texto_mcl: list[str],
                                                      tokens_texto_concatenado_pln: str,
                                                      pln: PLN,
                                                      dic_excecao: dict = {"": 0,}) -> dict:
        '''
        De um texto tokenizado pelo algoritmo SentencePiece e seus embeddings retorna os embeddings das palavras segundo a ferramenta de PLN.
        
        Condidera os tratamentos de tokenização do MCL na ordem a seguir.  
        1 - Exceção 
            Procura o token e o próximo no dicionário.wi_pln_pos_excecao	
            1.1 - Exceção maior que 0(valores positivos) - Tokenização do MCL gera mais tokens que a PLN.
                Ex.: {"St.":2}, a ferramenta de pln tokeniza "St." em 1 token "St." e o MCL em dois tokens"St" e "." ou seja 2 tokens do MCL devem virar 1.

            1.2 - Exceção menor que 0(valores negativos) - Tokenização do MCL gera menos tokens que a PLN.
                Ex.: {"1°": -1}, a ferramenta de pln tokeniza "1°" em 2 tokens "1" e "°" e o MCL em um token "1°", ou seja 2 tokens do PLN devem virar 1.

        2 - Token PLN igual ao token MCL ou desconhecida(UNK) e que não possui separador no próximo subtoken do MCL
            Token do PLN é igual ao token do MCL adiciona diretamente na lista de tokens.
            
        3 - Palavra foi dividida(tokenizada), o próximo token MCL possui separador e não é o fim da lista de tokens do MCL
            3.1 - Palavra completa MCL é igual a palavra completa PLN
                Tokens da palavra adicionado a lista de tokens e embeddings dos tokens consolidados para gerar as palavras.
            3.2 - Palavra completa MCL diferente da palavra completa PLN
                Especificar exceção maior ou menor para tratar tokens de palavra não tokenizado em palavra.
                          
        Parâmetros:
           `texto` - Um texto a ser recuperado os embeddings das palavras do modelo de linguagem.
           `embeddings_texto` - Os embeddings do texto gerados pelo método getEmbeddingsTexto.
           `lista_tokens_texto_mcl` - Os tokens do texto gerados pelo método getEmbeddingsTexto.
           `tokens_texto_concatenado_pln` - Os tokens do texto concatenado gerados pela ferramenta de PLN.
           `pln` - Uma instância da classe PLN para realizar a tokenização e POS-Tagging do texto.
           `dic_excecao` - Um dicionário de tokens de exceções e seus deslocamentos para considerar mais ou menos tokens do modelo de linguagem em relação ao spaCy. Valor positivo para considerar mais tokens e negativo para considerar menos tokens. Exemplo exceção negativo: {"1°": -1}, a ferramenta de PLN separa o token "1°" em "1" e "°", portanto é necessário reduzir 1 token pois o MCL gera somente um. Exemplo exceção positiva: {"1°": 1}, a ferramenta de PLN não separa o token "1°", mas o MCL separa em dois "1" e "°" portanto é necessário agrupar em 1 token.
               
        Retorna um dicionário com as seguintes chaves: 
           `tokens_texto` - Uma lista com os tokens do texto gerados pelo método.
           `pos_texto_pln` - Uma lista com as postagging dos tokens gerados pela ferramenta de PLN.
           `tokens_oov_texto_mcl` - Uma lista com os tokens OOV do MCL.
           `palavra_embeddings_MEAN` - Uma lista dos embeddings de palavras com a média dos embeddings(Estratégia MEAN) dos tokens que formam a palavra.
           `palavra_embeddings_MAX` - Uma lista dos embeddings de palavras com o máximo dos embeddings(Estratégia MAX) dos tokens que formam a palavra.
        '''

        # Inicializa os dicionários de exceção
        self._inicializaDicionarioExcecao(dic_excecao=dic_excecao)
              
        # Guarda os tokens e embeddings de retorno
        lista_tokens = []
        lista_tokens_oov_mcl = []
        lista_palavra_embeddings_MEAN = []
        lista_palavra_embeddings_MAX = []
        
        # Gera a tokenização e POS-Tagging da sentença    
        lista_tokens_texto_pln, lista_pos_texto_pln = pln.getListaTokensPOSTexto(texto=tokens_texto_concatenado_pln)

        # print("\tokens_texto_concatenado    :",tokens_texto_concatenado)    
        # print("lista_tokens_texto_pln       :",lista_tokens_texto_pln)
        # print("len(lista_tokens_texto_pln)  :",len(lista_tokens_texto_pln))    
        # print("lista_pos_texto_pln          :",lista_pos_texto_pln)
        # print("len(lista_pos_texto_pln)     :",len(lista_pos_texto_pln))
        
        # embedding <qtde_tokens x 4096>        
        # print("embeddings_texto             :",embeddings_texto.shape)
        # print("lista_tokens_texto_mcl       :",lista_tokens_texto_mcl)
        # print("len(lista_tokens_texto_mcl)  :",len(lista_tokens_texto_mcl))

        # Seleciona os pares de palavra a serem avaliadas
        pos_wi_pln = 0 # Posição do token da palavra gerado pelo spaCy
        pos_wj_mcl = pos_wi_pln # Posição do token da palavra gerado pelo MCL

        # Enquanto o indíce da palavra pos_wj(2a palavra) não chegou ao final da quantidade de tokens do MCL
        while (pos_wj_mcl < len(lista_tokens_texto_mcl)):  

            # Seleciona o token da sentença gerado pela ferramenta de PLN
            wi_pln = lista_tokens_texto_pln[pos_wi_pln] 
            # Recupera o token da palavra gerado pelo MCL
            wj_mcl = lista_tokens_texto_mcl[pos_wj_mcl] 
            # print("wi[",pos_wi_pln,"]=", wi_pln)
            # print("wj[",pos_wj_mcl,"]=", wj_mcl)
            
            # Procura o token e a concatenação dos seguintes no dicionário de exceções
            wi_pln_pos_excecao, pos_pln_excecao, pos_excecao = self._procuraTokenDicionario(wi_pln, 
                                                                                            lista_tokens_texto_pln, 
                                                                                            pos_wi_pln)
            
            # Se existe uma exceção
            if pos_pln_excecao != 0:
                #print("Exceção /pos_pln_excecao:", pos_pln_excecao, "/wi_pln_pos_excecao:", wi_pln_pos_excecao, "/pos_excecao:", pos_excecao)
               
                # É uma exceção maior
                if pos_pln_excecao > 0:
                    #print("Adiciona 1 Exceção maior wi_pln:", wi_pln, "/ wi_pln_pos_excecao:", wi_pln_pos_excecao, "/pos_excecao:", pos_excecao)
                    lista_tokens.append(wi_pln)
                    # Marca como fora do vocabulário do MCL
                    lista_tokens_oov_mcl.append(PALAVRA_FORA_DO_VOCABULARIO)
                    # Verifica se tem mais de um token para pegar o intervalo
                    if wi_pln_pos_excecao != 1:
                        # Localiza o indíce final do token do MCL
                        indice_final_token = pos_wj_mcl + pos_pln_excecao                        
                        #print("Calcula a média de :", pos_wj_mcl , "até", indice_final_token)
                        
                        # Recupera os embeddings dos tokens do MCL da posição pos_wj até indice_final_token
                        embeddings_tokens_palavra = embeddings_texto[pos_wj_mcl:indice_final_token]
                        #print("embeddings_tokens_palavra:",embeddings_tokens_palavra.shape)
                        
                        if isinstance(embeddings_tokens_palavra, torch.Tensor): 
                            # calcular a média dos embeddings dos tokens do MCL da palavra
                            embedding_estrategia_MEAN = torch.mean(embeddings_tokens_palavra, dim=0)
                        else:
                            embedding_estrategia_MEAN = np.mean(embeddings_tokens_palavra, axis=0)
                            
                        #print("embedding_estrategia_MEAN:",embedding_estrategia_MEAN.shape)
                        lista_palavra_embeddings_MEAN.append(embedding_estrategia_MEAN)
                        
                        if isinstance(embeddings_tokens_palavra, torch.Tensor): 
                            # calcular o máximo dos embeddings dos tokens do MCL da palavra
                            embedding_estrategia_MAX, linha = torch.max(embeddings_tokens_palavra, dim=0)
                        else:
                            embedding_estrategia_MAX, linha = np.max(embeddings_tokens_palavra, axis=0)
                        
                        #print("embedding_estrategia_MAX:",embedding_estrategia_MAX.shape)
                        lista_palavra_embeddings_MAX.append(embedding_estrategia_MAX)
                    else:
                        # Adiciona o embedding do token a lista de embeddings
                        lista_palavra_embeddings_MEAN.append(embeddings_texto[wi_pln_pos_excecao])            
                        lista_palavra_embeddings_MAX.append(embeddings_texto[wi_pln_pos_excecao])
             
                    # Avança para o próximo token do PLN e MCL
                    pos_wi_pln = pos_wi_pln + 1
                    pos_wj_mcl = pos_wj_mcl + pos_pln_excecao
                    #print("Proxima:")            
                    #print("wi[",pos_wi_pln,"]=", lista_tokens_texto_pln[pos_wi_pln])
                    #print("wj[",pos_wi_mcl,"]=", lista_tokens_texto_mcl[pos_wi_mcl])
                else:
                    # É uma exceção menor
                    if (pos_pln_excecao < 0):
                        #print("Adiciona 2 Exceção menor wi_pln:", wi_pln, "/ wi_pln_pos_excecao:", wi_pln_pos_excecao, "/pos_excecao:", pos_excecao)
                        lista_tokens.append(wi_pln_pos_excecao)
                        # Marca como dentro do vocabulário do MCL
                        lista_tokens_oov_mcl.append(PALAVRA_DENTRO_DO_VOCABULARIO)
                        # 1 é o mínimo a ser considerado
                        if ((pos_pln_excecao * -1) == 1): 
                            # Adiciona o embedding do token a lista de embeddings
                            lista_palavra_embeddings_MEAN.append(embeddings_texto[pos_wj_mcl])
                            lista_palavra_embeddings_MAX.append(embeddings_texto[pos_wj_mcl])
                        else:
                            logger.error("Erro: Exceção menor SentencePiece não pode ser maior que 1 token /wi_pln: \"{}\" wi_pln_pos_excecao: \"{}\" /pos_excecao: \"{}\".".format(wi_pln, wi_pln_pos_excecao, pos_excecao))
                            
                        # Avança para o próximo token do PLN e MCL nos deslocamentos de exceção
                        pos_wi_pln = pos_wi_pln + (pos_excecao - pos_wi_pln) # Avança os tokens do PLN considerando o deslocamento da exceção
                        pos_wj_mcl = pos_wj_mcl + (pos_pln_excecao * -1) # Avança os tokens do MCL considerando o deslocamento da exceção
                        #print("Proxima:")            
                        #print("wi[",pos_wi_pln,"]=", lista_tokens_texto_pln[pos_wi_pln])
                        #print("wj[",pos_wi_mcl,"]=", lista_tokens_texto_mcl[pos_wi_mcl])
            else:                  
                # Concatena o token com o separador de subtoken no inicio
                wi_pln_separador_token = self.SEPARADOR_SUBTOKEN + wi_pln                
                # Caso contrário concatena no fim
                if self.SEPARADOR_SUBTOKEN_POSICAO == 1:                    
                    wi_pln_separador_token = wi_pln + self.SEPARADOR_SUBTOKEN
                
                # Tokens iguais, com ou sem o separador, ou desconhecido adiciona diretamente na lista, pois o token não possui subtokens    
                if (wi_pln == wj_mcl) or (wi_pln_separador_token == wj_mcl) or (wj_mcl == self.TOKEN_DESCONHECIDO):
                    # Adiciona o token a lista de tokens
                    #print("Adiciona 3 (wi==wj or wj==TOKEN_DESCONHECIDO) e (próximo subtoken sem separador):", wi, wj)
                    lista_tokens.append(wi_pln)    
                    # Marca como dentro do vocabulário do MCL
                    lista_tokens_oov_mcl.append(PALAVRA_DENTRO_DO_VOCABULARIO)
                    # Adiciona o embedding do token a lista de embeddings
                    lista_palavra_embeddings_MEAN.append(embeddings_texto[pos_wj_mcl])
                    lista_palavra_embeddings_MAX.append(embeddings_texto[pos_wj_mcl])
                    #print("embedding1[pos_wj_mcl]:", embedding_texto[pos_wj_mcl].shape)
                    # Avança para o próximo token do PLN e MCL
                    pos_wi_pln = pos_wi_pln + 1
                    pos_wj_mcl = pos_wj_mcl + 1   
                  
                else:          
                    # A palavra foi tokenizada pelo SentencePice com SEPARADOR_SUBTOKEN (Ex.: ##) 
                    # ou diferente do spaCy ou desconhecida                   
                    # Inicializa a palavra a ser completada com os subtokens do PLN e MCL       
                    #print("wi[",pos_wi_pln,"]=", wi_pln) # PLN
                    #print("wj[",pos_wj_mcl,"]=", wj_mcl) # MCL
                    palavra_completa_wi_pln = wi_pln
                    
                    # Remove os caracteres SEPARADOR_SUBTOKEN do token
                    if (self.SEPARADOR_SUBTOKEN != None) and (self.SEPARADOR_SUBTOKEN in wj_mcl):
                        # Remove os caracteres SEPARADOR_SUBTOKEN do token
                        palavra_completa_wj_mcl = wj_mcl.replace(self.SEPARADOR_SUBTOKEN, "")
                    else:                
                        palavra_completa_wj_mcl = wj_mcl
                    
                    # recupera o indíce do próximo do token do MCL
                    indice_proximo_token_wj_mcl = pos_wj_mcl + 1
                     # recupera o indíce do próximo do token do PLN
                    indice_proximo_token_wi_pln = pos_wi_pln + 1
                           
                    # Concatena os subtokens até formar a palavra
                    while (self.SEPARADOR_SUBTOKEN != None) and (palavra_completa_wj_mcl != palavra_completa_wi_pln) and (indice_proximo_token_wj_mcl < len(lista_tokens_texto_mcl)) and (self.SEPARADOR_SUBTOKEN not in lista_tokens_texto_mcl[indice_proximo_token_wj_mcl]):
                       
                        # Separa o subtoken
                        if (self.SEPARADOR_SUBTOKEN != None) and (self.SEPARADOR_SUBTOKEN in lista_tokens_texto_mcl[indice_proximo_token_wj_mcl]):
                            # Remove os caracteres SEPARADOR_SUBTOKEN do subtoken
                            subtoken_palavra = lista_tokens_texto_mcl[indice_proximo_token_wj_mcl].replace(self.SEPARADOR_SUBTOKEN, "")
                        else:
                            # Recupera o subtoken
                            subtoken_palavra = lista_tokens_texto_mcl[indice_proximo_token_wj_mcl]
                  
                        # Concatena a parte do subtoken a palavra
                        palavra_completa_wj_mcl = palavra_completa_wj_mcl + subtoken_palavra
                        #print("palavra_completa:",palavra_completa)
                        # Avança para o próximo subtoken do MCL
                        indice_proximo_token_wj_mcl = indice_proximo_token_wj_mcl + 1
                        
                    # Monta a palavra do PLN se ela é diferente da palavra do MCL
                    if (palavra_completa_wj_mcl != palavra_completa_wi_pln) or (palavra_completa_wj_mcl != self.TOKEN_DESCONHECIDO):
                        
                        # Concatena os subtokens até formar a palavra do PLN
                        while (indice_proximo_token_wi_pln < len(lista_tokens_texto_pln)) and (palavra_completa_wi_pln != palavra_completa_wj_mcl):
                            # Recupera o subtoken
                            subtoken_palavra_pln = lista_tokens_texto_pln[indice_proximo_token_wi_pln]                            
                            # Concatena subtoken_palavra a palavra a ser completada
                            palavra_completa_wi_pln = palavra_completa_wi_pln + subtoken_palavra_pln
                            # Avança para o próximo token do PLN
                            indice_proximo_token_wi_pln = indice_proximo_token_wi_pln + 1    

                    # print("\nMontei palavra_completa_wj_mcl:",palavra_completa_wj_mcl)
                    # print("Montei palavra_completa_wi_pln:",palavra_completa_wi_pln)
                    # print("Montei indice_proximo_token_wj_mcl:",indice_proximo_token_wj_mcl)
                    # print("Montei indice_proximo_token_wi_pln:",indice_proximo_token_wi_pln)
                    # print("Montei pos_wj_mcl:",pos_wj_mcl)                    
                    # print("Montei pos_wi_pln:",pos_wi_pln)
                    
                    # Verifica se a palavra é igual ao token do PLN ou se é desconhecida
                    if (palavra_completa_wj_mcl == palavra_completa_wi_pln) or (palavra_completa_wj_mcl == self.TOKEN_DESCONHECIDO):                    
                        # Adiciona a palavra a lista
                        #print("Adiciona 4 palavra == wi or palavra_completa = TOKEN_DESCONHECIDO:",palavra_completa_wj_mcl, palavra_completa_wi_pln)
                        #lista_tokens.append(wi_pln)
                        lista_tokens.append(palavra_completa_wj_mcl)

                        # Se a diferença do indice_proximo_token_wj_mcl e pos_wj_mcl for maior que 1 
                        # então a palavra é fora do vocabulário.                        
                        if indice_proximo_token_wj_mcl - pos_wj_mcl > 1:
                            # Marca como fora do vocabulário do MCL
                            lista_tokens_oov_mcl.append(PALAVRA_FORA_DO_VOCABULARIO)
                        else:
                            # Marca como dentro do vocabulário do MCL
                            lista_tokens_oov_mcl.append(PALAVRA_DENTRO_DO_VOCABULARIO)
                        
                        # Calcula a média dos tokens da palavra
                        #print("Calcula o máximo :", pos_wj_mcl , "até", indice_proximo_token_wj_mcl)
                        embeddings_tokens_palavra = embeddings_texto[pos_wj_mcl:indice_proximo_token_wj_mcl]
                        #print("embeddings_tokens_palavra2:",embeddings_tokens_palavra)
                        #print("embeddings_tokens_palavra2:",embeddings_tokens_palavra.shape)
                        
                        if isinstance(embeddings_tokens_palavra, torch.Tensor): 
                            # calcular a média dos embeddings dos tokens do MCL da palavra
                            embedding_estrategia_MEAN = torch.mean(embeddings_tokens_palavra, dim=0)
                            #print("embedding_estrategia_MEAN:",embedding_estrategia_MEAN)
                            #print("embedding_estrategia_MEAN.shape:",embedding_estrategia_MEAN.shape)      
                        else:
                            embedding_estrategia_MEAN = np.mean(embeddings_tokens_palavra, axis=0)
                        # Adiciona a média dos embeddings dos tokens da palavra a lista de embeddings      
                        lista_palavra_embeddings_MEAN.append(embedding_estrategia_MEAN)
                 
                        if isinstance(embeddings_tokens_palavra, torch.Tensor): 
                            # calcular o valor máximo dos embeddings dos tokens do MCL da palavra
                            embedding_estrategia_MAX, linha = torch.max(embeddings_tokens_palavra, dim=0)
                            #print("embedding_estrategia_MAX:",embedding_estrategia_MAX)
                            #print("embedding_estrategia_MAX.shape:",embedding_estrategia_MAX.shape)     
                        else:
                             embedding_estrategia_MAX = np.max(embeddings_tokens_palavra, axis=0)
                        # Adiciona o máximo dos embeddings dos tokens da palavra a lista de embeddings
                        lista_palavra_embeddings_MAX.append(embedding_estrategia_MAX)
                    else:                        
                        logger.error("A palavra tokenizada pelo PLN \"{}\" é diferente da palavra tokenizada pelo SentencePice do MCL \"{}\".".format(palavra_completa_wi_pln,palavra_completa_wj_mcl))
                    
                    # Avança para o próximo token da ferramenta de PLN
                    pos_wi_pln = indice_proximo_token_wi_pln
                    # Avança para o próximo token da ferramenta de MCL
                    pos_wj_mcl = indice_proximo_token_wj_mcl
        
        # Verificação se as listas estão com o mesmo tamanho        
        self._verificaSituacaoListaPalavras("getTokensPalavrasEmbeddingsTextoSentencePiece.",
                                            tokens_texto_concatenado_pln,
                                            lista_tokens, 
                                            lista_tokens_texto_pln,
                                            lista_pos_texto_pln,
                                            lista_tokens_texto_mcl,
                                            lista_tokens_oov_mcl, 
                                            lista_palavra_embeddings_MEAN, 
                                            lista_palavra_embeddings_MAX)            
       
        # Remove as variáveis que não serão mais utilizadas
        del embeddings_texto
        del lista_tokens_texto_mcl
        del lista_tokens_texto_pln

        # Retorna os tokens de palavras e os embeddings em um dicionário
        saida = {}
        saida.update({'tokens_texto' : lista_tokens,
                      'pos_texto_pln' : lista_pos_texto_pln,
                      'tokens_oov_texto_mcl' : lista_tokens_oov_mcl,
                      'palavra_embeddings_MEAN' : lista_palavra_embeddings_MEAN,
                      'palavra_embeddings_MAX' : lista_palavra_embeddings_MAX})

        return saida
   
    # ============================  
    # getTokensPalavrasEmbeddingsTextoBPE
    # Gera os tokens, POS e embeddings de cada texto.
    def getTokensPalavrasEmbeddingsTextoBPE(self,
                                            embeddings_texto, 
                                            lista_tokens_texto_mcl: list[str],
                                            tokens_texto_concatenado_pln: str,
                                            pln: PLN,
                                            dic_excecao: dict = {"":0,}) -> dict:
        '''
        De um texto tokenizado pelo algoritmo BPE e seus embeddings retorna os embeddings das palavras segundo a ferramenta de PLN.
        
        Condidera os tratamentos de tokenização do MCL na ordem a seguir.  
        1 - Exceção 
            Procura o token e o próximo no dicionário.wi_pln_pos_excecao	
            1.1 - Exceção maior que 0(valores positivos) - Tokenização do MCL gera mais tokens que a PLN.
                Ex.: {"St.":2}, a ferramenta de pln tokeniza "St." em 1 token "St." e o MCL em dois tokens"St" e "." ou seja 2 tokens do MCL devem virar 1.

            1.2 - Exceção menor que 0(valores negativos) - Tokenização do MCL gera menos tokens que a PLN.
                Ex.: {"1°": -1}, a ferramenta de pln tokeniza "1°" em 2 tokens "1" e "°" e o MCL em um token "1°", ou seja 2 tokens do PLN devem virar 1.

        2 - Token PLN igual ao token MCL ou desconhecida(UNK) e que não possui separador no próximo subtoken do MCL
            Token do PLN é igual ao token do MCL adiciona diretamente na lista de tokens.
            
        3 - Palavra foi dividida(tokenizada), o próximo token MCL possui separador e não é o fim da lista de tokens do MCL
            3.1 - Palavra completa MCL é igual a palavra completa PLN
                Tokens da palavra adicionado a lista de tokens e embeddings dos tokens consolidados para gerar as palavras.
            3.2 - Palavra completa MCL diferente da palavra completa PLN
                Especificar exceção maior ou menor para tratar tokens de palavra não tokenizado em palavra.
                          
        Parâmetros:
           `texto` - Um texto a ser recuperado os embeddings das palavras do modelo de linguagem.
           `embeddings_texto` - Os embeddings do texto gerados pelo método getEmbeddingsTexto.
           `lista_tokens_texto_mcl` - Os tokens do texto gerados pelo método getEmbeddingsTexto.
           `tokens_texto_concatenado_pln` - Os tokens do texto concatenado gerados pela ferramenta de PLN.
           `pln` - Uma instância da classe PLN para realizar a tokenização e POS-Tagging do texto.
           `dic_excecao` - Um dicionário de tokens de exceções e seus deslocamentos para considerar mais ou menos tokens do modelo de linguagem em relação ao spaCy. Valor positivo para considerar mais tokens e negativo para considerar menos tokens. Exemplo exceção negativo: {"1°": -1}, a ferramenta de PLN separa o token "1°" em "1" e "°", portanto é necessário reduzir 1 token pois o MCL gera somente um. Exemplo exceção positiva: {"1°": 1}, a ferramenta de PLN não separa o token "1°", mas o MCL separa em dois "1" e "°" portanto é necessário agrupar em 1 token.
               
        Retorna um dicionário com as seguintes chaves: 
           `tokens_texto` - Uma lista com os tokens do texto gerados pelo método.
           `pos_texto_pln` - Uma lista com as postagging dos tokens gerados pela ferramenta de PLN.
           `tokens_oov_texto_mcl` - Uma lista com os tokens OOV do MCL.
           `palavra_embeddings_MEAN` - Uma lista dos embeddings de palavras com a média dos embeddings(Estratégia MEAN) dos tokens que formam a palavra.
           `palavra_embeddings_MAX` - Uma lista dos embeddings de palavras com o máximo dos embeddings(Estratégia MAX) dos tokens que formam a palavra.
        '''
 
        # Inicializa os dicionários de exceção
        self._inicializaDicionarioExcecao(dic_excecao=dic_excecao)
       
        # Guarda os tokens e embeddings de retorno
        lista_tokens = []
        lista_tokens_oov_mcl = []
        lista_palavra_embeddings_MEAN = []
        lista_palavra_embeddings_MAX = []
        
        # Gera a tokenização e POS-Tagging da sentença    
        lista_tokens_texto_pln, lista_pos_texto_pln = pln.getListaTokensPOSTexto(texto=tokens_texto_concatenado_pln)

        # print("\tokens_texto_concatenado    :",tokens_texto_concatenado)    
        # print("lista_tokens_texto_pln       :",lista_tokens_texto_pln)
        # print("len(lista_tokens_texto_pln)  :",len(lista_tokens_texto_pln))    
        # print("lista_pos_texto_pln          :",lista_pos_texto_pln)
        # print("len(lista_pos_texto_pln)     :",len(lista_pos_texto_pln))
        
        # embedding <qtde_tokens x 4096>        
        # print("embeddings_texto             :",embeddings_texto.shape)
        # print("lista_tokens_texto_mcl       :",lista_tokens_texto_mcl)
        # print("len(lista_tokens_texto_mcl)  :",len(lista_tokens_texto_mcl))

        # Seleciona os pares de palavra a serem avaliadas
        pos_wi_pln = 0 # Posição do token da palavra gerado pelo spaCy
        pos_wj_mcl = pos_wi_pln # Posição do token da palavra gerado pelo MCL

        # Enquanto o indíce da palavra pos_wj(2a palavra) não chegou ao final da quantidade de tokens do MCL
        while (pos_wj_mcl < len(lista_tokens_texto_mcl)):  

            # Seleciona o token da sentença gerado pela ferramenta de PLN
            wi_pln = lista_tokens_texto_pln[pos_wi_pln] 
            # Recupera o token da palavra gerado pelo MCL
            wj_mcl = lista_tokens_texto_mcl[pos_wj_mcl] 
            # print("wi[",pos_wi_pln,"]=", wi_pln)
            # print("wj[",pos_wj_mcl,"]=", wj_mcl)
            
            # Procura o token e a concatenação dos seguintes no dicionário de exceções
            wi_pln_pos_excecao, pos_pln_excecao, pos_excecao = self._procuraTokenDicionario(wi_pln, 
                                                                                            lista_tokens_texto_pln, 
                                                                                            pos_wi_pln)
            
            # S existe uma exceção
            if pos_pln_excecao != 0:
                #print("Exceção /pos_pln_excecao:", pos_pln_excecao, "/wi_pln_pos_excecao:", wi_pln_pos_excecao, "/pos_excecao:", pos_excecao)
                
                # É uma exceção maior
                if pos_pln_excecao > 0:
                    #print("Adiciona 1 Exceção maior wi_pln:", wi_pln, "/ wi_pln_pos_excecao:", wi_pln_pos_excecao, "/pos_excecao:", pos_excecao)
                    lista_tokens.append(wi_pln)
                    # Marca como fora do vocabulário do MCL
                    lista_tokens_oov_mcl.append(PALAVRA_FORA_DO_VOCABULARIO)
                    # Verifica se tem mais de um token para pegar o intervalo
                    if wi_pln_pos_excecao != 1:
                        # Localiza o indíce final do token do MCL
                        indice_final_token = pos_wj_mcl + pos_pln_excecao                        
                        #print("Calcula a média de :", pos_wj_mcl , "até", indice_final_token)
                        
                        # Recupera os embeddings dos tokens do MCL da posição pos_wj até indice_final_token
                        embeddings_tokens_palavra = embeddings_texto[pos_wj_mcl:indice_final_token]
                        #print("embeddings_tokens_palavra:",embeddings_tokens_palavra.shape)
                        
                        if isinstance(embeddings_tokens_palavra, torch.Tensor): 
                            # calcular a média dos embeddings dos tokens do MCL da palavra
                            embedding_estrategia_MEAN = torch.mean(embeddings_tokens_palavra, dim=0)
                        else:
                            embedding_estrategia_MEAN = np.mean(embeddings_tokens_palavra, axis=0)
                            
                        #print("embedding_estrategia_MEAN:",embedding_estrategia_MEAN.shape)
                        lista_palavra_embeddings_MEAN.append(embedding_estrategia_MEAN)
                        
                        if isinstance(embeddings_tokens_palavra, torch.Tensor): 
                            # calcular o máximo dos embeddings dos tokens do MCL da palavra
                            embedding_estrategia_MAX, linha = torch.max(embeddings_tokens_palavra, dim=0)
                        else:
                            embedding_estrategia_MAX, linha = np.max(embeddings_tokens_palavra, axis=0)
                        
                        #print("embedding_estrategia_MAX:",embedding_estrategia_MAX.shape)
                        lista_palavra_embeddings_MAX.append(embedding_estrategia_MAX)
                    else:
                        # Adiciona o embedding do token a lista de embeddings
                        lista_palavra_embeddings_MEAN.append(embeddings_texto[wi_pln_pos_excecao])            
                        lista_palavra_embeddings_MAX.append(embeddings_texto[wi_pln_pos_excecao])
             
                    # Avança para o próximo token do PLN e MCL
                    pos_wi_pln = pos_wi_pln + 1
                    pos_wj_mcl = pos_wj_mcl + pos_pln_excecao
                    #print("Proxima:")            
                    #print("wi[",pos_wi_pln,"]=", lista_tokens_texto_pln[pos_wi_pln])
                    #print("wj[",pos_wi_mcl,"]=", lista_tokens_texto_mcl[pos_wi_mcl])
                else:
                    # É uma exceção menor
                    if (pos_pln_excecao < 0):
                        #print("Adiciona 2 Exceção menor wi_pln:", wi_pln, "/ wi_pln_pos_excecao:", wi_pln_pos_excecao, "/pos_excecao:", pos_excecao)
                        lista_tokens.append(wi_pln_pos_excecao)
                        # Marca como dentro do vocabulário do MCL
                        lista_tokens_oov_mcl.append(PALAVRA_DENTRO_DO_VOCABULARIO)
                        # 1 é o mínimo a ser considerado
                        if ((pos_pln_excecao * -1) == 1): 
                            # Adiciona o embedding do token a lista de embeddings
                            lista_palavra_embeddings_MEAN.append(embeddings_texto[pos_wj_mcl])
                            lista_palavra_embeddings_MAX.append(embeddings_texto[pos_wj_mcl])
                        else:
                            logger.error("Erro: Exceção menor BPE não pode ser maior que 1 token /wi_pln: \"{}\" wi_pln_pos_excecao: \"{}\" /pos_excecao: \"{}\".".format(wi_pln, wi_pln_pos_excecao, pos_excecao))
                            
                        # Avança para o próximo token do PLN e MCL nos deslocamentos de exceção
                        pos_wi_pln = pos_wi_pln + (pos_excecao - pos_wi_pln) # Avança os tokens do PLN considerando o deslocamento da exceção
                        pos_wj_mcl = pos_wj_mcl + (pos_pln_excecao * -1) # Avança os tokens do MCL considerando o deslocamento da exceção
                        #print("Proxima:")            
                        #print("wi[",pos_wi_pln,"]=", lista_tokens_texto_pln[pos_wi_pln])
                        #print("wj[",pos_wi_mcl,"]=", lista_tokens_texto_mcl[pos_wi_mcl])
            else:  
                # Tokens iguais ou desconhecido adiciona diretamente na lista, pois o token não possui subtokens
                
                # Concatena o token com o separador de subtoken no inicio
                wi_separador_token = self.SEPARADOR_SUBTOKEN + wi_pln                
                # Caso contrário concatena no fim
                if self.SEPARADOR_SUBTOKEN_POSICAO == 1:                    
                    wi_separador_token = wi_pln + self.SEPARADOR_SUBTOKEN
                    
                if (wi_pln == wj_mcl) or (wi_separador_token == wj_mcl) or (wj_mcl[0] == self.TOKEN_DESCONHECIDO):
                    # Adiciona o token a lista de tokens
                    # print("Adiciona 2 (wi_pln==wj_mcl) or (wi_separador_token == wj_mcl) or (wj_mcl==TOKEN_DESCONHECIDO[0]):", wi_pln, wj_mcl )
                    lista_tokens.append(wi_pln)    
                    # Marca como dentro do vocabulário do MCL
                    lista_tokens_oov_mcl.append(PALAVRA_DENTRO_DO_VOCABULARIO)
                    # Adiciona o embedding do token a lista de embeddings
                    lista_palavra_embeddings_MEAN.append(embeddings_texto[pos_wj_mcl])
                    lista_palavra_embeddings_MAX.append(embeddings_texto[pos_wj_mcl])
                    #print("embedding1[pos_wj_mcl]:", embedding_texto[pos_wj_mcl].shape)
                    # Avança para a próxima palavra e token do MCL
                    pos_wi_pln = pos_wi_pln + 1
                    pos_wj_mcl = pos_wj_mcl + 1   
                  
                else:          
                    # A palavra foi tokenizada pelo BPE com SEPARADOR_SUBTOKEN (Ex.: ##) ou diferente do spaCy ou desconhecida
                    # Inicializa a palavra a ser completada com os subtokens do PLN e MCL
                    palavra_completa_wi_pln = wi_pln
                    # Remove os caracteres SEPARADOR_SUBTOKEN do token
                    if (self.SEPARADOR_SUBTOKEN != None) and (self.SEPARADOR_SUBTOKEN in wj_mcl):
                        # Remove os caracteres SEPARADOR_SUBTOKEN do token
                        palavra_completa_wj_mcl = wj_mcl.replace(self.SEPARADOR_SUBTOKEN, "")
                    else:                
                        palavra_completa_wj_mcl = wj_mcl
                    
                    # recupera o indíce do próximo do token do MCL
                    indice_proximo_token_wj_mcl = pos_wj_mcl + 1
                    # recupera o indíce do próximo do token do PLN
                    indice_proximo_token_wi_pln = pos_wi_pln + 1
                    
                    # Concatena os subtokens até formar a palavra    
                    while (self.SEPARADOR_SUBTOKEN != None) and (palavra_completa_wj_mcl != wi_pln) and (indice_proximo_token_wj_mcl < len(lista_tokens_texto_mcl)) and (self.SEPARADOR_SUBTOKEN not in lista_tokens_texto_mcl[indice_proximo_token_wj_mcl]):
                       
                        # Separa o subtoken
                        if (self.SEPARADOR_SUBTOKEN != None) and (self.SEPARADOR_SUBTOKEN in lista_tokens_texto_mcl[indice_proximo_token_wj_mcl]):
                            # Remove os caracteres SEPARADOR_SUBTOKEN do token
                            subtoken_palavra = lista_tokens_texto_mcl[indice_proximo_token_wj_mcl].replace(self.SEPARADOR_SUBTOKEN, "")
                        else:
                            # Recupera o subtoken
                            subtoken_palavra = lista_tokens_texto_mcl[indice_proximo_token_wj_mcl]
                  
                        # Concatena subtoken_palavra a palavra a ser completada
                        palavra_completa_wj_mcl = palavra_completa_wj_mcl + subtoken_palavra
                        #print("palavra_POS:",palavra_POS)
                        # Avança para o próximo token do MCL
                        indice_proximo_token_wj_mcl = indice_proximo_token_wj_mcl + 1

                    # Monta a palavra do PLN se ela é diferente da palavra do MCL
                    if (palavra_completa_wj_mcl != wi_pln) or (palavra_completa_wj_mcl != self.TOKEN_DESCONHECIDO):
                        
                        # Concatena os subtokens até formar a palavra do PLN
                        while (indice_proximo_token_wi_pln < len(lista_tokens_texto_pln)) and (palavra_completa_wi_pln != palavra_completa_wj_mcl):
                            # Recupera o subtoken
                            subtoken_palavra_pln = lista_tokens_texto_pln[indice_proximo_token_wi_pln]                            
                            # Concatena subtoken_palavra a palavra a ser completada
                            palavra_completa_wi_pln = palavra_completa_wi_pln + subtoken_palavra_pln
                            # Avança para o próximo token do PLN
                            indice_proximo_token_wi_pln = indice_proximo_token_wi_pln + 1
                    
                    # print("\nMontei palavra_completa_wj_mcl:",palavra_completa_wj_mcl)
                    # print("Montei palavra_completa_wi_pln:",palavra_completa_wi_pln)
                    # print("Montei indice_proximo_token_wj_mcl:",indice_proximo_token_wj_mcl)
                    # print("Montei indice_proximo_token_wi_pln:",indice_proximo_token_wi_pln)
                    # print("Montei pos_wj_mcl:",pos_wj_mcl)                    
                    # print("Montei pos_wi_pln:",pos_wi_pln)
                    
                    # Verifica se a palavra é igual ao token do PLN ou se é desconhecida
                    if (palavra_completa_wj_mcl == palavra_completa_wi_pln) or (palavra_completa_wj_mcl == self.TOKEN_DESCONHECIDO):
                        # Adiciona o token a lista
                        #print("Adiciona 4 palavra == wi or palavra_completa = TOKEN_DESCONHECIDO:",palavra_completa_wj_mcl, palavra_completa_wi_pln)
                        lista_tokens.append(wi_pln)
                        
                        # Se a diferença do indice_proximo_token_wj_mcl e pos_wj_mcl for maior que 1 
                        # então a palavra é fora do vocabulário.                        
                        if indice_proximo_token_wj_mcl - pos_wj_mcl > 1:
                            # Marca como fora do vocabulário do MCL
                            lista_tokens_oov_mcl.append(PALAVRA_FORA_DO_VOCABULARIO)
                        else:
                            # Marca como dentro do vocabulário do MCL
                            lista_tokens_oov_mcl.append(PALAVRA_DENTRO_DO_VOCABULARIO)
                        
                        # Calcula a média dos tokens da palavra
                        #print("Calcula o máximo :", pos_wj_mcl , "até", indice_proximo_token_wj_mcl)
                        embeddings_tokens_palavra = embeddings_texto[pos_wj_mcl:indice_proximo_token_wj_mcl]
                        #print("embeddings_tokens_palavra2:",embeddings_tokens_palavra)
                        #print("embeddings_tokens_palavra2:",embeddings_tokens_palavra.shape)
                        
                        if isinstance(embeddings_tokens_palavra, torch.Tensor): 
                            # calcular a média dos embeddings dos tokens do MCL da palavra
                            embedding_estrategia_MEAN = torch.mean(embeddings_tokens_palavra, dim=0)
                            #print("embedding_estrategia_MEAN:",embedding_estrategia_MEAN)
                            #print("embedding_estrategia_MEAN.shape:",embedding_estrategia_MEAN.shape)      
                        else:
                            embedding_estrategia_MEAN = np.mean(embeddings_tokens_palavra, axis=0)      
                        # Adiciona a média dos embeddings dos tokens da palavra a lista de embeddings     
                        lista_palavra_embeddings_MEAN.append(embedding_estrategia_MEAN)
                 
                        if isinstance(embeddings_tokens_palavra, torch.Tensor): 
                            # calcular o valor máximo dos embeddings dos tokens do MCL da palavra
                            embedding_estrategia_MAX, linha = torch.max(embeddings_tokens_palavra, dim=0)
                            #print("embedding_estrategia_MAX:",embedding_estrategia_MAX)
                            #print("embedding_estrategia_MAX.shape:",embedding_estrategia_MAX.shape)     
                        else:
                             embedding_estrategia_MAX = np.max(embeddings_tokens_palavra, axis=0)
                        # Adiciona o máximo dos embeddings dos tokens da palavra a lista de embeddings   
                        lista_palavra_embeddings_MAX.append(embedding_estrategia_MAX)
                    else:
                        logger.error("A palavra tokenizada pelo PLN \"{}\" é diferente da palavra tokenizada pelo BPE do MCL \"{}\".".format(palavra_completa_wi_pln,palavra_completa_wj_mcl))

                    # Avança para o próximo token da ferramenta de PLN
                    pos_wi_pln = indice_proximo_token_wi_pln
                    # Avança para o próximo token da ferramenta de MCL
                    pos_wj_mcl = indice_proximo_token_wj_mcl
        
        # Verificação se as listas estão com o mesmo tamanho        
        self._verificaSituacaoListaPalavras("getTokensPalavrasEmbeddingsTextoBPE.",
                                            tokens_texto_concatenado_pln,
                                            lista_tokens, 
                                            lista_tokens_texto_pln,
                                            lista_pos_texto_pln,
                                            lista_tokens_texto_mcl,
                                            lista_tokens_oov_mcl, 
                                            lista_palavra_embeddings_MEAN, 
                                            lista_palavra_embeddings_MAX)   
       
        # Remove as variáveis que não serão mais utilizadas
        del embeddings_texto
        del lista_tokens_texto_mcl
        del lista_tokens_texto_pln

        # Retorna os tokens de palavras e os embeddings em um dicionário
        saida = {}
        saida.update({'tokens_texto' : lista_tokens,
                      'pos_texto_pln' : lista_pos_texto_pln,
                      'tokens_oov_texto_mcl' : lista_tokens_oov_mcl,
                      'palavra_embeddings_MEAN' : lista_palavra_embeddings_MEAN,
                      'palavra_embeddings_MAX' : lista_palavra_embeddings_MAX})

        return saida
    
    # ============================   
    def getDimensaoEmbedding(self) -> int:
        '''
        Retorna a dimensão do embedding
        '''

        return self.auto_model.config.hidden_size        
        
    # ============================   
    def save(self, output_path: str):
        '''
        Salva o modelo.

        Parâmetros:
           `output_path` - caminho para salvar o modelo
        '''

        self.auto_model.save_pretrained(output_path)
        self.auto_tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'modelo_linguagem_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    # ============================   
    def getAutoModel(self) -> AutoModel:
        '''
        Recupera o modelo.
        '''

        return self.auto_model

    # ============================
    def getTokenizer(self) -> AutoTokenizer:
        '''
        Recupera o tokenizador.
        '''

        return self.auto_tokenizer
    
    # ============================   
    def batchToDevice(self, lote, 
                      target_device: device):
        '''
        Envia lote pytorch batch para um dispositivo (CPU/GPU)

        Parâmetros:
           `lote` - lote pytorch
           `target_device` - dispositivo de destino (CPU/GPU)
        
        Retorno:
           lote enviado para o dispositivo        
        '''

        for key in lote:
            if isinstance(lote[key], Tensor):
                lote[key] = lote[key].to(target_device)
                
        return lote
    
    # ============================   
    def trataListaTokensEspeciais(self, tokens_texto_mcl: List[str]) -> List[str]:    
        '''
        Trata a lista de tokens tokenizador do MCL.

        Parâmetros:
           `tokens_texto_mcl` - Lista dos tokens gerados pelo tokenizador.
           
        Retorno:
           Lista de tokens tratada.        
        '''  
        
        # Se o primeiro token não for o TOKEN_INICIO e o token não tem caracter inicial igual ao separador, adiciona
        if (self.TOKEN_INICIO != tokens_texto_mcl[0]) and (self.SEPARADOR_SUBTOKEN != tokens_texto_mcl[0][0]):
        
            tokens_texto_mcl = [self.SEPARADOR_SUBTOKEN + tokens_texto_mcl[0]] + tokens_texto_mcl[1:]
            #print("tokens_texto_mcl:", tokens_texto_mcl)
        
        return tokens_texto_mcl
