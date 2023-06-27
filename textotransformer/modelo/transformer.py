# Import das bibliotecas.

# Biblioteca de logging
import logging  
# Biblioteca de aprendizado de máquina
from torch import nn 
import torch 
import numpy as np
from torch import Tensor, device
# Biblioteca do transformer
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config, MT5Config
# Biblioteca de manipulação json
import json
# Biblioteca de tipos
from typing import List, Dict, Optional, Union
# Biblioteca de manipulação sistema
import os

# Bibliotecas próprias
from textotransformer.modelo.modeloarguments import ModeloArgumentos
from textotransformer.modelo.modeloenum import AbordagemExtracaoEmbeddingsCamadas
from textotransformer.util.utilconstantes import LISTATIPOCAMADA_NOME

logger = logging.getLogger(__name__)

class Transformer(nn.Module):
    '''
    Classe que encapsula a classe AutoModel da Huggingface para gerar embeddings de token, palavra, sentença ou texto.
    Carrega a classe correta, por exemplo BERT / RoBERTa etc.

    Parâmetros:
       `modelo_args' - Argumentos passados para o modelo Huggingface Transformers.
       `cache_dir` - Cache dir para Huggingface Transformers para armazenar/carregar modelos.
       `tokenizer_args` - Argumentos (chave, pares de valor) passados para o modelo Huggingface Tokenizer
       `tokenizer_name_or_path` - Nome ou caminho do tokenizer. Quando None, model_name_or_path é usado    
    '''

    def __init__(self, 
                modelo_args : ModeloArgumentos,                
                cache_dir: Optional[str] = None,
                tokenizer_args: Dict = {}, 
                tokenizer_name_or_path : str = None):
        
        # Inicializa o construtor da superclasse
        super(Transformer, self).__init__()
        
        # Define os argumentos do modelo
        self.modelo_args = modelo_args

        # Recupera o nome do modelo dos argumentos
        model_name_or_path = modelo_args.pretrained_model_name_or_path;
      
        # Recupera parâmetros do transformador dos argumentos e cria um dicionário para o AutoConfig
        model_args = {"output_attentions": modelo_args.output_attentions, 
                      "output_hidden_states": modelo_args.output_hidden_states}
    
        # Configuração do modelo        
        self.config = AutoConfig.from_pretrained(model_name_or_path, 
                                            **model_args, 
                                            cache_dir=cache_dir)
        
        # Carrega o modelo
        self._load_model(model_name_or_path, 
                         self.config, 
                         cache_dir)

        # Carrega o tokenizador
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else  model_name_or_path, cache_dir=cache_dir, **tokenizer_args)

        # Se max_seq_length não foi especificado, tenta inferir do modelo
        if self.modelo_args.max_seq_len is None:
            if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                self.modelo_args.max_seq_len = min(self.auto_model.config.max_position_embeddings,
                                     self.tokenizer.model_max_length)

        # Define a classe do tokenizador
        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__
            
        logger.info("Classe \"{}\" carregada: \"{}\".".format(self.__class__.__name__, modelo_args))

    # ============================   
    def __repr__(self):
        '''
        Retorna uma string com descrição do objeto.
        '''

        return "Classe (\"{}\") carregada com o modelo \"{}\", m AutoConfig \"{}\", Transformer \"{}\" e tokenizador: \"{}\".".format(self.__class__.__name__,
                                                                                                                                      self.modelo_args.pretrained_model_name_or_path,
                                                                                                                                      self.config.__class__.__name__,
                                                                                                                                      self.auto_model.__class__.__name__,
                                                                                                                                      self.tokenizer.__class__.__name__)

    # ============================   
    def _load_model(self, 
                    model_name_or_path: str, 
                    config, 
                    cache_dir):
        '''
        Carrega o modelo transformer

        Parâmetros:
           `model_name_or_path` - Nome ou caminho do modelo.
           `config` - Configuração do modelo.
           `cache_dir` - Diretório de cache.
        '''

        # Carregamento T5
        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, 
                                config, 
                                cache_dir)
        
        else:
            # Carregamento MT5
            if isinstance(config, MT5Config):
                self._load_mt5_model(model_name_or_path, 
                                    config, 
                                    cache_dir)
            else:
                # Carrega modelos genéricos
                self.auto_model = AutoModel.from_pretrained(model_name_or_path, 
                                                            config=config, 
                                                            cache_dir=cache_dir)

    # ============================   
    def _load_t5_model(self, model_name_or_path: str, 
                       config, 
                       cache_dir):
        '''
        Carrega codificador do modelo¨T5

        Parâmetros:
           `model_name_or_path` - Nome ou caminho do modelo.
           `config` - Configuração do modelo.
           `cache_dir` - Diretório de cache.
        '''

        from transformers import T5EncoderModel
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = T5EncoderModel.from_pretrained(model_name_or_path, 
                                                         config=config, 
                                                         cache_dir=cache_dir)

    # ============================   
    def _load_mt5_model(self, model_name_or_path: str, 
                        config, 
                        cache_dir):
        '''
        Carrega codificador do modelo MT5

        Parâmetros:
           `model_name_or_path` - Nome ou caminho do modelo.
           `config` - Configuração do modelo.
           `cache_dir` - Diretório de cache.
        '''

        from transformers import MT5EncoderModel
        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = MT5EncoderModel.from_pretrained(model_name_or_path, 
                                                          config=config, 
                                                          cache_dir=cache_dir)
   
    # ============================      
    def getTextoTokenizado(self, texto : str):
        '''
        Retorna um texto tokenizado e concatenado com tokens especiais '[CLS]' no início e o token '[SEP]' no fim para ser submetido ao modelo de linguagem.
        
        Parâmetros:
           `texto` - Um texto a ser tokenizado.
        
        Retorno:
           `texto_tokenizado` - Texto tokenizado.
        '''

        # Adiciona os tokens especiais.
        texto_marcado = '[CLS] ' + texto + ' [SEP]'

        # Tokeniza o texto
        texto_tokenizado = self.tokenizer.tokenize(texto_marcado)

        return texto_tokenizado

    # ============================    

    def tokenize(self, texto: Union[str, List[str]]):
        '''        
        Tokeniza um texto para submeter ao modelo de linguagem. 
        Retorna um dicionário listas de mesmo tamanho para garantir o processamento em lote.
        Use a quantidade de tokens para saber até onde deve ser recuperado em uma lista de saída.
        Ou use attention_mask diferente de 1 para saber que posições devem ser utilizadas na lista.

        Parâmetros:
           `texto` - Texto é uma string ou uma lista de strings a serem tokenizados para o modelo de linguagem.
                          
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
        if self.modelo_args.do_lower_case:
           to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        # Tokeniza o texto
        # Faz o mesmo que o método encode_plus com uma string e o mesmo que batch_encode_plus com uma lista de strings
        saida.update(self.tokenizer(*to_tokenize,  # Texto a ser codificado.
                                     add_special_tokens=True, # Adiciona os tokens especiais '[CLS]' e '[SEP]'
                                     padding=True, # Preenche o texto até max_length
                                     truncation='longest_first',  # Trunca o texto no maior texto
                                     return_tensors="pt",  # Retorna os dados como tensores pytorch.
                                     max_length=self.modelo_args.max_seq_len # Define o tamanho máximo para preencheer ou truncar.
                                    ) 
                    )
                        
        # Gera o texto tokenizado convertendo os ids para os respectivos tokens           
        saida['tokens_texto_mcl'] = [[self.tokenizer.convert_ids_to_tokens(s.item()) for s in col] for col in saida['input_ids']]

        # Guarda o texto original        
        saida['texto_original'] = [[s for s in col] for col in to_tokenize][0]     
        
        # Verifica se existe algum texto maior que o limite de tokenização
        for tokens in saida['tokens_texto_mcl']:
            if len(tokens) >= 512:
                logger.info("Utilizando embeddings do modelo de:\"{}\".".format(AbordagemExtracaoEmbeddingsCamadas.converteInt(self.modelo_args.abordagem_extracao_embeddings_camadas).getStr()))
  
        return saida
        
    # ============================           
    def getSaidaRede(self, texto: Union[str, dict]):
        '''
        De um texto preparado(tokenizado) ou não, retorna os embeddings dos tokens do texto. 
        O retorno é um dicionário com token_embeddings, input_ids, attention_mask, token_type_ids, 
        tokens_texto_mcl, texto_original  e all_layer_embeddings.
        
        Retorna os embeddings de todas as camadas de um texto.
    
        Parâmetros:
           `texto` - Um texto a ser recuperado os embeddings do modelo de linguagem
    
        Retorna um dicionário com as seguintes chaves:
           `token_embeddings` - Uma lista com os embeddings da última camada.
           `input_ids` - Uma lista com os textos indexados.            
           `attention_mask` - Uma lista com os as máscaras de atenção
           `token_type_ids` - Uma lista com os tipos dos tokens.            
           `tokens_texto_mcl` - Uma lista com os textos tokenizados com os tokens especiais.
           `texto_origina`l - Uma lista com os textos originais.
           `all_layer_embeddings` - Uma lista com os embeddings de todas as camadas.
        '''

        # Se o texto não estiver tokenizado, tokeniza
        if not isinstance(texto, dict):
            texto = self.tokenize(texto)
    
        # Recupera o texto preparado pelo tokenizador para envio ao modelo
        dic_texto_tokenizado = {'input_ids': texto['input_ids'],                                 
                                'attention_mask': texto['attention_mask']}
        
        # Se token_type_ids estiver no texto preparado copia para dicionário
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
        
        last_hidden_state = outputs[0]

        # Adiciona os embeddings da última camada e os dados do texto preparado na saída
        saida = {}
        saida.update({'token_embeddings': last_hidden_state,  # Embeddings da última camada
                      'input_ids': texto['input_ids'],
                      'attention_mask': texto['attention_mask'],
                      'token_type_ids': texto['token_type_ids'],        
                      'tokens_texto_mcl': texto['tokens_texto_mcl'],
                      'texto_original': texto['texto_original']
                      })

        # output_hidden_states == True existem embeddings nas camadas ocultas
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
    def getEmbeddingPrimeiraCamadaRede(self, saida_rede: dict):
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
        print('resultado=',resultado.size())

        return resultado

    # ============================
    def getEmbeddingPenultimaCamada(self, saida_rede: dict):
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
    def getEmbeddingUltimaCamada(self, saida_rede: dict):
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
    def getEmbeddingSoma4UltimasCamadas(self, saida_rede: dict):
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
    def getEmbeddingConcat4UltimasCamadas(self, saida_rede: dict):
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
        # Percorre os 4 últimos
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
    def getEmbeddingSomaTodasAsCamada(self, saida_rede: dict):
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
                           abordagem_extracao_embeddings_camadas: Union[int, AbordagemExtracaoEmbeddingsCamadas] = AbordagemExtracaoEmbeddingsCamadas.ULTIMA_CAMADA):
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
        else:
            if abordagem_extracao_embeddings_camadas == AbordagemExtracaoEmbeddingsCamadas.PENULTIMA_CAMADA:
              embedding_extraido_abordagem = self.getEmbeddingPenultimaCamada(saida_rede)
            else:
                if abordagem_extracao_embeddings_camadas == AbordagemExtracaoEmbeddingsCamadas.ULTIMA_CAMADA:
                  embedding_extraido_abordagem = self.getEmbeddingUltimaCamada(saida_rede)
                else:
                    if abordagem_extracao_embeddings_camadas == AbordagemExtracaoEmbeddingsCamadas.SOMA_4_ULTIMAS_CAMADAS:
                      embedding_extraido_abordagem = self.getEmbeddingSoma4UltimasCamadas(saida_rede)
                    else:
                        if abordagem_extracao_embeddings_camadas == AbordagemExtracaoEmbeddingsCamadas.CONCAT_4_ULTIMAS_CAMADAS:
                            embedding_extraido_abordagem = self.getEmbeddingConcat4UltimasCamadas(saida_rede)
                        else:
                            if abordagem_extracao_embeddings_camadas == AbordagemExtracaoEmbeddingsCamadas.TODAS_AS_CAMADAS:
                                embedding_extraido_abordagem = self.getEmbeddingSomaTodasAsCamada(saida_rede)
                            else:                                
                                logger.info("Não foi especificado uma abordagem de extração dos embeddings das camadas do transformer.") 
        
        # Verifica se foi realizado a extração
        if embedding_extraido_abordagem != None:
          # Atualiza a saída com os embeddings extraídos usando abordagem
          saida_rede.update({'embedding_extraido': embedding_extraido_abordagem,  # Embeddings extraídos usando abordagem de extração
                             'abordagem_extracao_embeddings_camadas': abordagem_extracao_embeddings_camadas})  # Tipo da abordagem da extração  dos embeddings
        else:
          logger.info("Não foi especificado uma abordagem de extração dos embeddings das camadas do transformer.") 
          saida_rede = None  

        return saida_rede

    # ============================  
    # getTokensEmbeddingsPOStexto
    # Gera os tokens, POS e embeddings de cada texto.
    
    # Dicionário de tokens de exceções e seus deslocamentos para considerar mais tokens do BERT em relação ao spaCy
    # A tokenização do BERT gera mais tokens que a tokenização das palavras do spaCy
    _dic_excecao_maior = {"":-1,
                         }
                             
    def _getExcecaoDicMaior(self, token: str):   
        '''
        Retorna o deslocamento do token no texto para considerar mais tokens do BERT em relação ao spaCy.

        Parâmetros:
           `token` - Um token a ser verificado se é uma exceção.

        Retorno:
           O deslocamento do token no texto para considerar mais tokens do BERT em relação ao spaCy.
        '''
    
        valor = self._dic_excecao_maior.get(token)
        if valor != None:
            return valor
        else:
            return -1                             
    
    # Dicionário de tokens de exceções e seus deslocamentos para considerar menos tokens do BERT em relação ao spaCy
    # A tokenização do BERT gera menos tokens que a tokenização das palavras do spaCy
    _dic_excecao_menor = {"1°":1,
                          }
    
    def _getExcecaoDicMenor(self, token: str): 
        '''
        Retorna o deslocamento do token no texto para considerar menos tokens do BERT em relação ao spaCy.

        Parâmetros:
           `token` - Um token a ser verificado se é uma exceção.

        Retorno:
           O deslocamento do token no texto para considerar menos tokens do BERT em relação ao spaCy.
        '''  
        
        valor = self._dic_excecao_menor.get(token)
        if valor != None:
            return valor
        else:
            return -1

    def getTokensEmbeddingsPOSTexto(self, 
                                    embeddings_texto, 
                                    tokens_texto_mcl,                                       
                                    tokens_texto_concatenado,
                                    pln):
        '''
        De um texto preparado(tokenizado) ou não, retorna os embeddings das palavras do texto. 
        Retorna 5 listas, os tokens(palavras), as postagging, tokens OOV, e os embeddings dos tokens igualando a quantidade de tokens do spaCy com a tokenização do MCL de acordo com a estratégia. 
        Utiliza duas estratégias para realizar o pooling de tokens que forma uma palavra.
            - Estratégia MEAN para calcular a média dos embeddings dos tokens que formam uma palavra.
            - Estratégia MAX para calcular o valor máximo dos embeddings dos tokens que formam uma palavra.
            
        Parâmetros:
           `texto` - Um texto a ser recuperado os embeddings das palavras do modelo de linguagem
           `embeddings_texto` - Os embeddings do texto gerados pelo método getEmbeddingsTexto
           `tokens_texto_mcl` - Os tokens do texto gerados pelo método getEmbeddingsTexto
           `tokens_texto_concatenado` - Os tokens do texto concatenado gerados pelo método getEmbeddingsTexto
           `pln` - Uma instância da classe PLN para realizar a tokenização e POS-Tagging do texto.
    
        Retorna um dicionário com as seguintes chaves:          
           `tokens_texto` - Uma lista com os tokens do texto gerados pelo método.
           `pos_texto_pln` - Uma lista com as postagging dos tokens gerados pela ferramenta de pln.
           `tokens_oov_texto_mcl` - Uma lista com os tokens OOV do mcl.
           `palavra_embeddings_MEAN` - Uma lista dos embeddings de palavras com a média dos embeddings(Estratégia MEAN) dos tokens que formam a palavra.
           `palavra_embeddings_MAX` - Uma lista dos embeddings de palavras com o máximo dos embeddings(Estratégia MAX) dos tokens que formam a palavra.
        '''
       
        # Guarda os tokens e embeddings de retorno
        lista_tokens = []
        lista_tokens_oov_mcl = []
        lista_palavra_embeddings_MEAN = []
        lista_palavra_embeddings_MAX = []
        
        # Gera a tokenização e POS-Tagging da sentença    
        lista_tokens_texto_pln, lista_pos_texto_pln = pln.getListaTokensPOSTexto(tokens_texto_concatenado)

        # print("\tokens_texto_concatenado    :",tokens_texto_concatenado)    
        # print("lista_tokens_texto_pln       :",lista_tokens_texto_pln)
        # print("len(lista_tokens_texto_pln)  :",len(lista_tokens_texto_pln))    
        # print("lista_pos_texto_pln          :",lista_pos_texto_pln)
        # print("len(lista_pos_texto_pln)     :",len(lista_pos_texto_pln))
        
        # embedding <qtde_tokens x 4096>        
        # print("embeddings_texto          :",embeddings_texto.shape)
        # print("tokens_texto_mcl          :",tokens_texto_mcl)
        # print("len(tokens_texto_mcl)     :",len(tokens_texto_mcl))

        # Seleciona os pares de palavra a serem avaliadas
        pos_wi = 0 # Posição do token da palavra gerado pelo spaCy
        pos_wj = pos_wi # Posição do token da palavra gerado pelo MCL
        pos2 = -1

        # Enquanto o indíce da palavra pos_wj(2a palavra) não chegou ao final da quantidade de tokens do MCL
        while pos_wj < len(tokens_texto_mcl):  

            # Seleciona os tokens da sentença
            wi = lista_tokens_texto_pln[pos_wi] # Recupera o token da palavra gerado pelo spaCy
            wi1 = ""
            pos2 = -1
            if pos_wi+1 < len(lista_tokens_texto_pln):
                wi1 = lista_tokens_texto_pln[pos_wi+1] # Recupera o próximo token da palavra gerado pelo spaCy
      
                # Localiza o deslocamento da exceção        
                pos2 = self._getExcecaoDicMenor(wi+wi1)  
                #print("Exceção pos2:", pos2)

            wj = tokens_texto_mcl[pos_wj] # Recupera o token da palavra gerado pelo MCL
            # print("wi[",pos_wi,"]=", wi)
            # print("wj[",pos_wj,"]=", wj)

            # Tratando exceções
            # Localiza o deslocamento da exceção
            pos = self._getExcecaoDicMaior(wi)  
            #print("Exceção pos:", pos)
                
            if pos != -1 or pos2 != -1:      
                if pos != -1:
                    #print("Adiciona 1 Exceção palavra == wi or palavra = [UNK]:",wi)
                    lista_tokens.append(wi)
                    # Marca como fora do vocabulário do MCL
                    lista_tokens_oov_mcl.append(1)
                    # Verifica se tem mais de um token
                    if pos != 1:
                        indice_token = pos_wj + pos
                        #print("Calcula a média de :", pos_wj , "até", indice_token)
                        embeddings_tokens_palavra = embeddings_texto[pos_wj:indice_token]
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
                        lista_palavra_embeddings_MEAN.append(embeddings_texto[pos_wj])            
                        lista_palavra_embeddings_MAX.append(embeddings_texto[pos_wj])
             
                    # Avança para a próxima palavra e token do MCL
                    pos_wi = pos_wi + 1
                    pos_wj = pos_wj + pos
                    #print("Proxima:")            
                    #print("wi[",pos_wi,"]=", texto_token[pos_wi])
                    #print("wj[",pos_wj,"]=", texto_tokenizada_MCL[pos_wj])
                else:
                    if pos2 != -1:
                        #print("Adiciona 1 Exceção palavra == wi or palavra = [UNK]:",wi)
                        lista_tokens.append(wi+wi1)
                        # Marca como fora do vocabulário do MCL
                        lista_tokens_oov_mcl.append(1)
                        # Verifica se tem mais de um token
                        if pos2 == 1: 
                            # Adiciona o embedding do token a lista de embeddings
                            lista_palavra_embeddings_MEAN.append(embeddings_texto[pos_wj])
                            lista_palavra_embeddings_MAX.append(embeddings_texto[pos_wj])
              
                        # Avança para a próxima palavra e token do MCL
                        pos_wi = pos_wi + 2
                        pos_wj = pos_wj + pos2
                        #print("Proxima:")            
                        #print("wi[",pos_wi,"]=", texto_token[pos_wi])
                        #print("wj[",pos_wj,"]=", texto_tokenizada_MCL[pos_wj])
            else:  
                # Tokens iguais adiciona a lista, o token não possui subtoken
                if (wi == wj or wj=="[UNK]"):
                    # Adiciona o token a lista de tokens
                    #print("Adiciona 2 wi==wj or wj==[UNK]:", wi )
                    lista_tokens.append(wi)    
                    # Marca como dentro do vocabulário do MCL
                    lista_tokens_oov_mcl.append(0)
                    # Adiciona o embedding do token a lista de embeddings
                    lista_palavra_embeddings_MEAN.append(embeddings_texto[pos_wj])
                    lista_palavra_embeddings_MAX.append(embeddings_texto[pos_wj])
                    #print("embedding1[pos_wj]:", embedding_texto[pos_wj].shape)
                    # Avança para a próxima palavra e token do MCL
                    pos_wi = pos_wi + 1
                    pos_wj = pos_wj + 1   
                  
                else:          
                    # A palavra foi tokenizada pelo Wordpice com ## ou diferente do spaCy ou desconhecida
                    # Inicializa a palavra a ser montada          
                    palavra_POS = wj
                    indice_token = pos_wj + 1                 
                    while  ((palavra_POS != wi) and indice_token < len(tokens_texto_mcl)):
                        if "##" in tokens_texto_mcl[indice_token]:
                            # Remove os caracteres "##" do token
                            parte = tokens_texto_mcl[indice_token][2:]
                        else:                
                            parte = tokens_texto_mcl[indice_token]
                  
                        palavra_POS = palavra_POS + parte
                        #print("palavra_POS:",palavra_POS)
                        # Avança para o próximo token do MCL
                        indice_token = indice_token + 1

                    #print("\nMontei palavra:",palavra_POS)
                    if (palavra_POS == wi or palavra_POS == "[UNK]"):
                        # Adiciona o token a lista
                        #print("Adiciona 3 palavra == wi or palavra_POS = [UNK]:",wi)
                        lista_tokens.append(wi)
                        # Marca como fora do vocabulário do MCL
                        lista_tokens_oov_mcl.append(1)
                        # Calcula a média dos tokens da palavra
                        #print("Calcula o máximo :", pos_wj , "até", indice_token)
                        embeddings_tokens_palavra = embeddings_texto[pos_wj:indice_token]
                        #print("embeddings_tokens_palavra2:",embeddings_tokens_palavra)
                        #print("embeddings_tokens_palavra2:",embeddings_tokens_palavra.shape)
                        
                        if isinstance(embeddings_tokens_palavra, torch.Tensor): 
                            # calcular a média dos embeddings dos tokens do MCL da palavra
                            embedding_estrategia_MEAN = torch.mean(embeddings_tokens_palavra, dim=0)
                            #print("embedding_estrategia_MEAN:",embedding_estrategia_MEAN)
                            #print("embedding_estrategia_MEAN.shape:",embedding_estrategia_MEAN.shape)      
                        else:
                            embedding_estrategia_MEAN = np.mean(embeddings_tokens_palavra, axis=0)      
                              
                        lista_palavra_embeddings_MEAN.append(embedding_estrategia_MEAN)
                 
                        if isinstance(embeddings_tokens_palavra, torch.Tensor): 
                            # calcular o valor máximo dos embeddings dos tokens do MCL da palavra
                            embedding_estrategia_MAX, linha = torch.max(embeddings_tokens_palavra, dim=0)
                            #print("embedding_estrategia_MAX:",embedding_estrategia_MAX)
                            #print("embedding_estrategia_MAX.shape:",embedding_estrategia_MAX.shape)     
                        else:
                             embedding_estrategia_MAX = np.max(embeddings_tokens_palavra, axis=0)
                            
                        lista_palavra_embeddings_MAX.append(embedding_estrategia_MAX)

                    # Avança para o próximo token do spaCy
                    pos_wi = pos_wi + 1
                    # Pula para o próximo token do MCL
                    pos_wj = indice_token
        
        # Verificação se as listas estão com o mesmo tamanho
        #if (len(lista_tokens) != len(texto_token)) or (len(lista_embeddings_MEAN) != len(texto_token)):
        if (len(lista_tokens) !=  len(lista_palavra_embeddings_MEAN)):
            logger.info("texto                      :{}.".format(tokens_texto_concatenado))            
            logger.info("texto_token_pln            :{}.".format(lista_tokens_texto_pln))
            logger.info("lista_pos_texto_pln        :{}.".format(lista_pos_texto_pln))
            logger.info("texto_tokenizado_mcl       :{}.".format(tokens_texto_mcl))
            logger.info("lista_tokens               :{}.".format(lista_tokens))
            logger.info("len(lista_tokens)          :{}.".format(len(lista_tokens)))
            logger.info("lista_embeddings_MEAN      :{}.".format(lista_palavra_embeddings_MEAN))
            logger.info("len(lista_embeddings_MEAN) :{}.".format(len(lista_palavra_embeddings_MEAN)))
            logger.info("lista_embeddings_MAX       :{}.".format(lista_palavra_embeddings_MAX))
            logger.info("len(lista_embeddings_MAX)  :{}.".format(len(lista_palavra_embeddings_MAX)))
       
        del embeddings_texto
        del tokens_texto_mcl
        del lista_tokens_texto_pln

        # Retorna as medidas em um dicionário
        saida = {}
        saida.update({'tokens_texto' : lista_tokens,
                      'pos_texto_pln' : lista_pos_texto_pln,
                      'tokens_oov_texto_mcl' : lista_tokens_oov_mcl,
                      'palavra_embeddings_MEAN' : lista_palavra_embeddings_MEAN,
                      'palavra_embeddings_MAX' : lista_palavra_embeddings_MAX})

        #return lista_tokens, lista_pos_texto_pln, lista_tokens_OOV_mcl, lista_embeddings_MEAN, lista_embeddings_MAX
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
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'modelo_linguagem_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    # ============================   
    def getAutoMmodel(self):
        '''
        Recupera o modelo.
        '''

        return self.auto_model

    # ============================   
    def getTokenizer(self):
        '''
        Recupera o tokenizador.
        '''

        return self.tokenizer

    # ============================   
    def batchToDevice(self, lote, target_device: device):
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
