# Import das bibliotecas.
import logging  # Biblioteca de logging
# Biblioteca de aprendizado de máquina
from torch import nn 
import torch 
# Biblioteca do transformer
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config, MT5Config
# Biblioteca de manipulação json
import json
# Biblioteca de tipos
from typing import List, Dict, Optional, Union, Tuple
# Biblioteca de manipulação sistema
import os

# Biblioteca dos modelos de linguagem
from modelolinguagem.modelo.modeloarguments import ModeloArgumentos

logger = logging.getLogger(__name__)

class Transformer(nn.Module):
    '''
    Huggingface AutoModel para gerar embeddings de token, palavra, sentença ou texto.
    Carrega a classe correta, por exemplo BERT / RoBERTa etc.

     :param modelo_args: Argumentos passados para o modelo Huggingface Transformers          
     :param cache_dir: Cache dir para Huggingface Transformers para armazenar/carregar modelos
     :param tokenizer_args: Argumentos (chave, pares de valor) passados para o modelo Huggingface Tokenizer
     :param tokenizer_name_or_path: Nome ou caminho do tokenizer. Quando None, model_name_or_path é usado
    
    '''
    def __init__(self, 
                modelo_args : ModeloArgumentos,
                max_seq_length: Optional[int] = None,                
                cache_dir: Optional[str] = None,
                tokenizer_args: Dict = {}, 
                tokenizer_name_or_path : str = None):
        
        # Inicializa o construtor da superclasse
        super(Transformer, self).__init__()
        
        # Recupera o nome do modelo dos argumentos
        model_name_or_path = modelo_args.pretrained_model_name_or_path;
        # Recupera o tamanho máximo de um texto
        max_seq_length = modelo_args.max_seq_len
        
        # Parâmetros do modelo
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = modelo_args.do_lower_case

        # Recupera parâmetros do transformador dos argumentos
        model_args = {"output_attentions": modelo_args.output_attentions, 
                      "output_hidden_states": modelo_args.output_hidden_states}
    
        # Configuração do modelo        
        config = AutoConfig.from_pretrained(model_name_or_path, 
                                            **model_args, 
                                            cache_dir=cache_dir)
        
        # Carrega o modelo
        self._load_model(model_name_or_path, 
                         config, 
                         cache_dir)

        # Carrega o tokenizador
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else  model_name_or_path, cache_dir=cache_dir, **tokenizer_args)

        #Se max_seq_length não foi especificado, tenta inferir do modelo
        if max_seq_length is None:
            if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(self.auto_model.config.max_position_embeddings,
                                     self.tokenizer.model_max_length)

        # Define o máximo de tokens
        self.max_seq_length = max_seq_length

        # Define a classe do tokenizador
        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__
            
        logger.info("Classe Transformer carregada: {}.".format(modelo_args))            

    # ============================   
    def _load_model(self, model_name_or_path, config, cache_dir):
        '''
        Carrega o modelo transformer
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
    def _load_t5_model(self, model_name_or_path, config, cache_dir):
        '''
        Carrega codificador do modelo¨T5
        '''
        from transformers import T5EncoderModel
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = T5EncoderModel.from_pretrained(model_name_or_path, 
                                                         config=config, 
                                                         cache_dir=cache_dir)

    # ============================   
    def _load_mt5_model(self, model_name_or_path, config, cache_dir):
        '''
        Carrega codificador do modelo MT5
        '''
        from transformers import MT5EncoderModel
        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = MT5EncoderModel.from_pretrained(model_name_or_path, 
                                                          config=config, 
                                                          cache_dir=cache_dir)
    
    # ============================   
    def __repr__(self):
        '''
        Retorna uma string com descrição do objeto
        '''
        return "Transformer({}) com modelo Transformer: {} ".format(self.get_config_dict(), 
                                                                    self.auto_model.__class__.__name__)
    # ============================      
    def getTextoTokenizado(self, texto: str):
        '''
        Retorna um texto tokenizado e concatenado com tokens especiais '[CLS]' no início e o token '[SEP]' no fim para ser submetido ao modelo de linguagem.
        
        Parâmetros:
        `texto` - Um texto a ser tokenizado.
        
        Retorno:
        `textoTokenizado` - Texto tokenizado.
        '''

        # Adiciona os tokens especiais.
        textoMarcado = '[CLS] ' + texto + ' [SEP]'

        # Tokeniza o texto
        textoTokenizado = self.tokenizer.tokenize(textoMarcado)

        return textoTokenizado
        
    # ============================       
    def tokenize(self, textos):
        '''        
        Tokeniza um texto para submeter ao modelo de linguagem.
        
        :param textos: Texto a ser tokenizado para o modelo de linguagem.
         
        Retorna um dicionário com:
            tokens_texto uma lista com os textos tokenizados com os tokens especiais.
            input_ids uma lista com os ids dos tokens de entrada mapeados em seus índices do vocabuário.
            token_type_ids uma lista com os tipos dos tokens.
            attention_mask uma lista com os as máscaras de atenção indicando com '1' os tokens  pertencentes à sentença.
        '''
        
        saida = {}
        
        # Se o texto for uma string coloca em uma lista de listas para tokenizar
        if isinstance(textos, str):
            to_tokenize = [[textos]]
        else:
            # Se for uma lista de strings coloca em uma lista para tokenizar
            if isinstance(textos[0], str):
                to_tokenize = [textos]
            else:
                # Se for uma lista de listas de strings, não faz nada
                to_tokenize = textos                          
                
        # Remove os espaços em branco antes e depois de cada texto usando strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Se for para colocar para minúsculo usa Lowercase nos textos
        if self.do_lower_case:
           to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        # Tokeniza o texto
        # Faz o mesmo que o método encode_plus com uma string e o mesmo que batch_encode_plus com uma lista de strings
        saida.update(self.tokenizer(*to_tokenize,  # Texto a ser codificado.
                                     add_special_tokens=True, # Adiciona os tokens especiais '[CLS]' e '[SEP]'
                                     padding=True, # Preenche o texto até max_length
                                     truncation='longest_first',  # Trunca o texto no maior texto
                                     return_tensors="pt",  # Retorna os dados como tensores pytorch.
                                     max_length=self.max_seq_length # Define o tamanho máximo para preencheer ou truncar.
                                    ) 
                    )
        
                
        # Gera o texto tokenizado        
        saida['tokens_texto'] = [[self.getTextoTokenizado(s) for s in col] for col in to_tokenize][0]
        
        # Verifica se existe algum texto maior que o limite de tokenização
        for tokens in  saida['tokens_texto']:
            if len(tokens) >= 512:
                logger.info("Utilizando embeddings do modelo de: {}.".format(listaTipoCamadas[modelo_argumentos.camadas_embeddings]))   
                        
        return saida
        
    # ============================           
    def getEmbeddings(self, texto_preparado):
        '''
        Retorna token_embeddings, cls_token
        '''
        
        '''   
        Retorna os embeddings de todas as camadas de um texto.
    
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings do modelo de linguagem
    
        Retorna um dicionário com:
            tokens_texto uma lista com os textos tokenizados com os tokens especiais.
            input_ids uma lista com os textos indexados.
            token_type_ids uma lista com os tipos dos tokens.
            attention_mask uma lista com os as máscaras de atenção
            token_embeddings uma lista com os embeddings da última camada
            all_layer_embeddings uma lista com os embeddings de todas as camadas.
        '''
    
        # Recupera o texto preparado pelo tokenizador
        dic_texto_preparado = {'input_ids': texto_preparado['input_ids'], 
                               'attention_mask': texto_preparado['attention_mask']}
        
        # Se token_type_ids estiver no texto preparado copia para dicionário
        if 'token_type_ids' in texto_preparado:
            dic_texto_preparado['token_type_ids'] = texto_preparado['token_type_ids']

        # Roda o texto através do modelo, e coleta todos os estados ocultos produzidos.
        with torch.no_grad():

            outputs = self.auto_model(**dic_texto_preparado, 
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
                      'input_ids': texto_preparado['input_ids'],
                      'attention_mask': texto_preparado['attention_mask'],
                      'input_ids': texto_preparado['input_ids'],        
                      'token_type_ids': texto_preparado['token_type_ids'],        
                      'tokens_texto': texto_preparado['tokens_texto']
                      }
                     )

        # output_hidden_states == True existem embeddings nas camadas ocultas
        if self.auto_model.config.output_hidden_states:
            # 2 é o índice da saída com todos os embeddings em outputs
            all_layer_idx = 2
            if len(outputs) < 3: #Alguns modelos apenas geram last_hidden_states e all_hidden_states
                all_layer_idx = 1

            hidden_states = outputs[all_layer_idx]
            # Adiciona os embeddings de todas as camadas na saída
            saida.update({'all_layer_embeddings': hidden_states})

        return saida

    # ============================  
    # getTokensEmbeddingsPOSSentenca
    # Gera os tokens, POS e embeddings de cada sentença.
    
    # Dicionário de tokens de exceções e seus deslocamentos para considerar mais tokens do BERT em relação ao spaCy
    # A tokenização do BERT gera mais tokens que a tokenização das palavras do spaCy
    dic_excecao_maior = {"":-1,
                        }
                             
    def getExcecaoDicMaior(self, token):   
    
        valor = self.dic_excecao_maior.get(token)
        if valor != None:
            return valor
        else:
            return -1                             
    
    # Dicionário de tokens de exceções e seus deslocamentos para considerar menos tokens do BERT em relação ao spaCy
    # A tokenização do BERT gera menos tokens que a tokenização das palavras do spaCy
    dic_excecao_menor = {"1°":1,
                        }
    def getExcecaoDicMenor(self, token):   
        
        valor = self.dic_excecao_menor.get(token)
        if valor != None:
            return valor
        else:
            return -1


    def getTokensEmbeddingsPOSSentenca(self, 
                                       embedding_documento, 
                                       token_MCL_documento,                                       
                                       sentenca,
                                       NLP):
        '''    
          Retorna os tokens, as postagging e os embeddings dos tokens igualando a quantidade de tokens do spaCy com a tokenização do MCL de acordo com a estratégia. 
          Utiliza duas estratégias para realizar o pooling de tokens que forma uma palavra.
            - Estratégia MEAN para calcular a média dos embeddings dos tokens que formam uma palavra.
            - Estratégia MAX para calcular o valor máximo dos embeddings dos tokens que formam uma palavra.
        '''
       
        #Guarda os tokens e embeddings
        lista_tokens = []
        lista_tokens_OOV = []
        lista_embeddings_MEAN = []
        lista_embeddings_MAX = []
        
        # Gera a tokenização e POS-Tagging da sentença    
        sentenca_token, sentenca_pos = NLP.getListaTokensPOSSentenca(sentenca)

        # print("\nsentenca          :",sentenca)    
        # print("sentenca_token      :",sentenca_token)
        # print("len(sentenca_token) :",len(sentenca_token))    
        # print("sentenca_pos        :",sentenca_pos)
        # print("len(sentenca_pos)   :",len(sentenca_pos))
        
        # Recupera os embeddings da sentença dos embeddings do documento    
        embedding_sentenca = embedding_documento    
        sentenca_tokenizada_MCL = token_MCL_documento
        
        # embedding <qtde_tokens x 4096>        
        # print("embedding_sentenca          :",embedding_sentenca.shape)
        # print("sentenca_tokenizada_MCL     :",sentenca_tokenizada_MCL)
        # print("len(sentenca_tokenizada_MCL):",len(sentenca_tokenizada_MCL))

        # Seleciona os pares de palavra a serem avaliadas
        pos_wi = 0 # Posição do token da palavra gerado pelo spaCy
        pos_wj = pos_wi # Posição do token da palavra gerado pelo MCL
        pos2 = -1

        # Enquanto o indíce da palavra pos_wj(2a palavra) não chegou ao final da quantidade de tokens do MCL
        while pos_wj < len(sentenca_tokenizada_MCL):  

            # Seleciona os tokens da sentença
            wi = sentenca_token[pos_wi] # Recupera o token da palavra gerado pelo spaCy
            wi1 = ""
            pos2 = -1
            if pos_wi+1 < len(sentenca_token):
                wi1 = sentenca_token[pos_wi+1] # Recupera o próximo token da palavra gerado pelo spaCy
      
                # Localiza o deslocamento da exceção        
                pos2 = self.getExcecaoDicMenor(wi+wi1)  
                #print("Exceção pos2:", pos2)

            wj = sentenca_tokenizada_MCL[pos_wj] # Recupera o token da palavra gerado pelo MCL
            # print("wi[",pos_wi,"]=", wi)
            # print("wj[",pos_wj,"]=", wj)

            # Tratando exceções
            # Localiza o deslocamento da exceção
            pos = self.getExcecaoDicMaior(wi)  
            #print("Exceção pos:", pos)
                
            if pos != -1 or pos2 != -1:      
                if pos != -1:
                    #print("Adiciona 1 Exceção palavra == wi or palavra = [UNK]:",wi)
                    lista_tokens.append(wi)
                    # Marca como fora do vocabulário do MCL
                    lista_tokens_OOV.append(1)
                    # Verifica se tem mais de um token
                    if pos != 1:
                        indice_token = pos_wj + pos
                        #print("Calcula a média de :", pos_wj , "até", indice_token)
                        embeddings_tokens_palavra = embedding_sentenca[pos_wj:indice_token]
                        #print("embeddings_tokens_palavra:",embeddings_tokens_palavra.shape)
                        # calcular a média dos embeddings dos tokens do MCL da palavra
                        embedding_estrategia_MEAN = torch.mean(embeddings_tokens_palavra, dim=0)
                        #print("embedding_estrategia_MEAN:",embedding_estrategia_MEAN.shape)
                        lista_embeddings_MEAN.append(embedding_estrategia_MEAN)

                        # calcular o máximo dos embeddings dos tokens do MCL da palavra
                        embedding_estrategia_MAX, linha = torch.max(embeddings_tokens_palavra, dim=0)
                        #print("embedding_estrategia_MAX:",embedding_estrategia_MAX.shape)
                        lista_embeddings_MAX.append(embedding_estrategia_MAX)
                    else:
                        # Adiciona o embedding do token a lista de embeddings
                        lista_embeddings_MEAN.append(embedding_sentenca[pos_wj])            
                        lista_embeddings_MAX.append(embedding_sentenca[pos_wj])
             
                    # Avança para a próxima palavra e token do MCL
                    pos_wi = pos_wi + 1
                    pos_wj = pos_wj + pos
                    #print("Proxima:")            
                    #print("wi[",pos_wi,"]=", sentenca_token[pos_wi])
                    #print("wj[",pos_wj,"]=", sentenca_tokenizada_MCL[pos_wj])
                else:
                    if pos2 != -1:
                        #print("Adiciona 1 Exceção palavra == wi or palavra = [UNK]:",wi)
                        lista_tokens.append(wi+wi1)
                        # Marca como fora do vocabulário do MCL
                        lista_tokens_OOV.append(1)
                        # Verifica se tem mais de um token
                        if pos2 == 1: 
                            # Adiciona o embedding do token a lista de embeddings
                            lista_embeddings_MEAN.append(embedding_sentenca[pos_wj])
                            lista_embeddings_MAX.append(embedding_sentenca[pos_wj])
              
                        # Avança para a próxima palavra e token do MCL
                        pos_wi = pos_wi + 2
                        pos_wj = pos_wj + pos2
                        #print("Proxima:")            
                        #print("wi[",pos_wi,"]=", sentenca_token[pos_wi])
                        #print("wj[",pos_wj,"]=", sentenca_tokenizada_MCL[pos_wj])
            else:  
                # Tokens iguais adiciona a lista, o token não possui subtoken
                if (wi == wj or wj=="[UNK]"):
                    # Adiciona o token a lista de tokens
                    #print("Adiciona 2 wi==wj or wj==[UNK]:", wi )
                    lista_tokens.append(wi)    
                    # Marca como dentro do vocabulário do MCL
                    lista_tokens_OOV.append(0)
                    # Adiciona o embedding do token a lista de embeddings
                    lista_embeddings_MEAN.append(embedding_sentenca[pos_wj])
                    lista_embeddings_MAX.append(embedding_sentenca[pos_wj])
                    #print("embedding1[pos_wj]:", embedding_sentenca[pos_wj].shape)
                    # Avança para a próxima palavra e token do MCL
                    pos_wi = pos_wi + 1
                    pos_wj = pos_wj + 1   
                  
                else:          
                    # A palavra foi tokenizada pelo Wordpice com ## ou diferente do spaCy ou desconhecida
                    # Inicializa a palavra a ser montada          
                    palavra_POS = wj
                    indice_token = pos_wj + 1                 
                    while  ((palavra_POS != wi) and indice_token < len(sentenca_tokenizada_MCL)):
                        if "##" in sentenca_tokenizada_MCL[indice_token]:
                            # Remove os caracteres "##" do token
                            parte = sentenca_tokenizada_MCL[indice_token][2:]
                        else:                
                            parte = sentenca_tokenizada_MCL[indice_token]
                  
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
                        lista_tokens_OOV.append(1)
                        # Calcula a média dos tokens da palavra
                        #print("Calcula o máximo :", pos_wj , "até", indice_token)
                        embeddings_tokens_palavra = embedding_sentenca[pos_wj:indice_token]
                        #print("embeddings_tokens_palavra2:",embeddings_tokens_palavra)
                        #print("embeddings_tokens_palavra2:",embeddings_tokens_palavra.shape)
                  
                        # calcular a média dos embeddings dos tokens do MCL da palavra
                        embedding_estrategia_MEAN = torch.mean(embeddings_tokens_palavra, dim=0)        
                        #print("embedding_estrategia_MEAN:",embedding_estrategia_MEAN)
                        #print("embedding_estrategia_MEAN.shape:",embedding_estrategia_MEAN.shape)      
                        lista_embeddings_MEAN.append(embedding_estrategia_MEAN)
                 
                        # calcular o valor máximo dos embeddings dos tokens do MCL da palavra
                        embedding_estrategia_MAX, linha = torch.max(embeddings_tokens_palavra, dim=0)
                        #print("embedding_estrategia_MAX:",embedding_estrategia_MAX)
                        #print("embedding_estrategia_MAX.shape:",embedding_estrategia_MAX.shape)     
                        lista_embeddings_MAX.append(embedding_estrategia_MAX)

                    # Avança para o próximo token do spaCy
                    pos_wi = pos_wi + 1
                    # Pula para o próximo token do MCL
                    pos_wj = indice_token
        
        # Verificação se as listas estão com o mesmo tamanho
        #if (len(lista_tokens) != len(sentenca_token)) or (len(lista_embeddings_MEAN) != len(sentenca_token)):
        if (len(lista_tokens) !=  len(lista_embeddings_MEAN)):
            logger.info("sentenca                   :{}.".format(sentenca))
            logger.info("sentenca_pos               :{}.".format(sentenca_pos))
            logger.info("sentenca_token             :{}.".format(sentenca_token))
            logger.info("sentenca_tokenizada_MCL    :{}.".format(sentenca_tokenizada_MCL))
            logger.info("lista_tokens               :{}.".format(lista_tokens))
            logger.info("len(lista_tokens)          :{}.".format(len(lista_tokens)))
            logger.info("sentenca_token             :{}.".format(sentenca_token))            
            logger.info("lista_embeddings_MEAN      :{}.".format(lista_embeddings_MEAN))
            logger.info("len(lista_embeddings_MEAN) :{}.".format(len(lista_embeddings_MEAN)))
            logger.info("lista_embeddings_MAX       :{}.".format(lista_embeddings_MAX))
            logger.info("len(lista_embeddings_MAX)  :{}.".format(len(lista_embeddings_MAX)))
            
       
        del embedding_sentenca
        del token_MCL_documento
        del sentenca_tokenizada_MCL
        del sentenca_token

        return lista_tokens, sentenca_pos, lista_tokens_OOV, lista_embeddings_MEAN, lista_embeddings_MAX


    # ============================   
    def get_dimensao_embedding(self) -> int:
        '''
        Retorna a dimensão do embedding
        '''
        return self.auto_model.config.hidden_size        
        
    # ============================           
    def get_config_dict(self):
        '''
        Retorna as configurações com dicionário
        '''
        return {key: self.__dict__[key] for key in self.config_keys}

    # ============================   
    def save(self, output_path: str):
        '''
        Salva o modelo.
        '''
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'modelo_linguagem_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    # ============================   
    def get_auto_model(self):
        '''
        Recupera o modelo.
        '''
        return self.auto_model

    # ============================   
    def get_tokenizer(self):
        '''
        Recupera o tokenizador.
        '''
        return self.tokenizer


