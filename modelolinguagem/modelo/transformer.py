# Import das bibliotecas.
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
from modelo.modeloarguments import ModeloArgumentos

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
            
        logging.info("Transformer carregado: {}.".format(modelo_args))            


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

    def _load_t5_model(self, model_name_or_path, config, cache_dir):
        '''
        Carrega codificador do modelo¨T5
        '''
        from transformers import T5EncoderModel
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = T5EncoderModel.from_pretrained(model_name_or_path, 
                                                         config=config, 
                                                         cache_dir=cache_dir)

    def _load_mt5_model(self, model_name_or_path, config, cache_dir):
        '''
        Carrega codificador do modelo MT5
        '''
        from transformers import MT5EncoderModel
        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = MT5EncoderModel.from_pretrained(model_name_or_path, 
                                                          config=config, 
                                                          cache_dir=cache_dir)

    def __repr__(self):
        '''
        Retorna uma string com descrição do objeto
        '''
        return "Transformer({}) com modelo Transformer: {} ".format(self.get_config_dict(), 
                                                                    self.auto_model.__class__.__name__)
   
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
                logging.info("Utilizando embeddings do modelo de:", listaTipoCamadas[modelo_argumentos.camadas_embeddings]) 
                        
        return saida
        
        
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

    def get_dimensao_embedding(self) -> int:
        '''
        Retorna a dimensão do embedding
        '''
        return self.auto_model.config.hidden_size        
        
        
    def get_config_dict(self):
        '''
        Retorna as configurações com dicionário
        '''
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        '''
        Salva o modelo.
        '''
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'modelo_linguagem_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    def get_auto_model(self):
        '''
        Recupera o modelo.
        '''
        return self.auto_model

    def get_tokenizer(self):
        '''
        Recupera o tokenizador.
        '''
        return self.tokenizer


