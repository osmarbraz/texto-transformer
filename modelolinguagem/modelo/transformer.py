# Import das bibliotecas.
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config, MT5Config
import json
from typing import List, Dict, Optional, Union, Tuple
import os

# Biblioteca dos modelos de linguagem
from modelo.modeloarguments import ModeloArgumentos

class Transformer(nn.Module):
    """Huggingface AutoModel para gerar embeddings de token, palavra, sentença ou documento.
     Carrega a classe correta, por exemplo BERT / RoBERTa etc.

     :param modelo_args: Argumentos passados para o modelo Huggingface Transformers          
     :param cache_dir: Cache dir para Huggingface Transformers para armazenar/carregar modelos
     :param tokenizer_args: Argumentos (chave, pares de valor) passados para o modelo Huggingface Tokenizer
     :param tokenizer_name_or_path: Nome ou caminho do tokenizer. Quando None, model_name_or_path é usado
    """
    def __init__(self, 
                modelo_args : ModeloArgumentos,
                max_seq_length: Optional[int] = None,                
                cache_dir: Optional[str] = None,
                tokenizer_args: Dict = {}, 
                tokenizer_name_or_path : str = None):
        
        # Inicializa o construtor
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


    def _load_model(self, model_name_or_path, config, cache_dir):
        """Carrega o modelo transformer"""
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
        """Carrega codificador do modelo¨T5"""
        from transformers import T5EncoderModel
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = T5EncoderModel.from_pretrained(model_name_or_path, 
                                                         config=config, 
                                                         cache_dir=cache_dir)

    def _load_mt5_model(self, model_name_or_path, config, cache_dir):
        """Carrega codificador do modelo MT5"""
        from transformers import MT5EncoderModel
        MT5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = MT5EncoderModel.from_pretrained(model_name_or_path, 
                                                          config=config, 
                                                          cache_dir=cache_dir)

    def __repr__(self):
        """Retorna uma string com descrição do objeto"""
        return "Transformer({}) com modelo Transformer: {} ".format(self.get_config_dict(), 
                                                                    self.auto_model.__class__.__name__)

    def forward(self, features):
        """Retorna token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Alguns modelos apenas geram last_hidden_states e all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size

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

        # Documento tokenizado
        textoTokenizado = self.tokenizer.tokenize(textoMarcado)

        return textoTokenizado

    def tokenize(self, textos: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """        
        Tokeniza um texto para submeter ao modelo de linguagem.
        
        :param textos: Nome ou caminho do tokenizer. Quando None, model_name_or_path é usado
         
        Retorna um dicionário
        """
        
        saida = {}
        
        # Se o texto for uma string coloca em uma lista para tokenizar
        if isinstance(textos[0], str):
            to_tokenize = [textos]
            
        else:
            # Se o texto for um dicionário
            if isinstance(textos[0], dict):
                to_tokenize = []
                saida['texto_keys'] = []
                for lookup in textos:
                    texto_key, texto = next(iter(lookup.items()))
                    to_tokenize.append(texto)
                    saida['texto_keys'].append(texto_key)
                
                to_tokenize = [to_tokenize]
            else:
                # Se o texto for uma lista
                batch1, batch2 = [], []
                for texto_tuple in textos:
                    batch1.append(texto_tuple[0])
                    batch2.append(texto_tuple[1])
                    
                to_tokenize = [batch1, batch2]

        # Remove os espaços em branco antes e depois de cada palavra usando strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Se for para colocar para minúscolo usa Lowercase
        if self.do_lower_case:
           to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        saida.update(self.tokenizer(*to_tokenize, 
                                     padding=True, 
                                     truncation='longest_first', 
                                     return_tensors="pt", 
                                     max_length=self.max_seq_length))
        
            
        # Documento tokenizado        
        #saida['tokens_texto'] = []
        #for texto in to_tokenize:
        #    saida['tokens_texto'].append(self.getTextoTokenizado(texto))
                
        return saida

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'modelo_linguagem_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    def get_auto_model(self):
        return self.auto_model

    def get_tokenizer(self):
        return self.tokenizer


