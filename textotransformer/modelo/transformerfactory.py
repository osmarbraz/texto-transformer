# Import das bibliotecas.

# Biblioteca de logging
import logging  

# Biblioteca de logging
import logging  
# Biblioteca de tipos
from typing import Dict, Optional
# Biblioteca do transformer hunggingface
from transformers import AutoModel, AutoTokenizer, AutoConfig, T5Config, MT5Config
from transformers import AlbertModel, BertModel, DistilBertModel, GPT2Model, OpenAIGPTModel, RobertaModel, XLNetModel

# Bibliotecas próprias
from textotransformer.modelo.transformeralbert import TransformerAlbert
from textotransformer.modelo.transformerbert import TransformerBert
from textotransformer.modelo.transformerdistilbert import TransformerDistilbert
from textotransformer.modelo.transformergpt2 import TransformerGPT2
from textotransformer.modelo.transformeropenaigpt import TransformerOpenAIGPT
from textotransformer.modelo.transformerroberta import TransformerRoberta
from textotransformer.modelo.transformerxlnet import TransformerXLNet
from textotransformer.modelo.modeloargumentos import ModeloArgumentos

# Objeto de logger
logger = logging.getLogger(__name__)

class TransformerFactory():
    '''
    Classe construtora de objetos Transformer de Texto-Transformer.
    Retorna um objeto Transformer de acordo com auto_model dos parâmetros. 
    '''
    
    @staticmethod
    def getTransformer(modelo_args : ModeloArgumentos,
                       cache_dir: Optional[str] = None,
                       tokenizer_args: Dict = {}, 
                       tokenizer_name_or_path : str = None):
        ''' 
        Retorna um objeto Transformer de Texto-Transformer de acordo com auto_model. 
        Para o Albert que utiliza AlbertaModel, retorna um TransformerAlbert.
        Para o BERT que utiliza BertModel, retorna um TransformerBert.
        Para o RoBERTa que utiliza RobertaModel, retorna um TransformerRoberta.
        Para o Distilbert que utiliza DistilbertModel, retorna um TransformerDistilbert.
        Para o Distilbert que utiliza DistilbertModel, retorna um TransformerGPT2.
            
        Parâmetros:
           `modelo_args' - Argumentos passados para o modelo Huggingface Transformers.
           `cache_dir` - Cache dir para Huggingface Transformers para armazenar/carregar modelos.
           `tokenizer_args` - Argumentos (chave, pares de valor) passados para o modelo Huggingface Tokenizer
           `tokenizer_name_or_path` - Nome ou caminho do tokenizer. Quando None, model_name_or_path é usado.
        '''    
        
        # Recupera parâmetros do transformador dos argumentos e cria um dicionário para o AutoConfig
        modelo_args_config = {"output_attentions": modelo_args.output_attentions, 
                              "output_hidden_states": modelo_args.output_hidden_states}
    
        # Configuração do modelo        
        auto_config = AutoConfig.from_pretrained(modelo_args.pretrained_model_name_or_path,
                                                 **modelo_args_config, 
                                                 cache_dir=cache_dir)
        
        # Carrega o modelo
        auto_model = TransformerFactory._carregar_modelo(modelo_args.pretrained_model_name_or_path,
                                                         auto_config, 
                                                         cache_dir)
        
        # Carrega o tokenizador
        auto_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else  modelo_args.pretrained_model_name_or_path,
                                                       cache_dir=cache_dir, 
                                                       **tokenizer_args)
        
        # Se max_seq_length não foi especificado, tenta inferir do modelo
        if modelo_args.max_seq_len is None:
            if hasattr(auto_model, "config") and hasattr(auto_model.config, "max_position_embeddings") and hasattr(auto_tokenizer, "model_max_length"):
                modelo_args.max_seq_len = min(auto_model.config.max_position_embeddings,
                                                   auto_tokenizer.model_max_length)

        # Define a classe do tokenizador
        if tokenizer_name_or_path is not None:
            auto_model.config.tokenizer_class = auto_tokenizer.__class__.__name__

        # Verifica qual o modelo deve ser retornado pelos parâmetros auto                                  
        return TransformerFactory.getInstanciaTransformer(auto_model=auto_model, 
                                                          auto_config=auto_config, 
                                                          auto_tokenizer=auto_tokenizer, 
                                                          modelo_args=modelo_args)
    
    # ============================ 
    @staticmethod
    def getInstanciaTransformer(auto_model: AutoModel, 
                                auto_config: AutoConfig, 
                                auto_tokenizer: AutoTokenizer, 
                                modelo_args:ModeloArgumentos = None):
        '''
        Retorna uma classe Transformer com um modelo de linguagem carregado de acordo com os parâmetros auto_model.

        Parâmetros:
           `auto_model` - Auto model modelo carregado.
           `auto_config` - Auto config carregado.
           `auto_tokenizer` - Auto tokenizer carregado.
           `modelo_args' - Argumentos passados para o modelo Huggingface Transformers.
        '''
        
        if isinstance(auto_model, BertModel):
            return TransformerBert(auto_model=auto_model, 
                                   auto_config=auto_config, 
                                   auto_tokenizer=auto_tokenizer, 
                                   modelo_args=modelo_args)
        elif isinstance(auto_model, AlbertModel):
            return TransformerAlbert(auto_model=auto_model, 
                                     auto_config=auto_config, 
                                     auto_tokenizer=auto_tokenizer, 
                                     modelo_args=modelo_args)
        elif isinstance(auto_model, DistilBertModel):            
            return TransformerDistilbert(auto_model=auto_model, 
                                         auto_config=auto_config, 
                                         auto_tokenizer=auto_tokenizer, 
                                         modelo_args=modelo_args)
        elif isinstance(auto_model, GPT2Model):
            return TransformerGPT2(auto_model=auto_model, 
                                   auto_config=auto_config, 
                                   auto_tokenizer=auto_tokenizer, 
                                   modelo_args=modelo_args)
        elif isinstance(auto_model, OpenAIGPTModel):
            return TransformerOpenAIGPT(auto_model=auto_model, 
                                        auto_config=auto_config, 
                                        auto_tokenizer=auto_tokenizer, 
                                        modelo_args=modelo_args)
        elif isinstance(auto_model, RobertaModel):
            return TransformerRoberta(auto_model=auto_model, 
                                      auto_config=auto_config, 
                                      auto_tokenizer=auto_tokenizer, 
                                      modelo_args=modelo_args)        
        elif isinstance(auto_model, XLNetModel):
            return TransformerXLNet(auto_model=auto_model, 
                                    auto_config=auto_config, 
                                    auto_tokenizer=auto_tokenizer, 
                                    modelo_args=modelo_args)        
        else:
            logger.error("Modelo não suportado: \"{}\".".format(auto_model.__class__.__name__))
            
            return None
    
    # ============================ 
    @staticmethod
    def _carregar_modelo(model_name_or_path: str, 
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
            return TransformerFactory._load_t5_model(model_name_or_path=model_name_or_path,
                                                     config=config, 
                                                     cache_dir=cache_dir)
        
        else:
            # Carregamento MT5
            if isinstance(config, MT5Config):
                return TransformerFactory._load_mt5_model(model_name_or_path=model_name_or_path,
                                                          config=config, 
                                                          cache_dir=cache_dir)
            else:
                # Carrega modelos genéricos
                return AutoModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                 config=config, 
                                                 cache_dir=cache_dir)

    # ============================ 
    @staticmethod
    def _load_t5_model(model_name_or_path: str, 
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
        
        return T5EncoderModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path, 
                                              config=config, 
                                              cache_dir=cache_dir)

    # ============================ 
    @staticmethod
    def _load_mt5_model(model_name_or_path: str, 
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
        
        return MT5EncoderModel.from_pretrained(pretrained_model_name_or_path=model_name_or_path, 
                                               config=config, 
                                               cache_dir=cache_dir)