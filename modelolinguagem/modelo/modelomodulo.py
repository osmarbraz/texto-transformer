# Import das bibliotecas.
import logging  # Biblioteca de logging
import zipfile # Biblioteca para descompactar
import os # Biblioteca de manipulação de arquivos
import shutil # iblioteca de manipulação arquivos de alto nível
import torch # Biblioteca para manipular os tensores

# Bibliotecas Transformer
#from transformers import BertModel # Importando as bibliotecas do Modelo BERT.
#from transformers import BertTokenizer # Importando as bibliotecas do tokenizador BERT.

# Import de bibliotecas próprias.
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *

from modelo.modeloarguments import ModeloArgumentos
from modelo.transformer import Transformer

# ============================
def getNomeModeloLinguagem(model_args):
    '''    
    Recupera uma string com uma descrição do modelo de linguagem para nomes de arquivos e diretórios.
    
    Parâmetros:
    `model_args` - Objeto com os argumentos do modelo.       
    
    Retorno:
    `MODELO_BERT` - Nome do modelo de linguagem.
    '''

    # Verifica o nome do modelo(default SEM_MODELO_BERT)
    MODELO_BERT = "SEM_MODELO_BERT"
    
    if 'neuralmind' in model_args.pretrained_model_name_or_path:
        MODELO_BERT = "_BERTimbau"
        
    else:
        if 'multilingual' in model_args.pretrained_model_name_or_path:
            MODELO_BERT = "_BERTmultilingual"
            
    return MODELO_BERT

# ============================
def getTamanhoModeloLinguagem(model_args):
    '''    
    Recupera uma string com o tamanho(dimensão) do modelo de linguagem para nomes de arquivos e diretórios.
    
    Parâmetros:
    `model_args` - Objeto com os argumentos do modelo.       
    
    Retorno:
    `TAMANHO_BERT` - Nome do tamanho do modelo de linguagem.
    '''
    
    # Verifica o tamanho do modelo(default large)
    TAMANHO_BERT = "_large"
    
    if 'base' in model_args.pretrained_model_name_or_path:
        TAMANHO_BERT = "_base"
        
    return TAMANHO_BERT  



# ============================
def carregaModeloLinguagem(modelo_argumentos):
    ''' 
    Carrega o BERT para cálculo de medida e retorna o modelo e o tokenizador.
    O tipo do model retornado é BertModel.
    
    Parâmetros:
    `modelo_argumentos` - Objeto com os argumentos do modelo.               

    Retorno:    
    `model` - Um objeto do modelo BERT carregado.       
    `tokenizer` - Um objeto tokenizador BERT carregado.       
    ''' 
            
    # Verifica a origem do modelo
    # DIRETORIO_MODELO = verificaModelo(modelo_argumentos)
    
    # Variável para conter o modelo
    model = None
    
    # Carrega o model
    #model = carregaModelo(DIRETORIO_MODELO, modelo_argumentos)
    
    # Carrega o modelo de linguagem    
    model_args = {"output_attentions": modelo_argumentos.output_attentions, 
                         "output_hidden_states": modelo_argumentos.output_hidden_states}
    transformer_model = Transformer(modelo_argumentos, model_args=model_args)
       
    # Carrega o tokenizador. 
    # O tokenizador é o mesmo para o classificador e medidor.
    #tokenizer = carregaTokenizadorModeloPretreinado(DIRETORIO_MODELO, model_args)
    
    model = transformer_model.get_auto_model()
    tokenizer = transformer_model.get_tokenizer()
    
    return model, tokenizer
