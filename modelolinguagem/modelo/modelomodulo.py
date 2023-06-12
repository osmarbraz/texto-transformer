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
def downloadModeloPretreinado(model_args):
    ''' 
    Realiza o download do modelo BERT(MODELO) e retorna o diretório onde o modelo foi descompactado.
    
    Parâmetros:
    `model_args` - Objeto com os argumentos do modelo.
    
    Retorno:
    `DIRETORIO_MODELO` - Diretório de download do modelo.
    ''' 
    
    # Nome diretório base modelo BERT
    NOME_DIRETORIO_BASE_MODELO = "modeloBERT"
    
    # Verifica se existe o diretório base do cohebert e retorna o nome do diretório
    DIRETORIO_MODELO_LINGUAGEM = verificaDiretorioModeloLinguagem()
    
    # Recupera o nome ou caminho do modelo
    MODELO = model_args.pretrained_model_name_or_path

    # Variável para setar o arquivo.
    URL_MODELO = None

    if "http" in MODELO:
        URL_MODELO = MODELO

    # Se a variável foi setada.
    if URL_MODELO:

        # Diretório do modelo.
        DIRETORIO_MODELO = DIRETORIO_MODELO_LINGUAGEM + "/" + NOME_DIRETORIO_BASE_MODELO
        
        # Recupera o nome do arquivo do modelo da url.
        NOME_ARQUIVO = URL_MODELO.split('/')[-1]

        # Nome do arquivo do vocabulário.
        ARQUIVO_VOCAB = 'vocab.txt'
        
        # Caminho do arquivo na url.
        CAMINHO_ARQUIVO = URL_MODELO[0:len(URL_MODELO)-len(NOME_ARQUIVO)]

        # Verifica se o diretório de descompactação existe no diretório corrente
        if os.path.exists(DIRETORIO_MODELO):
            logging.info('Apagando diretório existente do modelo!')
            # Apaga o diretório e os arquivos existentes                     
            shutil.rmtree(DIRETORIO_MODELO)
        
        # Realiza o download do arquivo do modelo        
        downloadArquivo(URL_MODELO, NOME_ARQUIVO)

        # Descompacta o arquivo no diretório de descompactação.                
        arquivoZip = zipfile.ZipFile(NOME_ARQUIVO, "r")
        arquivoZip.extractall(DIRETORIO_MODELO)

        # Baixa o arquivo do vocabulário.
        # O vocabulário não está no arquivo compactado acima, mesma url mas arquivo diferente.
        URL_MODELO_VOCAB = CAMINHO_ARQUIVO + ARQUIVO_VOCAB
        # Coloca o arquivo do vocabulário no diretório do modelo.        
        downloadArquivo(URL_MODELO_VOCAB, DIRETORIO_MODELO + "/" + ARQUIVO_VOCAB)
        
        # Apaga o arquivo compactado
        os.remove(NOME_ARQUIVO)

        logging.info("Diretório {} do modelo BERT pronta!".format(DIRETORIO_MODELO))

    else:
        DIRETORIO_MODELO = MODELO
        logging.info("Variável URL_MODELO não setada!")

    return DIRETORIO_MODELO

# ============================
def copiaModeloAjustado(model_args):
    ''' 
    Copia o modelo ajustado BERT do GoogleDrive para o projeto.
    
    Parâmetros:
    `model_args` - Objeto com os argumentos do modelo.
    
    Retorno:
    `DIRETORIO_LOCAL_MODELO_AJUSTADO` - Diretório de download ajustado do modelo.
    ''' 
    # Verifica o nome do modelo BERT a ser utilizado
    MODELO_BERT = getNomeModeloLinguagem(model_args)

    # Verifica o tamanho do modelo(default large)
    TAMANHO_BERT = getTamanhoModeloLinguagem(model_args)

    # Verifica se existe o diretório base do cohebert e retorna o nome do diretório
    DIRETORIO_MODELO_LINGUAGEM = verificaDiretorioModeloLinguagem()

    # Diretório local de salvamento do modelo.
    DIRETORIO_LOCAL_MODELO_AJUSTADO = DIRETORIO_MODELO_LINGUAGEM + "/modelo_ajustado/"

    # Diretório remoto de salvamento do modelo no google drive.
    DIRETORIO_REMOTO_MODELO_AJUSTADO = "/content/drive/MyDrive/Colab Notebooks/Data/CSTNEWS/validacao_classificacao/holdout/modelo/" + MODELO_BERT + TAMANHO_BERT

    # Copia o arquivo do modelo para o diretório no Google Drive.
    shutil.copytree(DIRETORIO_REMOTO_MODELO_AJUSTADO, DIRETORIO_LOCAL_MODELO_AJUSTADO) 
   
    logging.info("Modelo BERT ajustado copiado!")

    return DIRETORIO_LOCAL_MODELO_AJUSTADO

# ============================
def verificaModelo(model_args):
    ''' 
    Verifica de onde utilizar o modelo.
    
    Parâmetros:
    `model_args` - Objeto com os argumentos do modelo.
    
    Retorno:
    `DIRETORIO_MODELO` - Diretório de download do modelo.
    ''' 

    DIRETORIO_MODELO = None
    
    if model_args.usar_mcl_ajustado == True:        
        # Diretório do modelo
        DIRETORIO_MODELO = copiaModeloAjustado()
        
        logging.info("Usando modelo BERT ajustado.")
        
    else:
        DIRETORIO_MODELO = downloadModeloPretreinado(model_args)
        logging.info("Usando modelo BERT pré-treinado.")        
        
    return DIRETORIO_MODELO

# ============================
def carregaTokenizadorModeloPretreinado(DIRETORIO_MODELO, model_args):
    ''' 
    Carrega o tokenizador do DIRETORIO_MODELO.
    O tokenizador utiliza WordPiece.
    Carregando o tokenizador do diretório './modelo/' do diretório padrão se variável `DIRETORIO_MODELO` setada.
    Caso contrário carrega da comunidade
    Por default(`do_lower_case=True`) todas as letras são colocadas para minúsculas. Para ignorar a conversão para minúsculo use o parâmetro `do_lower_case=False`. Esta opção também considera as letras acentuadas(ãçéí...), que são necessárias a língua portuguesa.
    O parâmetro `do_lower_case` interfere na quantidade tokens a ser gerado a partir de um texto. Quando igual a `False` reduz a quantidade de tokens gerados.
    
    Parâmetros:
    `DIRETORIO_MODELO` - Diretório a ser utilizado pelo modelo BERT.           
    `model_args` - Objeto com os argumentos do modelo.       
    
    Retorno:
    `tokenizer` - Tokenizador BERT.
    ''' 

    tokenizer = None
    
    # Se a variável DIRETORIO_MODELO foi setada.
    if DIRETORIO_MODELO:
        # Carregando o Tokenizador.
        logging.info("Carregando o tokenizador BERT do diretório {}.".format(DIRETORIO_MODELO))

        tokenizer = BertTokenizer.from_pretrained(DIRETORIO_MODELO, do_lower_case=model_args.do_lower_case)

    else:
        # Carregando o Tokenizador da comunidade.
        logging.info("Carregando o tokenizador BERT da comunidade.")

        tokenizer = BertTokenizer.from_pretrained(model_args.pretrained_model_name_or_path, do_lower_case=model_args.do_lower_case)

    return tokenizer

# ============================
def carregaModelo(DIRETORIO_MODELO, model_args):
    ''' 
    Carrega o modelo e retorna o modelo.
    
    Parâmetros:
    `DIRETORIO_MODELO` - Diretório a ser utilizado pelo modelo BERT.           
    `model_args` - Objeto com os argumentos do modelo.   
    
    Retorno:
    `model` - Um objeto do modelo de linguagem carregado.
    ''' 

    # Variável para setar o arquivo.
    URL_MODELO = None

    if 'http' in model_args.pretrained_model_name_or_path:
        URL_MODELO = model_args.pretrained_model_name_or_path

    # Se a variável URL_MODELO foi setada
    if URL_MODELO:        
        # Carregando o Modelo de Linguagem
        logging.info("Carregando o modelo de linguagem do diretório {}.".format(DIRETORIO_MODELO))

        model = BertModel.from_pretrained(DIRETORIO_MODELO,
                                          output_attentions=model_args.output_attentions,
                                          output_hidden_states=model_args.output_hidden_states)
        
    else:
        # Carregando o Modelo de Linguagem da comunidade
        logging.info("Carregando o modelo de linguagem da comunidade {}.".format(model_args.pretrained_model_name_or_path))

        model = BertModel.from_pretrained(model_args.pretrained_model_name_or_path,
                                          output_attentions=model_args.output_attentions,
                                          output_hidden_states=model_args.output_hidden_states)

    return model



# ============================
def carregaModeloLinguagem(model_args):
    ''' 
    Carrega o BERT para cálculo de medida e retorna o modelo e o tokenizador.
    O tipo do model retornado é BertModel.
    
    Parâmetros:
    `model_args` - Objeto com os argumentos do modelo.               

    Retorno:    
    `model` - Um objeto do modelo BERT carregado.       
    `tokenizer` - Um objeto tokenizador BERT carregado.       
    ''' 
            
    # Verifica a origem do modelo
    DIRETORIO_MODELO = verificaModelo(model_args)
    
    # Variável para conter o modelo
    model = None
    
    # Carrega o model
    model = carregaModelo(DIRETORIO_MODELO, model_args)
    
    transformer_model = Transformer(model_args.pretrained_model_name_or_path)
       
    # Carrega o tokenizador. 
    # O tokenizador é o mesmo para o classificador e medidor.
    #tokenizer = carregaTokenizadorModeloPretreinado(DIRETORIO_MODELO, model_args)
    
    model = transformer_model.get_model()
    tokenizer = transformer_model.get_tokenizer()
    
    return model, tokenizer
