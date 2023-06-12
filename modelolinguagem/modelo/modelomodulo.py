# Import das bibliotecas.
import logging  # Biblioteca de logging
import zipfile # Biblioteca para descompactar
import os # Biblioteca de manipulação de arquivos
import shutil # iblioteca de manipulação arquivos de alto nível
import torch # Biblioteca para manipular os tensores

# Bibliotecas Transformer
from transformers import BertModel # Importando as bibliotecas do Modelo BERT.
from transformers import BertForSequenceClassification # Importando as bibliotecas do Modelo BERT.
from transformers import BertTokenizer # Importando as bibliotecas do tokenizador BERT.
from transformers import AdamW # Biblioteca do otimizador
from transformers import get_linear_schedule_with_warmup # Biblioteca do agendador

# Import de bibliotecas próprias.
from util.utilmodulo import *
from util.utiltempo import *
from util.utilarquivo import *

from bert.bertarguments import ModeloArgumentosClassificacao
from bert.bertarguments import ModeloArgumentos

# ============================
def carregaOtimizador(training_args, model):
    '''
    Esta função carrega o otimizador utilizado no agendador de aprendizado.
    
    Parâmetros:
    `training_args` - Objeto com os argumentos do treinamento. 
    `model` - Um objeto do modelo BERT. 
    '''
    
    # Nota: AdamW é uma classe da biblioteca huggingface (ao contrário de pytorch).
    # Eu acredito que o 'W' significa 'Correção de redução de peso "
    otimizador = AdamW(model.parameters(),
                       lr=training_args.learning_rate, # (ou alfa) A taxa de aprendizado a ser usada. - default é 3e-5
                       # betas = (0.9, 0.999), # (beta1, beta2) - default é (0.9, 0.999)
                       # beta1 é taxa de decaimento exponencial para as estimativas do primeiro momento. 
                       # beta2 é taxa de decaimento exponencial para as estimativas do segundo momento. Este valor deve ser definido próximo a 1,0 em problemas com gradiente esparso (por exemplo, PNL e problemas de visão de computacional)
                       # eps = 1e-6, #  É um número muito pequeno para evitar qualquer divisão por zero na implementação - default é 1e-6.
                       # weight_decay = 0.0, # Correção de redução de peso. - default é 0.0
                       # A redução da taxa de aprendizagem também pode ser usada com Adam. A taxa de decaimento é atualizada a cada época para a demonstração da regressão logística.
                       # correct_bias = True #  Se não deve corrigir o viés(bias) no Adam mudar para False.- default é True
                       )
    
    logging.info("Otimizador carregado.")
  
    return otimizador
    
# ============================
def carregaAgendador(training_args, otimizador, tamanho_conjunto):
    '''
    Esta função carrega o agendador com um taxa de aprendizado que diminua linearmente até 0.
    
    Parâmetros:
    `training_args` - Objeto com os argumentos do treinamento. 
    `otimizador` - Objeto do otmizador do modelo. 
    `tamanho_conjunto` - Tamanho do conjunto de dados de treino. 
    '''

    # O número total de etapas de ajuste fino é [número de lotes] x [número de épocas].
    # (Observe que este não é o mesmo que o número de amostras de ajuste fino).
    total_etapas = tamanho_conjunto * training_args.num_train_epochs
    
    #Cria o agendador de taxa de aprendizagem.
    agendador = get_linear_schedule_with_warmup(otimizador, # O otimizador para o qual agendar a taxa de aprendizado.
                                                num_warmup_steps=0, # O número de etapas para a fase de aquecimento. Valor default value em run_glue.py
                                                num_training_steps=total_etapas) # O número total de etapas de treinamento.

    logging.info("Total de etapas do agendador: {}.".format(total_etapas))

    return agendador  

# ============================
def obter_intervalo_atualizacao(total_iteracoes, numero_atualizacoes):
    '''
    Esta função tentará escolher um intervalo de atualização de progresso inteligente com base na magnitude das iterações totais.

    Parâmetros:
    `total_iteracoes` - O número de iterações no loop for.
    `numero_atualizacoes` - Quantas vezes queremos ver uma atualização sobre o curso do loop for.
    '''
    
    # Divide o total de iterações pelo número desejado de atualizações. Provavelmente
    # este será um número ruim.
    intervalo_exato = total_iteracoes / numero_atualizacoes

    # A função `arredondar` tem a capacidade de arredondar um número, por exemplo, o
    # milésimo mais próximo: round (intervalo_exato, -3)
    #
    # Para determinar a magnitude para arredondar, encontre a magnitude do total, e então vá uma magnitude abaixo disso.
    
    # Obtenha a ordem de magnitude do total.
    ordem_magnitude = len(str(total_iteracoes)) - 1
    
    # Nosso intervalo de atualização deve ser arredondado para uma ordem de magnitude menor.
    magnitude_arrendonda = ordem_magnitude - 1

    # Arredonde para baixo e transforme em um int.
    intervalo_atualizacao = int(round(intervalo_exato, -magnitude_arrendonda))

    # Não permite que o intervalo seja zero!
    if intervalo_atualizacao == 0:
        intervalo_atualizacao = 1

    return intervalo_atualizacao

# ============================
def cria_lotes_inteligentes(model_args, tokenizer, documentos, classes, documentoids, tamanho_lote):
    '''
    Esta função combina todos os passos para preparar os lotes inteligentes(smartbatch).
    
    Parâmetros:
    `model_args` - Objeto com os argumentos do modelo.       
    `tokenizer` - Tokenizador BERT dos documentos.
    `documentos` - Lista dos documentos a serem colocados nos lotes inteligentes.
    `classes` - Lista das classes dos documentos a serem colocados nos lotes inteligentes.
    `documentosis` - Lista dos ids dos documentos a serem colocados nos lotes inteligentes.
    `tamanho_lote` - Tamanho do lotes inteligente.
    
    Retorno:
    `py_input_ids` - Tensores Pytorh com os ids dos tokens.
    `py_attention_masks`- Tensores Pytorh com a máscara de atenção. 
    `py_labels` - Tensores Pytorh com as classes dos documentos.
    `list_documentoids` Ids dos documentos.
    '''
    #print('Criando Lotes Inteligentes de {:,} amostras com tamanho de lote {:,}...\n'.format(len(documentos), tamanho_lote))
    
    # ============================
    #   Tokenização & Truncamento
    # ============================

    input_ids_completos = []
    
    # Tokeniza todas as amostras de treinamento
    logging.info("Tokenizando {:,} amostra.".format(len(classes)))
    
    # Escolha o intervalo que o progresso será atualizado.
    #intervalo_atualizacao = obter_intervalo_atualizacao(total_iteracoes=len(classes), numero_atualizacoes=10)
    
    # Para cada amostra de treinamento.
    for documento in documentos:
        
        # Relatório de progresso
        #if ((len(input_ids_completos) % intervalo_atualizacao) == 0):
            #print('  Tokenizado {:,} amostras.'.format(len(input_ids_completos)))

        # Tokeniza a amostra.
        input_ids = tokenizer.encode(text=documento, # Documento a ser codificado.
                                     add_special_tokens=True, # Adiciona os ttokens especiais.
                                     max_length=model_args.max_seq_len, # Tamanho do truncamento.
                                     truncation=True, # Faz o truncamento.
                                     padding=False) # Não preenche.
                
        # Adicione o resultado tokenizado à nossa lista.
        input_ids_completos.append(input_ids)
        
    logging.info("Tokenização concluída.")
    logging.info("{:>10,} amostras.".format(len(input_ids_completos)))

    # =========================
    #      Seleciona os Lotes
    # =========================    
    
    # Classifique as duas listas pelo comprimento da sequência de entrada.
    amostras = sorted(zip(input_ids_completos, classes, documentoids), key=lambda x: len(x[0]))

    logging.info("{:>10,} amostras após classificação.".format(len(amostras)))

    import random

    # Lista de lotes que iremos construir.
    batch_ordered_documentos = []
    batch_ordered_classes = []
    batch_ordered_documentoids = []

    logging.info("Criando lotes de tamanho {:}.".format(tamanho_lote))

    # Escolha um intervalo no qual imprimir atualizações de progresso.
    intervalo_atualizacao = obter_intervalo_atualizacao(total_iteracoes=len(amostras), numero_atualizacoes=10)
        
    # Faça um loop em todas as amostras de entrada.
    while len(amostras) > 0:
        
        # Mostra o progresso.
        #if ((len(batch_ordered_documentos) % intervalo_atualizacao) == 0 \
        #    and not len(batch_ordered_documentos) == 0):
        #    print('  Selecionado {:,} lotes.'.format(len(batch_ordered_documentos)))
        
        # `to_take` é o tamanho real do nosso lote. Será `tamanho_lote` até
        # chegamos ao último lote, que pode ser menor.
        to_take = min(tamanho_lote, len(amostras))
        
        # Escolha um índice aleatório na lista de amostras restantes para começar o nosso lote.
        select = random.randint(0, len(amostras) - to_take)

        # Selecione um lote contíguo de amostras começando em `select`.
        #logging.info("Selecionando lote de {:} a {:}".format(select, select+to_take))
        batch = amostras[select:(select + to_take)]

        #logging.info("Tamanho do lote: {}.".format(len(batch)))
        
        # Cada amostra é uma tupla --divida para criar uma lista separada de
        # sequências e uma lista de rótulos para este lote.
        batch_ordered_documentos.append([s[0] for s in batch])
        batch_ordered_classes.append([s[1] for s in batch])
        batch_ordered_documentoids.append([s[2] for s in batch])
        
        # Remova a amostra da lista
        del amostras[select:select + to_take]

    logging.info("Lote criado - Selecionado {:,} lotes.".format(len(batch_ordered_documentos)))

    # =========================
    #        Adicionando o preenchimento
    # =========================    

    #print('Preenchendo sequências dentro de cada lote...')

    py_input_ids = []
    py_attention_masks = []
    py_labels = []
    list_documentoids = []

    # Para cada lote.
    for (batch_input_ids, batch_labels, batch_documentoids) in zip(batch_ordered_documentos, batch_ordered_classes, batch_ordered_documentoids):

        # Nova versão do lote, desta vez com sequências preenchidas
        # e agora com as máscaras de atenção definidas.
        batch_padded_input_ids = []
        batch_attention_masks = []
                
        # Primeiro, encontre a amostra mais longa do lote.
        # Observe que as sequências atualmente incluem os tokens especiais!
        max_size = max([len(input) for input in batch_input_ids])
        
        # Para cada entrada neste lote.
        for input in batch_input_ids:
                        
            # Quantos tokens pad precisam ser adicionados
            num_pads = max_size - len(input)

            # Adiciona `num_pads` do pad token(tokenizer.pad_token_id) até o final da sequência.
            padded_input = input + [tokenizer.pad_token_id] * num_pads

            # Define a máscara de atenção --é apenas um `1` para cada token real
            # e um `0` para cada token de preenchimento(pad).
            attention_mask = [1] * len(input) + [0] * num_pads
            
            # Adiciona o resultado preenchido ao lote.
            batch_padded_input_ids.append(padded_input)
            batch_attention_masks.append(attention_mask)
        
        # Nosso lote foi preenchido, portanto, precisamos salvar este lote atualizado.
        # Também precisamos que as entradas sejam tensores PyTorch, então faremos isso aqui.
        py_input_ids.append(torch.tensor(batch_padded_input_ids))
        py_attention_masks.append(torch.tensor(batch_attention_masks))
        py_labels.append(torch.tensor(batch_labels))
        list_documentoids.append(batch_documentoids)
    
    # Retorna o conjunto de dados em lotes inteligentes.
    return (py_input_ids, py_attention_masks, py_labels, list_documentoids)

# ============================
def getDeviceGPU():
    '''
    Retorna um dispositivo de GPU se disponível ou CPU.
    
    Retorno:
    `device` - Um device de GPU ou CPU.       
    '''
        
    # Se existe GPU disponível.
    if torch.cuda.is_available():
        
        # Diz ao PyTorch para usar GPU.    
        device = torch.device("cuda")
        
        logging.info("Existem {} GPU(s) disponíveis.".format(torch.cuda.device_count()))
        logging.info("Iremos usar a GPU: {}.".format(torch.cuda.get_device_name(0)))

    # Se não.
    else:        
        logging.info("Sem GPU disponível, usando CPU.")
        device = torch.device("cpu")
        
    return device

# ============================
def conectaGPU(model, device):
    '''
    Conecta um modelo BERT a GPU.

    Parâmetros:
    `model` - Um modelo BERT carregado.       
    `device` - Um device de GPU.     
    
    Retorno:
    `model` - Um objeto model BERT conectado a GPU.     
    '''
    # Associa a GPU ao modelo.
    model.to(device)

    # Se existe GPU disponível.
    if torch.cuda.is_available():    
        # Diga ao pytorch para rodar este modelo na GPU.
        logging.info("Pytorch rodando o modelo na GPU.")
        model.cuda()
        
    else:
        logging.info("Pytorch rodando sem GPU.")

    return model

# ============================
def getNomeModeloBERT(model_args):
    '''    
    Recupera uma string com uma descrição do modelo BERT para nomes de arquivos e diretórios.
    
    Parâmetros:
    `model_args` - Objeto com os argumentos do modelo.       
    
    Retorno:
    `MODELO_BERT` - Nome do modelo BERT.
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
def getTamanhoBERT(model_args):
    '''    
    Recupera uma string com o tamanho(dimensão) do modelo BERT para nomes de arquivos e diretórios.
    
    Parâmetros:
    `model_args` - Objeto com os argumentos do modelo.       
    
    Retorno:
    `TAMANHO_BERT` - Nome do tamanho do modelo BERT.
    '''
    
    # Verifica o tamanho do modelo(default large)
    TAMANHO_BERT = "_large"
    
    if 'base' in model_args.pretrained_model_name_or_path:
        TAMANHO_BERT = "_base"
        
    return TAMANHO_BERT  

# ============================
def downloadModeloPretreinado(model_args):
    ''' 
    Realiza o download do modelo BERT(MODELO) e retorna o diretório onde o modelo BERT(MODELO) foi descompactado.
    
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
    MODELO_BERT = getNomeModeloBERT(model_args)

    # Verifica o tamanho do modelo(default large)
    TAMANHO_BERT = getTamanhoBERT(model_args)

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
def carregaModeloMedida(DIRETORIO_MODELO, model_args):
    ''' 
    Carrega o modelo e retorna o modelo.
    
    Parâmetros:
    `DIRETORIO_MODELO` - Diretório a ser utilizado pelo modelo BERT.           
    `model_args` - Objeto com os argumentos do modelo.   
    
    Retorno:
    `model` - Um objeto do modelo BERT carregado.
    ''' 

    # Variável para setar o arquivo.
    URL_MODELO = None

    if 'http' in model_args.pretrained_model_name_or_path:
        URL_MODELO = model_args.pretrained_model_name_or_path

    # Se a variável URL_MODELO foi setada
    if URL_MODELO:        
        # Carregando o Modelo BERT
        logging.info("Carregando o modelo BERT do diretório {} para cálculo de medidas.".format(DIRETORIO_MODELO))

        model = BertModel.from_pretrained(DIRETORIO_MODELO,
                                          output_attentions=model_args.output_attentions,
                                          output_hidden_states=model_args.output_hidden_states)
        
    else:
        # Carregando o Modelo BERT da comunidade
        logging.info("Carregando o modelo BERT da comunidade {} para cálculo de medidas.".format(model_args.pretrained_model_name_or_path))

        model = BertModel.from_pretrained(model_args.pretrained_model_name_or_path,
                                          output_attentions=model_args.output_attentions,
                                          output_hidden_states=model_args.output_hidden_states)

    return model

# ============================
def carregaModeloClassifica(DIRETORIO_MODELO, model_args):
    ''' 
    Carrega o modelo e retorna o modelo.
    
    Parâmetros:
    `DIRETORIO_MODELO` - Diretório a ser utilizado pelo modelo BERT.           
    `model_args` - Objeto com os argumentos do modelo.
    
    Retorno:
    `model` - Um objeto do modelo BERT carregado.
    ''' 

    # Variável para setar o arquivo.
    URL_MODELO = None

    if 'http' in model_args.pretrained_model_name_or_path:
        URL_MODELO = model_args.pretrained_model_name_or_path

    # Se a variável URL_MODELO foi setada
    if URL_MODELO:
        # Carregando o Modelo BERT
        logging.info("Carregando o modelo BERT do diretório {} para classificação.".format(DIRETORIO_MODELO))

        model = BertForSequenceClassification.from_pretrained(DIRETORIO_MODELO, 
                                                              num_labels=model_args.num_labels,
                                                              output_attentions=model_args.output_attentions,
                                                              output_hidden_states=model_args.output_hidden_states)
            
    else:
        # Carregando o Modelo BERT da comunidade
        logging.info("Carregando o modelo BERT da comunidade {} para classificação.".format(model_args.pretrained_model_name_or_path))

        model = BertForSequenceClassification.from_pretrained(model_args.pretrained_model_name_or_path,
                                                              num_labels=model_args.num_labels,
                                                              output_attentions=model_args.output_attentions,
                                                              output_hidden_states=model_args.output_hidden_states)
    return model

# ============================
def carregaBERT(model_args):
    ''' 
    Carrega o BERT para cálculo de medida ou classificação e retorna o modelo e o tokenizador.
    O tipo do model retornado pode ser BertModel ou BertForSequenceClassification, depende do tipo de model_args.
    
    Parâmetros:
    `model_args` - Objeto com os argumentos do modelo.       
        - Se model_args = ModeloArgumentosClassificacao deve ser carregado o BERT para classificação(BertForSequenceClassification).
        - Se model_args = ModeloArgumentos deve ser carregado o BERT para cálculo de medida(BertModel).

    Retorno:    
    `model` - Um objeto do modelo BERT carregado.       
    `tokenizer` - Um objeto tokenizador BERT carregado.       
    ''' 
            
    # Verifica a origem do modelo
    DIRETORIO_MODELO = verificaModelo(model_args)
    
    # Variável para conter o modelo
    model = None
    
    # Verifica o tipo do modelo em model_args    
    if type(model_args) == ModeloArgumentos:
        # Carrega o modelo para cálculo da medida
        model = carregaModeloMedida(DIRETORIO_MODELO, model_args)
        
    else:
        # Carrega o modelo para classificação
        model = carregaModeloClassifica(DIRETORIO_MODELO, model_args)
        
        # Recupera o dispotivo da GPU 
        device = getDeviceGPU()
    
        # Conecta o modelo a GPU
        model = conectaGPU(model, device)
       
    # Carrega o tokenizador. 
    # O tokenizador é o mesmo para o classificador e medidor.
    tokenizer = carregaTokenizadorModeloPretreinado(DIRETORIO_MODELO, model_args)
    
    return model, tokenizer
