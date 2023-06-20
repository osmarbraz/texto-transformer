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

# Biblioteca próprias
from textotransformer.pln.pln import PLN
from textotransformer.mensurador.mensurador import Mensurador
from textotransformer.modelo.transformer import Transformer
from textotransformer.modelo.modeloarguments import ModeloArgumentos
from textotransformer.modelo.modeloenum import LISTATIPOCAMADA_NOME, EstrategiasPooling, listaTipoCamadas
from textotransformer.modelo.modeloenum import EstrategiasPooling
from textotransformer.mensurador.mensuradorenum import PalavrasRelevantes 
from textotransformer.util.utiltexto import encontrarIndiceSubLista

logger = logging.getLogger(__name__)

# Definição dos parâmetros do Modelo para os cálculos das Medidas
modelo_argumentos = ModeloArgumentos(
        max_seq_len=512,
        pretrained_model_name_or_path="neuralmind/bert-base-portuguese-cased", # Nome do modelo de linguagem pré-treinado Transformer
        modelo_spacy="pt_core_news_lg", # Nome do modelo de linguagem da ferramenta de PLN
        do_lower_case=False,            # default True
        output_attentions=False,        # default False
        output_hidden_states=True,      # default False  /Retornar os embeddings das camadas ocultas  
        camadas_embeddings = 2,         # 0-Primeira/1-Penúltima/2-Ùltima/3-Soma 4 últimas/4-Concat 4 últiamas/5-Todas
        estrategia_pooling=0,           # 0 - MEAN estratégia média / 1 - MAX  estratégia maior
        palavra_relevante=0             # 0 - Considera todas as palavras das sentenças / 1 - Desconsidera as stopwords / 2 - Considera somente as palavras substantivas
        )

class TextoTransformer:
    
    ''' 
    Carrega e cria um objeto da classe TextoTransformer para manipular um modelo de linguagem baseado e transformer.
    Manipula embeddings de tokens, palavras, sentenças e textos.
     
    Parâmetros:
    `pretrained_model_name_or_path` - Se for um caminho de arquivo no disco, carrega o modelo a partir desse caminho. Se não for um caminho, ele primeiro faz o download do repositório de modelos do Huggingface com esse nome. Valor default: 'neuralmind/bert-base-portuguese-cased'.                  
    `modelo_spacy` - Nome do modelo a ser instalado e carregado pela ferramenta de pln spaCy. Valor default 'pt_core_news_lg'.                       
    `camadas_embeddings` - Especifica de qual camada ou camadas será recuperado os embeddings do transformer. Valor defaul '2'. Valores possíveis: 0-Primeira/1-Penúltima/2-Ùltima/3-Soma 4 últimas/4-Concat 4 últiamas/5-Todas.       
    `device` - Dispositivo (como 'cuda' / 'cpu') que deve ser usado para computação. Se none, verifica se uma GPU pode ser usada.
    ''' 
    
    # Construtor da classe
    def __init__(self, pretrained_model_name_or_path="neuralmind/bert-base-portuguese-cased", 
                       modelo_spacy="pt_core_news_lg",
                       camadas_embeddings: int = 2,
                       device=None):
                       
        # Parâmetro recebido para o modelo de linguagem
        modelo_argumentos.pretrained_model_name_or_path = pretrained_model_name_or_path
               
        # Parâmetro recebido para o modelo da ferramenta de pln
        modelo_argumentos.modelo_spacy = modelo_spacy
                
        # Carrega o modelo de linguagem da classe transformador
        self.transformer = Transformer(modelo_args=modelo_argumentos)
    
        # Recupera o modelo.
        self.model = self.transformer.get_auto_model()
    
        # Recupera o tokenizador.     
        self.tokenizer = self.transformer.get_tokenizer()
        
        # Especifica de qual camada utilizar os embeddings        
        logger.info("Utilizando embeddings do modelo da {} camada(s).".format(listaTipoCamadas[modelo_argumentos.camadas_embeddings][LISTATIPOCAMADA_NOME]))
                    
        # Especifica camadas para recuperar os embeddings
        modelo_argumentos.camadas_embeddings = camadas_embeddings
      
        # Carrega o spaCy
        self.pln = PLN(modelo_args=modelo_argumentos)
                        
        # Constroi um mensurador
        self.mensurador = Mensurador(modelo_args=modelo_argumentos, 
                                     transformer=self.transformer, 
                                     pln=self.pln)        
    
        # Verifica se é possível usar GPU
        if device is None:
            if torch.cuda.is_available():    
                device = "cuda"
                logging.info("Existem {} GPU(s) disponíveis.".format(torch.cuda.device_count()))
                logging.info("Iremos usar a GPU: {}.".format(torch.cuda.get_device_name(0)))

            else:                
                device = "cpu"
                logging.info("Sem GPU disponível, usando CPU.")
            
            # Diz ao PyTorch para usar o dispositvo (GPU ou CPU)
            self._target_device = torch.device(device)
        else:
            # Usa o device informado
            self._target_device = torch.device(device)

        # Mensagem de carregamento da classe
        logger.info("Classe TextoTransformer carregada: {}.".format(modelo_argumentos))
    
    # ============================
    def _defineEstrategiaPooling(self, estrategiaPooling):
        ''' 
        Define a estratégia de pooling para os parâmetros do modelo.

        Parâmetros:
        `estrategiaPooling` - Um número de 0 a 1 com a estratégia de pooling das camadas do modelo contextualizado. Valor defaul '0'. Valores possíveis: 0 - MEAN estratégia média / 1 - MAX  estratégia maior.
        ''' 
        
        if estrategiaPooling == EstrategiasPooling.MEAN.name:
            modelo_argumentos.estrategia_pooling = EstrategiasPooling.MEAN.value
        else:
            if estrategiaPooling == EstrategiasPooling.MAX.name:
                modelo_argumentos.estrategia_pooling = EstrategiasPooling.MAX.value            
            else:
                logger.info("Não foi especificado uma estratégia de pooling válida.") 
    
    # ============================
    def _definePalavraRelevante(self, palavraRelevante=0):
        ''' 
        Define a estratégia de palavra relavante para os parâmetros do modelo.
        
        Parâmetros:        
        `palavraRelevante` - Um número de 0 a 2 que indica a estratégia de relevância das palavras do texto. Valor defaul '0'. Valores possíveis: 0 - Considera todas as palavras das sentenças / 1 - Desconsidera as stopwords / 2 - Considera somente as palavras substantivas.
        ''' 
        
        if palavraRelevante == PalavrasRelevantes.CLEAN.name:
            modelo_argumentos.palavra_relevante = PalavrasRelevantes.CLEAN.value
            
        else:
            if palavraRelevante == PalavrasRelevantes.NOUN.name:
                modelo_argumentos.palavra_relevante = PalavrasRelevantes.NOUN.value
                
            else:
                if palavraRelevante == PalavrasRelevantes.ALL.name:
                    modelo_argumentos.palavra_relevante = PalavrasRelevantes.ALL.value                    
                else:
                    logger.info("Não foi especificado uma estratégia de relevância de palavras do texto válida.") 

    # ============================
    def getMedidasTexto(self, texto, estrategiaPooling='MEAN', palavraRelevante='ALL'):
        ''' 
        Retorna as medidas de (in)coerência Ccos, Ceuc, Cman do texto.
        
        Parâmetros:
        `texto` - Um texto a ser medido.           
        `estrategiaPooling` - Estratégia de pooling das camadas do BERT.
        `palavraRelevante` - Estratégia de relevância das palavras do texto.            
        
        Retorno:
        `Ccos` - Medida de coerência Ccos do do texto.            
        `Ceuc` - Medida de incoerência Ceuc do do texto.            
        `Cman` - Medida de incoerência Cman do do texto.            
        ''' 

        self._defineEstrategiaPooling(estrategiaPooling)
        self._definePalavraRelevante(palavraRelevante)
        
        self.Ccos, self.Ceuc, self.Cman = self.mensurador.getMedidasComparacaoTexto(texto, 
                                                                                    camada=modelo_argumentos.camadas_embeddings, 
                                                                                    tipoTexto='o')
          
        return self.Ccos, self.Ceuc, self.Cman
    
    # ============================
    def getMedidasTextoCosseno(self, 
                               texto, 
                               estrategiaPooling='MEAN', 
                               palavraRelevante='ALL'):
        ''' 
        Retorna a medida de coerência do texto utilizando a medida de similaridade de cosseno.
        
        Parâmetros:
        `texto` - Um texto a ser medido a coerência.           
        `estrategiaPooling` - Estratégia de pooling das camadas do BERT. 
        `palavraRelevante` - Estratégia de relevância das palavras do texto.            
        
        Retorno:
        `Ccos` - Medida de coerência Ccos do do texto.            
        '''         
        
        self._defineEstrategiaPooling(estrategiaPooling)
        self._definePalavraRelevante(palavraRelevante)
        
        self.Ccos, self.Ceuc, self.Cman = self.mensurador.getMedidasComparacaoTexto(texto, 
                                                                    camada=modelo_argumentos.camadas_embeddings, 
                                                                    tipoTexto='o')
          
        return self.Ccos
    
    # ============================
    def getMedidasTextoEuclediana(self, texto, estrategiaPooling='MEAN', palavraRelevante='ALL'):
        ''' 
        Retorna a medida de incoerência do texto utilizando a medida de distância de Euclidiana.
                 
        Parâmetros:
        `texto` - Um texto a ser medido a coerência.           
        `estrategiaPooling` - Estratégia de pooling das camadas do BERT.
        `palavraRelevante` - Estratégia de relevância das palavras do texto.            
        
        Retorno:
        `Ceuc` - Medida de incoerência Ceuc do do texto.            
        ''' 
        
        self._defineEstrategiaPooling(estrategiaPooling)
        self._definePalavraRelevante(palavraRelevante)

        self.Ccos, self.Ceuc, self.Cman = self.mensurador.getMedidasComparacaoTexto(texto,
                                                                    camada=modelo_argumentos.camadas_embeddings, 
                                                                    tipoTexto='o')
          
        return self.Ceuc        
       
    # ============================
    def getMedidasTextoManhattan(self, texto, estrategiaPooling='MEAN', palavraRelevante='ALL'):
        ''' 
        Retorna a medida de incoerência do texto utilizando a medida de distância de Manhattan.
                 
        Parâmetros:
        `texto` - Um texto a ser medido a coerência.           
        `estrategiaPooling` - Estratégia de pooling das camadas do BERT.
        `palavraRelevante` - Estratégia de relevância das palavras do texto.            
        
        Retorno:
        `Cman` - Medida de incoerência Cman do do texto.            
        ''' 
        
        self._defineEstrategiaPooling(estrategiaPooling)
        self._definePalavraRelevante(palavraRelevante)
        
        self.Ccos, self.Ceuc, self.Cman = self.mensurador.getMedidasComparacaoTexto(texto, 
                                                                    camada=modelo_argumentos.camadas_embeddings, 
                                                                    tipoTexto='o')
          
        return self.Cman                
    
    # ============================
    def tokenize(self, texto):
        '''
        D Tokeniza um texto para submeter ao modelo de linguagem. 
        Retorna um dicionário listas de mesmo tamanho para garantir o processamento em lote.
        Use a quantidade de tokens para saber até onde deve ser recuperado em uma lista de saída.
        Ou use attention_mask diferente de 1 para saber que posições devem ser utilizadas na lista.

        Facilita acesso a classe Transformer.    

        :param texto: Texto a ser tokenizado para o modelo de linguagem.
         
        Retorna um dicionário com:
            tokens_texto_mcl uma lista com os textos tokenizados com os tokens especiais.
            input_ids uma lista com os ids dos tokens de entrada mapeados em seus índices do vocabuário.
            token_type_ids uma lista com os tipos dos tokens.
            attention_mask uma lista com os as máscaras de atenção indicando com '1' os tokens  pertencentes à sentença.
        '''
        return self.get_transformer().tokenize(texto)
    
    # ============================
    def getCodificacao(self, texto: Union[str, List[str]],
                    tamanho_lote: int = 32, 
                    mostra_barra_progresso: bool = False,                     
                    tipo_saida: str = 'texto_embedding',
                    convert_to_numpy: bool = True,
                    convert_to_tensor: bool = False,
                    device: str = None,
                    normalize_embeddings: bool = False):

        '''
        Calcula os embeddings de um texto utilizando o modelo de linguagem.
    
        Parâmetros:
         `texto` - Um texto a ser recuperado os embeddings do modelo de linguagem
         `tamanho_lote` - o tamanho do lote usado para o computação
         `mostra_barra_progresso` - Mostra uma barra de progresso ao codificar o texto.
         `tipo_saida` -  Especifica o tipo dos embeddings de saída. Pode ser definido como texto_embedding, sentenca_embedding, word_embedding, token_embeddings para obter embeddings de token do texto. Defina como none, para obter todos os valores de saída
         `convert_to_numpy` - Se verdadeiro, a saída é uma lista de vetores numpy. Caso contrário, é uma lista de tensores pytorch.
         `convert_to_tensor` - Se verdadeiro, você obtém um grande tensor como retorno. Substitui qualquer configuração de convert_to_numpy
         `device` - Qual torch.device usar para o computação.
         `normalize_embeddings` - Se definido como verdadeiro, os vetores retornados terão comprimento 1. Nesse caso, o produto escalar mais rápido (util.dot_score) em vez da similaridade de cosseno pode ser usado.
        
    
        :return::
            Por padrão, uma lista de tensores é retornada. Se convert_to_tensor, um tensor empilhado é retornado. Se convert_to_numpy, uma matriz numpy é retornada.
        '''

        if convert_to_tensor:
            convert_to_numpy = False

        if tipo_saida != 'texto_embedding':
            convert_to_tensor = False
            convert_to_numpy = False

        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'): 
            #Colocar uma texto individual em uma lista com comprimento 1
            texto = [texto]
            entrada_eh_string = True

        if device is None:
            device = self._target_device

        self.to(device)

        # Lista com embeddings de saída
        all_embeddings = []
        # Ordena o texto pelo comprimento decrescente
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in texto])
        # Ordena o texto pelo comprimento decrescente
        textos_ordenados = [texto[idx] for idx in length_sorted_idx]

        for start_index in trange(0, 
                                  len(texto), 
                                  tamanho_lote, 
                                  desc="Lotes", 
                                  disable=not mostra_barra_progresso):
            # Recupera um lote
            lote_textos = textos_ordenados[start_index:start_index+tamanho_lote]

            # Tokeniza o lote
            features = self.get_transformer().tokenize(lote_textos)
            features = self.get_transformer().batch_to_device(features, device)

            # Recupera os embeddings do modelo
            with torch.no_grad():
                out_features = self.get_transformer().forward(features)

                # Retorno embeddings de tokens
                if tipo_saida == 'token_embeddings':
                    embeddings = []
                    for token_emb, attention in zip(out_features[tipo_saida], out_features['attention_mask']):
                        last_mask_id = len(attention)-1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0:last_mask_id+1])
                else:
                    # Retorna toda a saída
                    if tipo_saida is None: 
                        embeddings = []
                        for sent_idx in range(len(out_features['sentence_embedding'])):
                            row =  {name: out_features[name][sent_idx] for name in out_features}
                            embeddings.append(row)
                    else:   
                        #Retorna Texto embeddings
                        embeddings = out_features[tipo_saida]
                        embeddings = embeddings.detach()
                        if normalize_embeddings:
                            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                        if convert_to_numpy:
                            embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        else:
            if convert_to_numpy:
                all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if entrada_eh_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    # ============================
    def getEmbeddings(self, texto):
        '''
        De um texto preparado(tokenizado) ou não, retorna token_embeddings, input_ids, attention_mask, token_type_ids, 
        tokens_texto, texto_original  e all_layer_embeddings em um dicionário.
        
        Facilita acesso ao método "getEmbeddings" da classe Transformer.
    
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings do modelo de linguagem
    
        Retorna um dicionário com:            
            token_embeddings uma lista com os embeddings da última camada
            input_ids uma lista com os textos indexados.            
            attention_mask uma lista com os as máscaras de atenção
            token_type_ids uma lista com os tipos dos tokens.            
            tokens_texto uma lista com os textos tokenizados com os tokens especiais.
            texto_original uma lista com os textos originais.
            all_layer_embeddings uma lista com os embeddings de todas as camadas.
        '''
        return self.get_transformer().getEmbeddings(texto)

    # ============================
    def getEmbeddingsTextosMEAN(self, texto):
        '''
        De um texto preparado(tokenizado) ou não, retorna os embeddings dos textos consolidados dos tokens do texto utilizando estratégia pooling MEAN.
            
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings dos textos consolidados dos tokens com a estratégia MEAN utilizando o modelo de linguagem.
    
        Retorna uma lista com os embeddings dos textos consolidados dos tokens com a estratégia MEAN.
        '''

        return self.getCodificacaoTextos(texto)['texto_embeddings_MEAN']
    
    # ============================
    def getEmbeddingsTextosMAX(self, texto):
        '''
        De um texto preparado(tokenizado) ou não, retorna os embeddings dos textos consolidados dos tokens do texto utilizando estratégia pooling MAX.
            
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings dos textos consolidados dos tokens com a estratégia MAX utilizando o modelo de linguagem.
    
        Retorna uma lista com os embeddings dos textos consolidados dos tokens com a estratégia MAX.
        '''

        return self.getCodificacaoTextos(texto)['texto_embeddings_MAx']    

    # ============================
    def getCodificacaoTextos(self, texto):
        '''        
        De um texto preparado(tokenizado) ou não, retorna a codificação dos textos consolidados dos tokens do textos utilizando estratégia pooling MEAN e MAX.
    
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings dos textos consolidados dos tokens com a estratégia MEAN e MAX utilizando o modelo de linguagem
    
        Retorna uma lista com os embeddings consolidados dos textos utilizando embeddings da última camada do transformer.
        '''

        # Se o texto é uma string, coloca em uma lista de comprimento 1
        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'):             
            texto = [texto]
            entrada_eh_string = True

        # Recupera os embeddings do texto
        texto_embeddings = self.getEmbeddings(texto)

        # Acumula a saída do método
        saida = {}
        saida.update({'texto_original' : [],            # Lista com os textos originais
                      'tokens_texto_mcl' : [],          # Lista com os tokens dos textos originais
                      'texto_embeddings_MEAN': [],      # Lista de lista média dos embeddings dos tokens que da sentença.
                      'texto_embeddings_MAX': [],       # Lista de lista máximo dos embeddings dos tokens que da sentença.
                      #'all_layer_embeddings': []
                     }
        )

        # Percorre os textos da lista.
        for i, texto in enumerate(texto_embeddings['texto_original']):       

            # Recupera os embeddings do texto  
            embeddings_texto = texto_embeddings['token_embeddings'][i][0:len(texto_embeddings['tokens_texto_mcl'][i])]            
           
            # Recupera a lista de tokens do tokenizado pelo MCL sem CLS e SEP
            tokens_texto_mcl = texto_embeddings['tokens_texto_mcl'][i][1:-1]            
            
            # Calcula a média dos embeddings dos tokens das sentenças do texto
            embedding_documento_media = torch.mean(embeddings_texto, dim=0)

            # Calcula a média dos embeddings dos tokens das sentenças do texto
            embedding_documento_maximo, linha = torch.max(embeddings_texto, dim=0)
            
            #Acumula a saída do método 
            #Se é uma string uma lista com comprimento 1
            if entrada_eh_string:
                saida['texto_original'] = texto_embeddings['texto_original'][i]
                saida['tokens_texto_mcl'] =  tokens_texto_mcl
                saida['texto_embeddings_MEAN'] = embedding_documento_media
                saida['texto_embeddings_MAX'] = embedding_documento_maximo
            else:
                saida['texto_original'].append(texto_embeddings['texto_original'][i])
                saida['tokens_texto_mcl'].append(tokens_texto_mcl)
                saida['texto_embeddings_MEAN'].append(embedding_documento_media)
                saida['texto_embeddings_MAX'].append(embedding_documento_maximo)

        return saida
    
    # ============================
    def getEmbeddingsSentencasMEAN(self, texto):    
        '''
        De um texto preparado(tokenizado) ou não, retorna os embeddings das sentenças consolidados dos tokens do texto utilizando estratégia pooling MEAN.
            
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings das sentenças consolidados dos tokens com a estratégia MEAN utilizando o modelo de linguagem.
    
        Retorna uma lista com os embeddings das sentenças consolidados dos tokens com a estratégia MEAN.
        '''

        return self.getCodificacaoSentencas(texto)['sentenca_embeddings_MEAN']

    # ============================
    def getEmbeddingsSentencasMAX(self, texto):    
        '''
        De um texto preparado(tokenizado) ou não, retorna os embeddings das sentenças consolidados dos tokens do texto utilizando estratégia pooling MAX.
            
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings das sentenças consolidados dos tokens com a estratégia MAX utilizando o modelo de linguagem.
    
        Retorna uma lista com os embeddings das sentenças consolidados dos tokens com a estratégia MAX.
        '''

        return self.getCodificacaoSentencas(texto)['sentenca_embeddings_MAX']

    # ============================
    def getCodificacaoSentencas(self, texto):    
        '''        
        De um texto preparado(tokenizado) ou não, retorna a codificação das sentenças consolidados dos tokens do textos utilizando estratégia pooling MEAN e MAX.
    
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings das sentenças consolidados dos tokens com a estratégia MEAN e MAX utilizando o modelo de linguagem
    
        Retorna uma lista com os embeddings consolidados das sentenças utilizando embeddings da última camada do transformer.
        '''

        # Se o texto é uma string, coloca em uma lista de comprimento 1
        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'):             
            texto = [texto]
            entrada_eh_string = True

        # Recupera os embeddings do texto
        texto_embeddings = self.getEmbeddings(texto)

        # Acumula a saída do método
        saida = {}
        saida.update({'texto_original' : [],            # Lista com os textos originais
                      'tokens_texto_mcl' : [],          # Lista com os tokens dos textos originais
                      'sentencas_texto' : [],           # Lista com as sentenças do texto
                      'sentenca_embeddings_MEAN': [],      # Lista de lista média dos embeddings dos tokens que da sentença.
                      'sentenca_embeddings_MAX': [],       # Lista de lista máximo dos embeddings dos tokens que da sentença.
                      #'all_layer_embeddings': []
                     }
        )

        # Percorre os textos da lista.
        for i, texto in enumerate(texto_embeddings['texto_original']):       

            # Recupera os embeddings do texto  
            embeddings_texto = texto_embeddings['token_embeddings'][i][0:len(texto_embeddings['tokens_texto_mcl'][i])]            
           
            # Recupera a lista de tokens do tokenizado pelo MCL sem CLS e SEP
            tokens_texto_mcl = texto_embeddings['tokens_texto_mcl'][i][1:-1]            
            
            # Recupera as sentenças do texto
            lista_sentencas_texto = self.get_pln().getListaSentencasTexto(texto_embeddings['texto_original'][i])
                 
            lista_embeddings_tokens_sentencas_texto_media = []
            lista_embeddings_tokens_sentencas_texto_maximo = []
            
            # Percorre as sentenças do texto
            for j, sentenca in enumerate(lista_sentencas_texto):

                # Tokeniza a sentença
                sentenca_tokenizada =  self.transformer.getTextoTokenizado(sentenca)
                
                # Remove os tokens de início e fim da sentença
                sentenca_tokenizada.remove('[CLS]')
                sentenca_tokenizada.remove('[SEP]')    
                #print(len(sentencaTokenizada))

                # Localiza os índices dos tokens da sentença no texto
                inicio, fim = encontrarIndiceSubLista(tokens_texto_mcl, sentenca_tokenizada)

                # Recupera os embeddings dos tokens da sentença a partir dos embeddings do texto
                embedding_sentenca = embeddings_texto[inicio:fim + 1]

                # Calcula a média dos embeddings dos tokens das sentenças do texto
                embedding_sentenca_media = torch.mean(embedding_sentenca, dim=0)

                # Calcula a média dos embeddings dos tokens das sentenças do texto
                embedding_sentenca_maximo, linha = torch.max(embedding_sentenca, dim=0)

                # Guarda os tokens e os embeddings das sentenças do texto da média e do máximo
                lista_embeddings_tokens_sentencas_texto_media.append(embedding_sentenca_media)
                lista_embeddings_tokens_sentencas_texto_maximo.append(embedding_sentenca_maximo)

            
            #Acumula a saída do método 
            #Se é uma string uma lista com comprimento 1
            if entrada_eh_string:
                saida['texto_original'] = texto_embeddings['texto_original'][i]
                saida['tokens_texto_mcl'] =  tokens_texto_mcl
                saida['sentencas_texto'] = lista_sentencas_texto
                saida['sentenca_embeddings_MEAN'] = lista_embeddings_tokens_sentencas_texto_media
                saida['sentenca_embeddings_MAX'] = lista_embeddings_tokens_sentencas_texto_maximo
            else:
                saida['texto_original'].append(texto_embeddings['texto_original'][i])
                saida['tokens_texto_mcl'].append(tokens_texto_mcl)
                saida['sentencas_texto'].append(lista_sentencas_texto)
                saida['sentenca_embeddings_MEAN'].append(lista_embeddings_tokens_sentencas_texto_media)
                saida['sentenca_embeddings_MAX'].append(lista_embeddings_tokens_sentencas_texto_maximo)

        return saida
   
    # ============================
    def getEmbeddingsPalavrasMEAN(self, 
                                  texto):
        '''
        De um texto preparado(tokenizado) ou não, retorna os embeddings das palavras consolidados dos tokens do texto utilizando estratégia pooling MEAN.
            
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings das palavras consolidados dos tokens com a estratégia MEAN utilizando o modelo de linguagem.
    
        Retorna uma lista com os embeddings das palavras consolidados dos tokens com a estratégia MEAN.
        '''
        saida = self.getCodificacaoPalavras(texto)['embeddings_MEAN']
        
        return saida

    # ============================
    def getEmbeddingsPalavrasMAX(self, 
                                  texto):
        '''
        De um texto preparado(tokenizado) ou não, retorna os embeddings das palavras consolidados dos tokens do texto utilizando estratégia pooling MAX.
            
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings das palavras consolidados dos tokens com a estratégia MAX utilizando o modelo de linguagem.
    
        Retorna uma lista com os embeddings das palavras consolidados dos tokens com a estratégia MAX.
        '''
        saida = self.getCodificacaoPalavras(texto)['embeddings_MAX']
        
        return saida
            
    # ============================
    def getCodificacaoPalavras(self, 
                               texto):
        
        '''
        De um texto preparado(tokenizado) ou não, retorna os embeddings das palavras do texto. 
        Retorna um dicionário 5 listas, os tokens(palavras), as postagging, tokens OOV, e os embeddings dos tokens igualando a quantidade de tokens do spaCy com a tokenização do MCL de acordo com a estratégia. 
        Utiliza duas estratégias para realizar o pooling de tokens que forma uma palavra.
            - Estratégia MEAN para calcular a média dos embeddings dos tokens que formam uma palavra.
            - Estratégia MAX para calcular o valor máximo dos embeddings dos tokens que formam uma palavra.
            
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings das palavras do modelo de linguagem
    
        Retorna um dicionário com:    
            tokens_texto uma lista com os tokens(palavras) realizados pelo método.
            tokens_texto_mcl uma lista com os tokens e tokens especiais realizados pelo mcl.
            tokens_oov_texto_mcl uma lista com os tokens OOV(com ##) do mcl.
            tokens_texto_pln uma lista com os tokens realizados pela ferramenta de pln(spaCy).
            pos_texto_pln uma lista com as postagging dos tokens realizados pela ferramenta de pln(spaCy).            
            embeddings_MEAN uma lista com os embeddings com a estratégia MEAN.
            embeddings_MAX uma lista com os embeddings com a estratégia MAX.
        '''
        
        # Se o texto é uma string, coloca em uma lista de comprimento 1
        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'):             
            texto = [texto]
            entrada_eh_string = True

        # Tokeniza o texto
        texto_embeddings = self.get_transformer().getEmbeddings(texto)

        # Acumula a saída do método
        saida = {}
        saida.update({'texto_original' : [],
                      'tokens_texto': [], 
                      'tokens_texto_mcl' : [],
                      'tokens_oov_texto_mcl': [],                      
                      'tokens_texto_pln' : [],
                      'pos_texto_pln': [],
                      'embeddings_MEAN': [],        
                      'embeddings_MAX': []
                     }
        )

        # Percorre os textos da lista.
        for i, sentenca in enumerate(texto_embeddings['tokens_texto_mcl']):
            # Recupera o texto tokenizado pela ferramenta de pln do texto original
            lista_tokens_texto_pln = self.get_pln().getTokensTexto(texto_embeddings['texto_original'][i])
            # Recupera os embeddings do texto  
            embeddings_texto = texto_embeddings['token_embeddings'][i][0:len(texto_embeddings['tokens_texto_mcl'][i])]            
            # Recupera a lista de tokens do tokenizado pelo MCL sem CLS e SEP
            tokens_texto_mcl = texto_embeddings['tokens_texto_mcl'][i][1:-1]            
            # Concatena os tokens gerandos pela ferramenta de pln
            tokens_texto_concatenado = " ".join(lista_tokens_texto_pln)
            # Recupera os embeddings e tokens de palavra            
            lista_tokens_texto, lista_pos_texto_pln, lista_tokens_oov_texto_mcl, lista_embeddings_MEAN, lista_embeddings_MAX = self.get_transformer().getTokensEmbeddingsPOSTexto(
                                                    embeddings_texto,
                                                    tokens_texto_mcl,
                                                    tokens_texto_concatenado,
                                                    self.get_pln())

            #Acumula a saída do método 
            #Se é uma string uma lista com comprimento 1
            if entrada_eh_string:
                saida['texto_original'] = texto_embeddings['texto_original'][i]
                saida['tokens_texto'] = lista_tokens_texto
                saida['tokens_texto_mcl'] = tokens_texto_mcl
                saida['tokens_oov_texto_mcl'] = lista_tokens_oov_texto_mcl
                saida['tokens_texto_pln'] = lista_tokens_texto_pln
                saida['pos_texto_pln'] = lista_pos_texto_pln
                saida['embeddings_MEAN'] = lista_embeddings_MEAN
                saida['embeddings_MAX'] = lista_embeddings_MAX
            else:
                saida['texto_original'].append(texto_embeddings['texto_original'][i])
                saida['tokens_texto'].append(lista_tokens_texto)
                saida['tokens_texto_mcl'].append(tokens_texto_mcl)
                saida['tokens_oov_texto_mcl'].append(lista_tokens_oov_texto_mcl)            
                saida['tokens_texto_pln'].append(lista_tokens_texto_pln)
                saida['pos_texto_pln'].append(lista_pos_texto_pln)            
                saida['embeddings_MEAN'].append(lista_embeddings_MEAN)
                saida['embeddings_MAX'].append(lista_embeddings_MAX)

        return saida
    
    def getEmbeddingsTokens(self, texto):
        '''
        De um texto preparado(tokenizado) ou não, retorna os embeddings dos tokens do texto.
            
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings das palavras do modelo de linguagem.
    
        Retorna uma lista com os embeddings dos tokens.
        '''
        saida = self.getCodificaoTokens(texto)['token_embeddings']
        
        return saida

    # ============================
    def getCodificaoTokens(self, texto):
        '''        
        De um texto preparado(tokenizado) ou não, retorna os embeddings dos tokens do texto utilizando estratégia pooling MEAN.
    
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings do modelo de linguagem
    
        Retorna uma lista com os embeddings da última camada.
        '''

        # Se o texto é uma string, coloca em uma lista de comprimento 1
        entrada_eh_string = False
        if isinstance(texto, str) or not hasattr(texto, '__len__'):             
            texto = [texto]
            entrada_eh_string = True

        # Recupera os embeddings do texto
        texto_embeddings = self.getEmbeddings(texto)

        # Acumula a saída do método
        saida = {}
        saida.update({'texto_original' : [],
                      'tokens_texto_mcl' : [],                      
                      'token_embeddings': [],        
                      'all_layer_embeddings': []
                     }
        )

        # Percorre os textos da lista.
        for i, sentenca in enumerate(texto_embeddings['tokens_texto_mcl']):            
            # Recupera os embeddings do texto  
            lista_token_embeddings = texto_embeddings['token_embeddings'][i][0:len(texto_embeddings['tokens_texto_mcl'][i])]

            # Recupera os embeddings do texto  
            lista_all_layer_embeddings = texto_embeddings['all_layer_embeddings'][i][0:len(texto_embeddings['tokens_texto_mcl'][i])]
            
            # Recupera a lista de tokens do tokenizado pelo MCL sem CLS e SEP
            tokens_texto_mcl = texto_embeddings['tokens_texto_mcl'][i][1:-1]

            #Acumula a saída do método 
            #Se é uma string uma lista com comprimento 1
            if entrada_eh_string:
                saida['texto_original'] = texto_embeddings['texto_original'][i]
                saida['tokens_texto_mcl'] = tokens_texto_mcl
                saida['token_embeddings'] = lista_token_embeddings
                saida['all_layer_embeddings'] = lista_all_layer_embeddings
            else:
                saida['texto_original'].append(texto_embeddings['texto_original'][i])
                saida['tokens_texto_mcl'].append(tokens_texto_mcl)
                saida['token_embeddings'].append(lista_token_embeddings)
                saida['all_layer_embeddings'].append(lista_all_layer_embeddings)

        return saida

    # ============================
    def get_model(self):
        return self.model

    # ============================
    def get_tokenizer(self):
        return self.tokenizer

    # ============================
    def get_transformer(self):
        return self.transformer

    # ============================    
    def get_mensurador(self):
        return self.mensurador        
        
    # ============================        
    def get_pln(self):
        return self.pln          


  # ============================
    def getEmbeddingsTexto(self, texto):
        '''
        De um texto preparado(tokenizado) ou não, retorna token_embeddings, input_ids, attention_mask, token_type_ids, 
        tokens_texto, texto_original  e all_layer_embeddings em um dicionário.
        
        Facilita acesso a classe Transformer.
    
        Parâmetros:
        `texto` - Um texto a ser recuperado os embeddings do modelo de linguagem
    
        Retorna:
            token_embeddings uma lista com os embeddings da última camada.
           
        '''
        return self.get_transformer().getEmbeddings(texto)['token_embeddings']