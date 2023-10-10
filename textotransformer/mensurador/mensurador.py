# Import das bibliotecas.

# Biblioteca de logging
import logging

# Biblioteca de aprendizado de máquina
import torch 

# Bibliotecas próprias
from textotransformer.modelo.transformer import Transformer
from textotransformer.modelo.modeloargumentos import ModeloArgumentos 
from textotransformer.pln.pln import PLN
from textotransformer.mensurador.medidas import similaridadeCosseno, produtoEscalar, distanciaEuclidiana, distanciaManhattan
from textotransformer.mensurador.mensuradorenum import PalavraRelevante
from textotransformer.modelo.modeloenum import EstrategiasPooling
from textotransformer.util.utiltexto import encontrarIndiceSubLista  

# Objeto de logger
logger = logging.getLogger(__name__)

class Mensurador:

    ''' 
    Realiza mensurações de embeddings em textos.
     
    Parâmetros:
       `modelo_args` - Parâmetros do modelo de linguagem.
       `transformer` - Modelo de linguagem carregado.
       `pln` - Processador de linguagem natural.
       `device` - Dispositivo (como 'cuda' / 'cpu') que deve ser usado para computação. Se none, verifica se uma GPU pode ser usada.
    ''' 

    # Construtor da classe
    def __init__(self, modelo_args: ModeloArgumentos,
                 transformer: Transformer, 
                 pln: PLN,
                 device: str = None):
    
        # Parâmetros do modelo
        self.model_args = modelo_args
    
        # Recupera o objeto do transformer.
        self.transformer = transformer
    
        # Recupera o modelo.
        self.auto_model = transformer.getAutoModel()
    
        # Recupera o tokenizador.     
        self.auto_tokenizer = transformer.getTokenizer()
        
        # Recupera a classe PLN
        self.pln = pln
        
        # Recupera o dispositivo
        self._target_device = device
                
        logger.info("Classe \"{}\" carregada: \"{}\".".format(self.__class__.__name__, modelo_args))

    # ============================   
    def __repr__(self):
        '''
        Retorna uma string com descrição do objeto.
        '''
        
        return "Classe (\"{}\") com  Transformer: \"{}\", tokenizador: \"{}\" e NLP: \"{}\" ".format(self.__class__.__name__,
                                                                                                     self.auto_model.__class__.__name__,
                                                                                                     self.auto_tokenizer.__class__.__name__,
                                                                                                     self.pln.__class__.__name__)

    # ============================
    def getMedidasSentencasEmbeddingMEAN(self, embedding_si, 
                                         embedding_sj):
        '''
        Retorna as medidas de duas sentenças Si e Sj utilizando a estratégia MEAN.
        
        Parâmetros:
           `embedding_si` - Embeddings da primeira sentença.
           `embedding_sj` - Embeddings da segunda sentença.
        
        Retorno:
           `Scos` - Similaridade do cosseno - usando a média dos embeddings Si e Sj das camadas especificadas.
           `Spro` - Produto escaar - usando a média dos embeddings Si e Sj das camadas especificadas.
           `Seuc` - Distância euclidiana - usando a média dos embeddings Si e Sj das camadas especificadas.
           `Sman` - Distância de manhattan - usando a média dos embeddings Si e Sj das camadas especificadas.           
        '''

        #print('embedding_si=', embedding_si.shape) 
        #print('embedding_sj=', embedding_sj.shape)
      
        # As operações de subtração(sub), mul(multiplicação/produto), soma(sum), cosseno(similaridade), euclediana(diferença) e manhattan(diferença)
        # Necessitam que os embeddings tenha a mesmo número de dimensões.
      
        # Calcula a média dos embeddings para os tokens de Si, removendo a primeira dimensão.
        # Entrada: <qtde_tokens> x <768 ou 1024>  
        media_embedding_si = torch.mean(embedding_si, dim=0)    
        # Retorno: <768 ou 1024>
        #print('mediaCamadasSi=', mediaCamadasSi.shape)
      
        # Calcula a média dos embeddings para os tokens de Sj, removendo a primeira dimensão.
        # Entrada: <qtde_tokens> x <768 ou 1024>  
        media_embedding_sj = torch.mean(embedding_sj, dim=0)    
        # Retorno: <768 ou 1024>
        #print('mediaCamadasSj=', mediaCamadasSj.shape)
        
        # Similaridade do cosseno entre os embeddings Si e Sj
        # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
        Scos = similaridadeCosseno(media_embedding_si, media_embedding_sj)
        # Retorno: Número real
        
        # Produto escalar entre os embeddings Si e Sj
        # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
        Spro = produtoEscalar(media_embedding_si, media_embedding_sj)
        # Retorno: Número real

        # Distância euclidiana entre os embeddings Si e Sj
        # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
        Seuc = distanciaEuclidiana(media_embedding_si, media_embedding_sj)
        # Retorno: Número real

        # Distância de manhattan entre os embeddings Si e Sj
        # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
        Sman = distanciaManhattan(media_embedding_si, media_embedding_sj)
        # Retorno: Número real
        
        # Retorno das medidas das sentenças  
        return media_embedding_si, media_embedding_sj, Scos, Spro, Seuc, Sman

    # ============================
    def getMedidasSentencasEmbeddingMAX(self, embedding_si, 
                                        embedding_sj):
        '''
        Retorna as medidas de duas sentenças Si e Sj utilizando a estratégia MAX.
        
        Parâmetros:
           `embedding_si` - Embeddings da primeira sentença.
           `embedding_sj` - Embeddings da segunda sentença.
           
        Retorno:
           `Scos` - Similaridade do cosseno - usando o maior dos embeddings Si e Sj das camadas especificadas.
           `Spro` - Produto escaar - usando o maior dos embeddings Si e Sj das camadas especificadas.
           `Seuc` - Distância euclidiana - usando o maior dos embeddings Si e Sj das camadas especificadas.
           `Sman` - Distância de manhattan - usando o maior dos embeddings Si e Sj das camadas especificadas.
        '''

        #print('embedding_si=', embedding_si.shape) 
        #print('embedding_sj=', embedding_sj.shape)

        # As operações de subtração(sub), mul(multiplicação/produto), soma(sum), cosseno(similaridade), euclediana(diferença) e manhattan(diferença)
        # Necessitam que os embeddings tenha a mesmo número de dimensões.

        # Encontra os maiores embeddings os tokens de Si, removendo a primeira dimensão.
        # Entrada: <qtde_tokens> x <768 ou 1024>  
        maior_embedding_si, linha = torch.max(embedding_si, dim=0)    
        # Retorno: <768 ou 1024>
        #print('maior_embedding_si=', maior_embedding_si.shape)

        # Encontra os maiores embeddings os tokens de Sj, removendo a primeira dimensão.
        # Entrada: <qtde_tokens> x <768 ou 1024>  
        maior_embedding_sj, linha = torch.max(embedding_sj, dim=0)    
        # Retorno: <768 ou 1024>
        #print('maior_embedding_sj=', maior_embedding_sj.shape)

        # Similaridade do cosseno entre os embeddings Si e Sj
        # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
        Scos = similaridadeCosseno(maior_embedding_si, maior_embedding_sj)
        # Retorno: Número real
        
        # Produto escalar entre os embeddings Si e Sj
        # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
        Spro = produtoEscalar(maior_embedding_si, maior_embedding_sj)
        # Retorno: Número real

        # Distância euclidiana entre os embeddings Si e Sj
        # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
        Seuc = distanciaEuclidiana(maior_embedding_si, maior_embedding_sj)
        # Retorno: Número real

        # Distância de manhattan entre os embeddings Si e Sj
        # Entrada: (<768 ou 1024>) x (<768 ou 1024>)
        Sman = distanciaManhattan(maior_embedding_si, maior_embedding_sj)
        # Retorno: Número real

        # Retorno das medidas das sentenças
        return maior_embedding_si, maior_embedding_sj, Scos, Spro, Seuc, Sman

    # ============================
    def getMedidasSentencasEmbedding(self, embedding_si, 
                                     embedding_sj):
        '''
        Realiza o cálculo da medida do texto de acordo com a estratégia de pooling(MAX ou MEAN).
        
        Parâmetros:
           `embedding_si` - Embeddings da primeira sentença.
           `embedding_sj` - Embeddings da segunda sentença.
           `estrategia_pooling` - Estratégia de pooling a ser utilizada.     
        
        Retorno:
           As medidas de duas sentenças Si e Sj utilizando a estratégia especificada.  
        '''

        if self.model_args.estrategia_pooling == EstrategiasPooling.MEAN.value:
            return self.getMedidasSentencasEmbeddingMEAN(embedding_si=embedding_si, embedding_sj=embedding_sj)
        else:
            if self.model_args.estrategia_pooling == EstrategiasPooling.MAX.value:
                return self.getMedidasSentencasEmbeddingMAX(embedding_si=embedding_si, embedding_sj=embedding_sj)
            else:
                logger.info("Nenhuma seleção da estratégia de pooling foi especificada.")
                return None

    # ============================
    def getEmbeddingSentencaEmbeddingTextoALL(self, embedding_texto, 
                                              texto: str, 
                                              sentenca: str,
                                              posicao_sentenca: int):
        '''
        Retorna os embeddings de uma sentença com todas as palavras(ALL) a partir dos embeddings do texto.
        
        Parâmetros:
           `embedding_texto` - Embeddings do texto.
           `texto` - Texto.
           `sentenca` - Sentença.
           `posicao_sentenca` - Posição da sentença no texto.
        
        Retorno:
           Uma lista com os embeddings de uma sentença com todas as palavras(ALL) a partir dos embeddings do texto.
        '''
                   
        # Tokeniza o texto
        texto_tokenizado =  self.transformer.getTextoTokenizado(texto=texto)
        #print(texto_tokenizado)
        
        # Tokeniza a sentença
        sentenca_tokenizada =  self.transformer.getTextoTokenizado(texto=sentenca)
        #print(sentenca_tokenizada)
        
        # Remove os tokens de início e fim da sentença
        sentenca_tokenizada = self.transformer.removeTokensEspeciais(lista_tokens=sentenca_tokenizada)
        #print(len(sentenca_tokenizada))
        
        # Se for do tipo Roberta, GTP2 Model, adiciona o token de separação no início da sentença
        if posicao_sentenca != 0:
            if self.getTransformer().getPrimeiroTokenSemSeparador():
                sentenca_tokenizada = self.transformer.trataListaTokensEspeciais(tokens_texto_mcl=sentenca_tokenizada)
        
        # Localiza os índices dos tokens da sentença no texto
        inicio, fim = encontrarIndiceSubLista(lista=texto_tokenizado, sublista=sentenca_tokenizada)        
        #print("inicio:", inicio, "   fim:", fim)
        if inicio == -1 or fim == -1:            
            logger.error("Não encontrei a sentença: {} dentro de {}.".format(sentenca_tokenizada, texto_tokenizado))

        # Recupera os embeddings dos tokens da sentença a partir dos embeddings do texto
        embedding_sentenca = embedding_texto[inicio:fim + 1]
        #print('embedding_sentenca=', embedding_sentenca.shape)

        # Retorna o embedding da sentença no texto
        return embedding_sentenca

    # ============================
    def getEmbeddingSentencaEmbeddingTextoCLEAN(self, embedding_texto, 
                                                texto: str, 
                                                sentenca: str,
                                                posicao_sentenca: int):
        '''
        Retorna os embeddings de uma sentença sem stopwords(CLEAN) a partir dos embeddings do texto.
        
        Parâmetros:
          `embedding_texto` - Embeddings do texto.
          `texto` - Texto.
          `sentenca` - Sentença.
          `posicao_sentenca` - Posição da sentença no texto.
        
        Retorno:
           Uma lista com os embeddings de uma sentença sem stopwords(CLEAN) a partir dos embeddings do texto.
        
        '''
          
        # Tokeniza o texto
        texto_tokenizado = self.transformer.getTextoTokenizado(texto=texto)  
        #print(sentenca_tokenizada)

        # Remove as stopword da sentença
        sentenca_sem_stopword = self.pln.removeStopWord(texto=sentenca)

        # Tokeniza a sentença sem stopword
        sentenca_tokenizada_sem_stopword =  self.transformer.getTextoTokenizado(texto=sentenca_sem_stopword)
        #print(sentenca_tokenizada_sem_stopwod)

        # Remove os tokens de início e fim da sentença
        sentenca_tokenizada_sem_stopword = self.transformer.removeTokensEspeciais(lista_tokens=sentenca_tokenizada_sem_stopword)        
        #print(sentenca_tokenizada_sem_stopword)      
        #print(len(sentenca_tokenizada_sem_stopword))

        # Tokeniza a sentença
        sentenca_tokenizada =  self.transformer.getTextoTokenizado(texto=sentenca)
        
        # Remove os tokens de início e fim da sentença
        sentenca_tokenizada = self.transformer.removeTokensEspeciais(lista_tokens=sentenca_tokenizada)
        #print(sentenca_tokenizada)
        #print(len(sentenca_tokenizada))
                
        # Se for do tipo Roberta, GTP2 Model, adiciona o token de separação no início da sentença
        if posicao_sentenca != 0:
            if self.getTransformer().getPrimeiroTokenSemSeparador():
                sentenca_tokenizada = self.transformer.trataListaTokensEspeciais(tokens_texto_mcl=sentenca_tokenizada)

        # Localiza os índices dos tokens da sentença no texto
        inicio, fim = encontrarIndiceSubLista(lista=texto_tokenizado, sublista=sentenca_tokenizada)
        #print("inicio:", inicio, "   fim:", fim)
        if inicio == -1 or fim == -1:
            logger.error("Não encontrei a sentença: {} dentro de {}.".format(sentenca_tokenizada, texto_tokenizado))

        # Recupera os embeddings dos tokens da sentença a partir dos embeddings do texto
        embedding_sentenca = embedding_texto[inicio:fim + 1]
        #print('embedding_sentenca=', embedding_sentenca.shape)

        # Lista com os tensores selecionados
        lista_tokens_selecionados = []
        # Localizar os embeddings dos tokens da sentença tokenizada sem stop word na sentença 
        # Procura somente no intervalo da sentença
        for i, token_sentenca in enumerate(sentenca_tokenizada):
            for token_sentenca_sem_stopword in sentenca_tokenizada_sem_stopword: 
                # Se o token da sentença é igual ao token da sentença sem stopword    
                if token_sentenca == token_sentenca_sem_stopword:
                    lista_tokens_selecionados.append(embedding_sentenca[i:i + 1])

        embedding_sentenca_sem_stopword = None

        if len(lista_tokens_selecionados) != 0:
            # Concatena os vetores da lista pela dimensão 0
            embedding_sentenca_sem_stopword = torch.cat(lista_tokens_selecionados, dim=0)
            #print("embedding_sentenca_sem_stopword:",embedding_sentenca_sem_stopword.shape)

        # Retorna o embedding da sentença no texto
        return embedding_sentenca_sem_stopword

    # ============================
    def getEmbeddingSentencaEmbeddingTextoNOUN(self, embedding_texto, 
                                               texto: str, 
                                               sentenca: str,
                                               posicao_sentenca: int):
        '''
        Retorna os embeddings de uma sentença somente com as palavras relevantes(NOUN) de um tipo a partir dos embeddings do texto.
        
        Parâmetros:
           `embedding_texto` - Embeddings do texto.
           `texto` - Texto.
           `sentenca` - Sentença.
           `posicao_sentenca` - Posição da sentença no texto.
        
        Retorno:
           Uma lista com os embeddings de uma sentença somente com as palavras relevantes(NOUN) de um tipo a partir dos embeddings do texto.
        
        '''

        # Tokeniza o texto
        texto_tokenizado =  self.transformer.getTextoTokenizado(texto=texto)  
        #print(sentenca_tokenizada)

        # Retorna as palavras relevantes da sentença do tipo especificado
        sentenca_somente_relevante = self.pln.retornaPalavraRelevante(texto=sentenca, 
                                                                      tipo_palavra_relevante=self.model_args.palavra_relevante)

        # Tokeniza a sentença 
        sentenca_tokenizada_somente_relevante =  self.transformer.getTextoTokenizado(texto=sentenca_somente_relevante)

        # Remove os tokens de início e fim da sentença
        sentenca_tokenizada_somente_relevante = self.transformer.removeTokensEspeciais(lista_tokens=sentenca_tokenizada_somente_relevante)
        #print(sentenca_tokenizada_somente_relevante)
        #print(len(sentenca_tokenizada_somente_relevante))

        # Tokeniza a sentença
        sentenca_tokenizada =  self.transformer.getTextoTokenizado(texto=sentenca)

        # Remove os tokens de início e fim da sentença
        sentenca_tokenizada = self.transformer.removeTokensEspeciais(lista_tokens=sentenca_tokenizada)
        #print(sentenca_tokenizada)
        #print(len(sentenca_tokenizada))
        
        # Se for do tipo Roberta, GTP2 Model, adiciona o token de separação no início da sentença
        if posicao_sentenca != 0:
            if self.getTransformer().getPrimeiroTokenSemSeparador():
                sentenca_tokenizada = self.transformer.trataListaTokensEspeciais(tokens_texto_mcl=sentenca_tokenizada)

        # Localiza os índices dos tokens da sentença no texto
        inicio, fim = encontrarIndiceSubLista(lista=texto_tokenizado, sublista=sentenca_tokenizada)
        #print("inicio:", inicio, "   fim:", fim)
        if inicio == -1 or fim == -1:
            logger.error("Não encontrei a sentença: {} dentro de {}.".format(sentenca_tokenizada, texto_tokenizado))

        # Recupera os embeddings dos tokens da sentença a partir dos embeddings do texto
        embedding_sentenca = embedding_texto[inicio:fim + 1]
        #print('embedding_sentenca=', embedding_sentenca.shape)

        # Lista com os tensores selecionados
        listaTokensSelecionados = []
        # Localizar os embeddings dos tokens da sentença tokenizada sem stop word na sentença 
        # Procura somente no intervalo da sentença
        for i, token_sentenca in enumerate(sentenca_tokenizada):
            for token_sentenca_somente_relevante in sentenca_tokenizada_somente_relevante: 
                if token_sentenca == token_sentenca_somente_relevante:        
                    listaTokensSelecionados.append(embedding_sentenca[i:i + 1])

        embedding_sentenca_com_substantivo = None

        if len(listaTokensSelecionados) != 0:
            # Concatena os vetores da lista pela dimensão 0  
            embedding_sentenca_com_substantivo = torch.cat(listaTokensSelecionados, dim=0)
            #print("embedding_sentenca_com_substantivo:",embedding_sentenca_com_substantivo.shape)

        # Retorna o embedding da sentença do texto
        return embedding_sentenca_com_substantivo

    # ============================
    def getEmbeddingSentencaEmbeddingTexto(self, embedding_texto, 
                                           texto: str, 
                                           sentenca: str,
                                           posicao_sentenca: int):
        '''
        Retorna os embeddings de uma sentença considerando a relevância das palavras (ALL, CLEAN ou NOUN) a partir dos embeddings do texto.    
        
        Parâmetros:
           `embedding_texto` - Embeddings do texto.
           `texto` - Texto.
           `sentenca` - Sentença.
           `posicao_sentenca' - Posição da sentença no texto.
        
        Retorno:
           Uma lista com os embeddings de uma sentença considerando a relevância das palavras (ALL, CLEAN ou NOUN) a partir dos embeddings do texto.
        '''

        if self.model_args.palavra_relevante == PalavraRelevante.ALL.value:
            return self.getEmbeddingSentencaEmbeddingTextoALL(embedding_texto=embedding_texto, 
                                                              texto=texto, 
                                                              sentenca=sentenca, 
                                                              posicao_sentenca=posicao_sentenca)
        else:
            if self.model_args.palavra_relevante == PalavraRelevante.CLEAN.value:                
                return self.getEmbeddingSentencaEmbeddingTextoCLEAN(embedding_texto=embedding_texto, 
                                                                    texto=texto, 
                                                                    sentenca=sentenca, 
                                                                    posicao_sentenca=posicao_sentenca)
            else:
                if self.model_args.palavra_relevante == PalavraRelevante.NOUN.value:
                    return self.getEmbeddingSentencaEmbeddingTextoNOUN(embedding_texto=embedding_texto, 
                                                                       texto=texto, 
                                                                       sentenca=sentenca, 
                                                                       posicao_sentenca=posicao_sentenca)
                else:
                    logger.info("Nenhuma estratégia de relevância de palavras foi especificada.") 
    
    # ============================
    def getSaidaRedeMensurador(self, texto,
                               abordagem_extracao_embeddings_camadas,
                               device: str = None):
        
        '''
        Retorna os embeddings do texto de acordo com a abordagem de extração especificada.
        
        Parâmetros:
           `texto` - Texto a ser recuperado os embeddings.
           `abordagem_extracao_embeddings_camadas` - Camada de onde deve ser recupera os embeddings.
           `device` - Qual torch.device usar para a computação.

        Retorno:
           Uma lista com os embeddings do texto de acordo com a abordagem de extração especificada.
        '''
        
        # Coloca o modelo em modo avaliação
        self.auto_model.eval()
        
        # Se o texto não estiver tokenizado, tokeniza
        if not isinstance(texto, dict):
            texto = self.getTransformer().tokenize(texto)
            
        # Se não foi especificado um dispositivo, use-o defaul
        if device is None:
            device = self._target_device
     
        # Adiciona um dispositivo ao modelo
        self.auto_model.to(device)            

        # Adiciona ao device gpu ou cpu
        lote_textos_tokenizados = self.getTransformer().batchToDevice(lote=texto, 
                                                                      target_device=device)            
        
        # Roda o texto através do modelo de linguagem, e coleta todos os estados ocultos produzidos.
        with torch.no_grad():
            
            # Recupera a saída da rede
            saida = self.getTransformer().getSaidaRedeCamada(texto=lote_textos_tokenizados,
                                                             abordagem_extracao_embeddings_camadas=abordagem_extracao_embeddings_camadas)
        
        return saida
        
    # ============================
    def getEmbeddingTextoCamada(self, texto, 
                                abordagem_extracao_embeddings_camadas,
                                converte_para_numpy: bool = True):
        '''
        Retorna os embeddings do texto de acordo com a abordagem de extração especificada.
        
        Parâmetros:
           `texto` - Texto a ser recuperado os embeddings.
           `abordagem_extracao_embeddings_camadas` - Camada de onde deve ser recupera os embeddings.
           `converte_para_numpy` - Se verdadeiro, a saída em vetores numpy. Caso contrário, é uma lista de tensores pytorch.

        Retorno:
           Uma lista com os embeddings do texto de acordo com a abordagem de extração especificada.
        '''
        
        # Roda o texto através do modelo de linguagem, e coleta todos os estados ocultos produzidos.
        saida = self.getSaidaRedeMensurador(texto=texto,
                                            abordagem_extracao_embeddings_camadas=abordagem_extracao_embeddings_camadas)
        
        # Remove o lote com [0]
        embedding_texto = saida['embedding_extraido'][0]
        
        # Desconecta
        if converte_para_numpy:
            embedding_texto = embedding_texto.cpu()
        
        return embedding_texto

    # ============================
    def getMedidasComparacaoTexto(self, texto, 
                                  abordagem_extracao_embeddings_camadas,
                                  converte_para_numpy: bool = True):
        '''
        Retorna as medidas do texto.
        Considera somente sentenças com pelo menos uma palavra.
        Estratégia de pooling padrão é MEAN(0).
        Palavra relavante padrão é ALL(0).
        
        Parâmetros:
           `texto` - Texto a ser realizado as comparações.
           `abordagem_extracao_embeddings_camadas` - Camada de onde deve ser recupera os embeddings.
           `converte_para_numpy` - Se verdadeiro, a saída em vetores numpy. Caso contrário, é uma lista de tensores pytorch.
                 
        Retorno um dicionário com:
           `cos` - Medida de cos do texto.
           `pro` - Medida de pro do texto.
           `euc` - Medida de euc do texto.
           `man` - Medida de man do texto.
        '''

        # Quantidade de sentenças no texto
        n = len(texto)
        
        # Divisor da quantidade de textos
        divisor = n - 1

        # Texto é uma lista com as sentenças
        #print('abordagem_extracao_embeddings_camadas=',abordagem_extracao_embeddings_camadas)
        #print('Texto=', texto)

        # Junta a lista de sentenças em um texto(string)
        string_texto = ' '.join(texto)
        #print('string_texto=', string_texto)

        # Recupera os embeddings dos tokens das camadas especificadas de acordo com a estratégia especificada para camada          
        embedding_texto = self.getEmbeddingTextoCamada(texto=string_texto,
                                                       abordagem_extracao_embeddings_camadas=abordagem_extracao_embeddings_camadas,
                                                       converte_para_numpy=converte_para_numpy)
        #print('embedding_texto=', embedding_texto.shape)

        # Acumuladores das medidas entre as sentenças  
        soma_Scos = 0
        soma_Spro = 0
        soma_Seuc = 0
        soma_Sman = 0

        # Seleciona os pares de sentença a serem avaliados
        pos_si = 0
        pos_sj = pos_si + 1

        #Enquanto o indíce da sentneça posSj(2a sentença) não chegou ao final da quantidade de sentenças
        while pos_sj <= (n-1):  

            # Seleciona as sentenças do texto  
            Si = texto[pos_si]
            Sj = texto[pos_sj]

            # Recupera os embedding das sentenças Si e Sj do embedding do texto      
            embedding_si = self.getEmbeddingSentencaEmbeddingTexto(embedding_texto=embedding_texto, 
                                                                   texto=string_texto, 
                                                                   sentenca=Si, 
                                                                   posicao_sentenca=pos_si)
            embedding_sj = self.getEmbeddingSentencaEmbeddingTexto(embedding_texto=embedding_texto, 
                                                                   texto=string_texto, 
                                                                   sentenca=Sj, 
                                                                   posicao_sentenca=pos_sj)

            # Verifica se os embeddings sentenças estão preenchidos
            if embedding_si != None and embedding_sj != None:

                # Recupera as medidas entre Si e Sj     
                ajustado_embedding_si, ajustado_embedding_sj, Scos, Spro, Seuc, Sman = self.getMedidasSentencasEmbedding(embedding_si=embedding_si, 
                                                                                                                   embedding_sj=embedding_sj)

                # Acumula as medidas
                soma_Scos = soma_Scos + Scos
                soma_Spro = soma_Spro + Spro
                soma_Seuc = soma_Seuc + Seuc
                soma_Sman = soma_Sman + Sman

                # avança para o próximo par de sentenças
                pos_si = pos_sj
                pos_sj = pos_sj + 1
            else:
                # Reduz um da quantidade de sentenças pois uma delas está vazia
                divisor = divisor - 1
                # Se embedding_si igual a None avanca pos1 e pos2
                if embedding_si == None:
                    # Avança a posição da sentença posSi para a posSj
                    pos_si = pos_sj
                    # Avança para a próxima sentença de posSj
                    pos_sj = pos_sj + 1        
                else:          
                    # Se embeddingSj = None avança somente posJ para a próxima sentença
                    if embedding_sj == None:
                        pos_sj = pos_sj + 1

        # Calcula a medida 
        Ccos = 0
        Cpro = 0
        Ceuc = 0
        Cman = 0

        if divisor != 0:
            Ccos = float(soma_Scos) / float(divisor)
            Cpro = float(soma_Spro) / float(divisor)
            Ceuc = float(soma_Seuc) / float(divisor)
            Cman = float(soma_Sman) / float(divisor)

        # Retorna as medidas em um dicionário
        saida = {}
        saida.update({'cos' : Ccos,
                      'pro' : Cpro,
                      'euc' : Ceuc,
                      'man' : Cman})

        return saida
    
    # ============================
    def getTransformer(self) -> Transformer:
        '''
        Recupera o transformer.
        '''
        
        return self.transformer
    
    # ============================
    def getModel(self):
        '''
        Recupera o modelo.
        '''
        
        return self.auto_model

    # ============================
    def getTokenizer(self):
        '''
        Recupera o tokenizador.
        '''
        
        return self.auto_tokenizer

    # ============================        
    def getPln(self) -> PLN:
        '''
        Recupera o pln.
        '''
        
        return self.pln