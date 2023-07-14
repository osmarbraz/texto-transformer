# Import das bibliotecas.

# Biblioteca de logging
import logging
# Biblioteca de tipos
from typing import List
# Biblioteca para o sorteio
from random import randint 

# Biblioteca de aprendizado de máquina
import torch 

# Bibliotecas próprias
from textotransformer.modelo.transformer import Transformer
from textotransformer.modelo.modeloargumentos import ModeloArgumentos 
from textotransformer.pln.pln import PLN
from textotransformer.util.utiltexto import contaElemento

# Objeto de logger
logger = logging.getLogger(__name__)

class Modificador:

    ''' 
    Realiza modificações em texto utilizando o modelo de linguagem mascarada do modelo de linguagem.
     
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
    def getTextoMascarado(self, texto: str,
                          classe: List[str] = ["VERB","NOUN","AUX"], 
                          qtde: int = 1):
        ''' 
        Gera o texto mascarado com [MAKS] para usar com MLM do BERT.
        Considera determinadas classes morfossintática das palavras e uma quantidade(qtde) de palavras a serem mascaradas.
            
        Parâmetros:
           `texto` - Texto a ser mascarada.           
           `classe` - Lista com as classes morfossintática das palavras a serem mascarada com [MASK].
           `qtde` - Quantidade de mascarada a serem realizadas nas palavras do texto.
                    Seleciona aleatoriamente a(s) palavra(s) a ser(em) mascarada(s) se a qtde 
                    for menor que quantidade de palavras das classes no texto.

        Retorno:    
           `texto_mascarado` - Texto mascarado.
           `palavra_mascarada` - Lista com as palavras substituidas pela máscara.
        '''
        
        # Somente modelos que possuem token de mascara.
        if self.getTransformer().getTokenMascara() != None:
            
            # Recupera os tokens e POSTagging do texto
            texto_token, texto_pos = self.getPln().getListaTokensPOSTexto(texto)
            # print("texto_token:", texto_token)
            # print("texto_pos:",texto_pos)
        
            texto_mascarado = ""
            palavra_mascarada = ""

            # Verifica a quantidade de trocas a ser realizada
            if qtde != 0:

                # Conta o número de palavras das classes especificadas
                if len(classe) > 1:
                    # Se tem duas classes usa a primeira para contar se existe uma palavra
                    # Pega o primeiro para realizar a conta
                    classe_conta = [classe[0]]
                    conta_mascara = contaElemento(texto_pos, classe_conta)
                    
                    # Senão encontrar pega a segunda classe
                    if conta_mascara == 0:
                        #Pega a segunda classe
                        classe_conta = [classe[1]]
                        conta_mascara = contaElemento(texto_pos, classe_conta)

                        # Senão encontrar pega a terceira classe
                        if conta_mascara == 0:
                            #Pega a terceira classe
                            classe_conta = [classe[2]]
                            conta_mascara = contaElemento(texto_pos, classe_conta) 
                    
                    # Usa a classe para gerar o texto mascarado
                    classe = classe_conta
                else:
                    conta_mascara = contaElemento(texto_pos, classe)
                
                # Verifica se existe palavras das classes a serem mascaradas
                if conta_mascara != 0:    
                    # Verifica a quantidade de trocas é menor que a quantidade palavras a serem trocadas encontradas
                    if qtde < conta_mascara:
                        # A quantidade de trocas é menor que a quantidade de palavras existentes
                        # Precisa sortear as posições que serão trocadas pela máscara dentro da quantidade
                            
                        roleta = []
                        # preenche a roleta com o indice das palavras as serem mascaradas
                        for i in range(conta_mascara):
                            roleta.append(i)

                        # Sorteia as posições das trocas
                        posicao = []
                        for i in range(qtde):
                            posicao_sorteio = randint(0, len(roleta)-1)
                            # Guarda o número sorteado
                            posicao.append(roleta[posicao_sorteio])
                            # Remove o elemento sorteado da roleta
                            del roleta[posicao_sorteio]
                        
                        # Conta o número das trocas realizadas
                        troca = 0

                        # Substitui o elemento pela máscara
                        for i, token in enumerate(texto_token):            
                            # Se a classe da palavra é a desejada
                            if texto_pos[i] in classe:
                                # Verifica se a troca deve ser realizada para a posição
                                if troca in posicao:      
                                    # Trocar palavra da classe por [MASK]
                                    texto_mascarado = texto_mascarado + self.getTransformer().getTokenMascara() + " "    
                                    # Guarda a palavra que foi mascarada
                                    palavra_mascarada = token                                  
                                else:                  
                                    # Adiciona o token
                                    texto_mascarado = texto_mascarado + token + " "
                                    # Avança para a próxima troca
                                    troca = troca + 1
                            else:
                                # Adiciona o token
                                texto_mascarado = texto_mascarado + token + " "
                    else:        
                        # Trocar todas as palavras pela mascara, pois a quantidade
                        # de trocas é igual a quantidade de mascaras existentes na sentença

                        # Substitui o elemento da classe pela mascara
                        for i, token in enumerate(texto_token):
                            #print(token, texto_token[i])        
                            # Se a classe da palavra é a desejada
                            if texto_pos[i] in classe:
                                # Trocar palavra da classe por [MASK]
                                texto_mascarado = texto_mascarado + self.getTransformer().getTokenMascara() + " "    
                                # Guarda a palavra que foi mascarada
                                palavra_mascarada = token 
                            else:
                                texto_mascarado = texto_mascarado + token + " "
                else:
                    # Não existe palavras da classe especificada      
                    logger.erro("Não existe palavras da classe especificada.")
                    logger.erro("texto:",texto)
                    logger.erro("texto_pos:",texto_pos)
                    texto_mascarado = texto    
            else:
                # Quantidade trocas igual a 0
                logger.erro("Não foi especificado uma quantidade de trocas.")
                texto_mascarado = texto

            # Retira o espaço em branco do início e fim do texto
            texto_mascarado = texto_mascarado.strip(" ")

            return texto_mascarado, palavra_mascarada
        
        else:
            logger.error("O modelo \"{}\" não possui um token de máscara.".format(self.getModel().__class__.__name__))
            
            return None        
    
    # ============================
    def getPrevisaoPalavraTexto(self, texto: str,
                                top_k_predicao: int = 1):
        ''' 
        Retorna uma lista com as k previsões para a palavra mascarada no texto.
            
        Parâmetros:
            `texto` - Texto mascarado.
            `top_k_predicao` - Quantidade de palavras a serem recuperadas mais próximas da máscara.

        Retorno:
            Lista com as k previsões para a palavra mascarada no texto.

        '''
        
        # Somente modelos que possuem token de mascara.
        if self.getTransformer().getTokenMascara() != None:
        
            # Divide as palavras em tokens        
            texto_tokenizado = self.getTransformer().getTextoTokenizado(texto)
            #print("texto_tokenizado:", texto_tokenizado)

            # Retorna o índice da mascara de atenção
            mascara_atencao_indice = texto_tokenizado.index(self.getTransformer().getTokenMascara())
            #print("mascara_atencao_indice:", mascara_atencao_indice)

            # Mapeia os tokens em seus índices do vocabulário
            tokens_indexados = self.getTokenizer().convert_tokens_to_ids(texto_tokenizado)
            #print("tokens_indexados:", tokens_indexados)

            # Define índices das sentenças A e B associados à 1ª e 2ª sentença 
            segmentos_ids = [0]*len(texto_tokenizado)
            
            # Converte as entradas de lista para tensores do torch
            tokens_tensores = torch.tensor([tokens_indexados])
            segmentos_tensores = torch.tensor([segmentos_ids])
            
            # Se existe GPU disponível.
            if torch.cuda.is_available():  
                # Se você tem uma GPU
                tokens_tensores = tokens_tensores.to('cuda')
                segmentos_tensores = segmentos_tensores.to('cuda')  

            # Realiza a predição dos tokens
            with torch.no_grad():
                ## Retorno de model quando ´output_hidden_states=True´ é setado:  
                ##outputs[0] = last_hidden_state, outputs[1] = pooler_output, outputs[2] = hidden_states
                outputs = self.getModel()(tokens_tensores, token_type_ids=segmentos_tensores)

                ## A predição é recuperada dos embeddings da última camada oculta do modelo        
                predicao = outputs[0]
                
            #print("shape:", predicao.shape)
            #print("mascara_atencao_indice:", mascara_atencao_indice)

            # Normaliza os pesos dos embeddings das predições e calcula sua probabilidade usando softmax.
            probabilidades = torch.nn.functional.softmax(predicao[0, mascara_atencao_indice], dim=-1)    
            # Probabilidade de cada uma das 29.794 palavras do vocabulário do BERT ser a palavra mascarada.
            #print("Tamanho vocabulário:", len(tokenizer.get_vocab())) #29.794
            
            # Retorna os k maiores elementos com as maiores probabilidades e sua posição(ordenada descrescentemente).
            top_k_predicao_pesos, top_k_predicao_indices = torch.topk(probabilidades, top_k_predicao, sorted=True)
            
            # Converte os ids para os tokens do vocabulário
            tokens_predicao = self.getTokenizer().convert_ids_to_tokens([ind.item() for ind in top_k_predicao_indices])

            # Retorna a predição e a probabilidade      
            return list(zip(tokens_predicao, top_k_predicao_pesos))[:top_k_predicao]            
        
        else:
            logger.error("O modelo \"{}\" não possui um token de máscara.".format(self.getModel().__class__.__name__))
            
            return None
    
    # ============================  
    def getModificacaoPalavraTextoSequencial(self, 
                                             texto : str,
                                             classe: List[str] = ["VERB","NOUN","AUX"], 
                                             qtde: int = 1,
                                             top_k_predicao: int = 10):
        ''' 
        Gera a palavras da modificação do texto com seleção das top_k predições(em sequencia).        
        Considera determinadas classes morfossintática das palavras.
            
        Parâmetros:
           `texto` - Texto a ser mascarada.
           `classe` - Lista com as classes morfossintática das palavras a serem mascarada com [MASK].
                      Valor default ["VERB","NOUN","AUX"], primeiro procura uma verbo, depois um substantivo e por último um auxiliar.
           `qtde` - Quantidade de palavras a serem substituidas pela máscara de acordo com a ordem das classes.
                    Seleciona aleatoriamente a(s) palavra(s) a ser(em) mascarada(s) se a qtde 
                    for menor que quantidade de palavras das classes no texto.
           `top_k_predicao` - Quantidade de palavras a serem recuperadas mais próximas da máscara.

        Retorno um lista de dicionários com as top_k previsões para cada palavra mascarada no texto:
           `indice_token` - Índice do token.
           `texto_mascarado` - Texto mascarada.
           `palavra_mascarada` - Palavra substituídas pela máscara.
           `token_predito` - Palavra prevista para a máscara.
           `peso_predito` - Peso da palavra prevista.
           `token_predito_marcado` - Token predito que foi marcado.
        '''
    
        # Somente modelos que possuem token de mascara.
        if self.getTransformer().getTokenMascara() != None:
            
            #print("texto:", texto)
            texto_mascarado, palavra_mascarada = self.getTextoMascarado(texto, classe=classe, qtde=qtde)
                    
            # Divide as palavras em tokens
            texto_tokenizado = self.getTransformer().getTextoTokenizado(texto_mascarado)
            #print("texto_tokenizado:", texto_tokenizado)

            # Retorna o índice da mascara de atenção
            mascara_atencao_indice = texto_tokenizado.index(self.getTransformer().getTokenMascara())
            #print("mascara_atencao_indice:", mascara_atencao_indice)

            # Mapeia os tokens em seus índices do vocabulário
            tokens_indexados =  self.getTokenizer().convert_tokens_to_ids(texto_tokenizado)
            #print("tokens_indexados:", tokens_indexados)
            
            # Converte as entradas de lista para tensores do torch
            tokens_tensores = torch.tensor([tokens_indexados])
            
            # Realiza a predição dos tokens
            with torch.no_grad():
                # Retorno de model quando ´output_hidden_states=True´ é setado:  
                #outputs[0] = last_hidden_state, outputs[1] = pooler_output, outputs[2] = hidden_states
                outputs =  self.getModel()(tokens_tensores)

            # Recupera a predição com os embeddings da última camada oculta    
            predicao = outputs[0]
            
            # Normaliza os pesos das predições nos embeddings e calcula sua probabilidade
            probabilidades = torch.nn.functional.softmax(predicao[0, mascara_atencao_indice], dim=-1)    
            
            # Retorna os k maiores elementos de determinado tensor de entrada ao longo de uma determinada 
            # dimensão de forma ordenada descrescentemente.    
            # Adiciona 20 elementos em topkpredicao para pular os tokens desconhecidos([UNK])
            MARGEM_UNK = 20
            top_k_predicao_pesos, top_k_predicao_indices = torch.topk(probabilidades, top_k_predicao + MARGEM_UNK, sorted=True)
            # print("top_k_predicao_pesos:",top_k_predicao_pesos)
            # print("top_k_predicao_indices:",top_k_predicao_indices)
            # print("len(top_k_predicao_indices):",len(top_k_predicao_indices))

            # Saída com as predições      
            saida = {}
            saida.update({'texto_mascarado' : [],
                          'palavra_mascarada' : [],
                          'token_predito' : [],
                          'token_peso' : [],
                          'token_predito_marcado': []})
            
            indice_token = 0
            for i, indice_predicao in enumerate(top_k_predicao_indices):

                # Mapeia os índices do vocabulário para os seus tokens
                token_predito =  self.getTokenizer().convert_ids_to_tokens([indice_predicao])[0]
                token_peso = top_k_predicao_pesos[i]

                # Pula o token se for desconhecido e existir tokens disponíveis
                if token_predito != self.getTransformer().getTokenDesconhecido() and indice_token < (top_k_predicao):
                
                    # Guarda o token original        
                    token_predito_marcado = token_predito
                
                    # Se o token tiver token separador
                    if (self.getTransformer().getSeparadorSubToken() != None) and (self.getTransformer().getSeparadorSubToken() in token_predito):
                        # Remove os caracteres SEPARADOR_SUBTOKEN do token
                        token_predito = token_predito.replace(self.getTransformer().getSeparadorSubToken(), "")

                    # Guarda o token
                    #lista_predicoes.append([indice_token, texto_mascarado, palavra_mascarada, token_predito, float(token_peso), token_predito_marcado])
                    saida['texto_mascarado'].append(texto_mascarado)
                    saida['palavra_mascarada'].append(palavra_mascarada)
                    saida['token_predito'].append(token_predito)
                    saida['token_peso'].append(float(token_peso))
                    saida['token_predito_marcado'].append(token_predito_marcado)
                    

                    # Incrementa para o próximo token
                    indice_token = indice_token + 1
            
            return saida

        else:
            logger.error("O modelo \"{}\" não possui um token de máscara.".format(self.getModel().__class__.__name__))
            
            return None
        
    # ============================
    def getModificacaoTextoSequencial(self, texto: str,
                                      classe: List[str] = ["VERB","NOUN","AUX"], 
                                      qtde: int = 1, 
                                      top_k_predicao: int = 10):

        ''' 
        Gera 'top_k_predicao' versões modificadas do texto, substituindo uma palavra por uma outra gerada pelo MLM do modelo.
        A quantidade de palavras no texto a serem modificadas é especificada por 'qtde'.
        A classe da palavra a ser modificada definida pelo parâmetro classe, que determina as classes morfossintática das palavras a serem selecionadas em ordem crescente. 
        Por exemplo: ["VERB","NOUN","AUX"], primeiro procura uma verbo, depois um substantivo e por último um auxiliar.
                           
        Parâmetros:
           `texto` - Texto a ser mascarado.
           `classe` - Lista com as classes morfossintática das palavras a serem mascarada com [MASK].
                      Valor default ["VERB","NOUN","AUX"], primeiro procura uma verbo, depois um substantivo e por último um auxiliar.
           `qtde` - Quantidade de palavras a serem substituidas pela máscara de acordo com a ordem das classes.
                    Seleciona aleatoriamente a(s) palavra(s) a ser(em) mascarada(s) se a qtde 
                    for menor que quantidade de palavras das classes no texto.
           `top_k_predicao` - Quantidade de palavras a serem recuperadas mais próximas da máscara.

        Retorno um lista de dicionários com as top_k previsões para cada palavra mascarada no texto:
            `texto_modificado` - Texto com a modificação.
            `texto_mascarado` - Texto mascarado.
            `palavra_mascarada` - Palavra substituídas pela máscara.
            `token_predito` - Palavra predita para a máscara.
            `token_peso` - Peso da palavra predita.
            `token_predito_marcado` - Token predito que foi marcado.
                
        '''
        
        # Somente modelos que possuem token de máscara.
        if self.getTransformer().getTokenMascara() != None:

            # Saída com as predições e substituições
            saida = {}
            saida.update({'texto_modificado' : [],
                          'texto_mascarado' : [],
                          'palavra_mascarada' : [],
                          'token_predito' : [],
                          'token_peso' : [],
                          'token_predito_marcado': []})
            
            # Recupera o texto mascarado e o token previsto
            predicao = self.getModificacaoPalavraTextoSequencial(texto=texto,
                                                                 classe=classe,
                                                                 qtde=qtde,
                                                                 top_k_predicao=top_k_predicao)            
            #print("predicao:",predicao)
            
            # Percorre a lista de predições e faz a substituicão da máscara pelo token previsto            
            for i, texto_mascarado in enumerate(predicao['texto_mascarado']):
            
                # Se existir o token especial [MASK] no texto marcado
                if  self.getTransformer().getTokenMascara() in texto_mascarado: 

                    # Substituir a máscara pelo token predito
                    texto_modificado = texto_mascarado.replace(self.getTransformer().getTokenMascara(), predicao['token_predito'][i])

                    # Guarda o registro do texto modificado
                    saida['texto_modificado'].append(texto_modificado)
                    saida['texto_mascarado'].append(texto_mascarado)
                    saida['palavra_mascarada'].append(predicao['palavra_mascarada'][i])
                    saida['token_predito'].append(predicao['token_predito'][i])
                    saida['token_peso'].append(predicao['token_peso'][i])
                    saida['token_predito_marcado'].append(predicao['token_predito_marcado'][i])
                    
                else:
                    logger.error("Não existe máscara no texto: \"{}\".".format(predicao['texto_mascarado'][i]))
            
            return saida
            
        else:
            logger.error("O modelo \"{}\" não possui um token de máscara.".format(self.getModel().__class__.__name__))
            
            return None
    
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