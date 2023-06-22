# Texto-Transformer: Framework para processamento de textos utilizando modelos de linguagem baseados baseados em Transformer

Este framework realiza o processamento de textos utilizando modelos de linguagem baseados em transformer. Permite gerar embeddings de textos, sentenças, palavras e tokens utilizando modelos contextualizados de linguagem  baseados em Transformer. Os embeddings de textos, sentenças e palavras podem ser consolidados utilizando as estratégias de pooling média e máximo dos tokens.

## Instalação

Recomendamos **Python 3.6**, **[PyTorch 1.6.0](https://pytorch.org/get-started/locally/)** e **[transformers 4.26.1](https://github.com/huggingface/transformers)** e **[spaCy 3.5.2](https://spacy.io)**. 

**Instalação com pip**

Para instalar o pacote utilizando o **pip**, basta executar o comando abaixo:

<pre><code>$ pip install texto-transformer</code></pre>

**Instalação dos fontes**

Você também pode clonar a versão mais recente do [repositório](https://github.com/osmarbraz/texto-transformer.git) e instalá-la diretamente do código-fonte:

<pre><code>$ pip install -e .</code></pre>

O comando deve ser executado no diretório onde foi realizado o download do repositório.

## Exemplos 

### Uso simples

````python
# Importa a classe
from textotransformer import TextoTransformer

# Instância uma objeto e baixa o modelo de linguagem
modelo = TextoTransformer("neuralmind/bert-base-portuguese-cased")

# Alguns textos a serem codificados
textos = ["Bom Dia, professor.",
          "Qual o conteúdo da prova?",
          "Vai cair tudo na prova?",
          "Aguardo uma resposta, João."]

# Recupera os embeddings consolidados dos textos
embeddings_texto = modelo.getEmbeddingTexto(textos)      

# Mostra os textos e seus embeddings
for texto, embedding in zip(textos, embeddings_texto):
    print("Texto:", texto)
    print("Embedding:", embedding)
    print("")

#Resultado
#Texto: Bom Dia, professor.
#Embedding: tensor([ 1.3736e-01,  6.1996e-02,  3.2554e-01, -3.1146e-02,  3.5892e-01,...
#Texto: Qual o conteúdo da prova?
#Embedding: tensor([ 8.3348e-02, -1.8269e-01,  5.9241e-01, -9.5235e-02,  5.0978e-01,...
#Texto: Vai cair tudo na prova?
#Embedding: tensor([ 1.3447e-01,  1.1854e-01,  6.0201e-02,  1.0271e-01,  2.6321e-01,...
#Texto: Aguardo uma resposta, João.
#Embedding: tensor([ 3.7160e-02, -7.3645e-02,  3.3942e-01,  8.0847e-02,  3.8259e-01,...
````

### Recuperando embeddings diversos

````python
# Importa a classe
from textotransformer import TextoTransformer

# Instância uma objeto e baixa o modelo de linguagem
modelo = TextoTransformer("neuralmind/bert-base-portuguese-cased")

texto = "Você gosta de sorvete de manga? Sim, adoro muito."

# Recupera os embeddings consolidados do texto
embeddings_texto = modelo.getEmbeddingTexto(texto)
print("Um texto de tamanho      :",len(embeddings_texto))

# Recupera os embeddings consolidados das sentenças do texto
embeddings_sentenca = modelo.getEmbeddingSentenca(texto)
print("\nQuantidade de sentenças  :",len(embeddings_sentenca))
print("Cada sentença de tamanho :",len(embeddings_sentenca[0]))

# Recupera os embeddings consolidados das palavras do texto
embeddings_palavra = modelo.getEmbeddingPalavra(texto)
print("\nQuantidade de palavras   :",len(embeddings_palavra))
print("Cada palavra de tamanho  :",len(embeddings_palavra[0]))

# Recupera os embeddings dos tokens do texto
embeddings_token = modelo.getEmbeddingToken(texto)
print("\nQuantidade de tokens     :",len(embeddings_token))
print("Cada token de tamanho    :",len(embeddings_token[0]))

#Resultado
#Um texto de tamanho      : 768
#Quantidade de sentenças  : 2
#Cada sentença de tamanho : 768
#Quantidade de palavras   : 12
#Cada palavra de tamanho  : 768
#Quantidade de tokens     : 15
#Cada token de tamanho    : 768
````

Os exemplos pode ser executado no notebook do GoogleColab [ExemplosTextoTransformer.ipynb](https://github.com/osmarbraz/texto-transformer/blob/main/notebooks/ExemplosTextoTransformer.ipynb).

## Métodos principais:

Métodos principais para recuperar embeddings de textos, sentenças, palavras e tokens. Os embeddings de textos, sentenças e palavras podem ser consolidados pelas estratégias de pooling média (MEAN) e máximo (MAX) dos embeddings de seus tokens.

- `getEmbeddingTexto(texto: Union[str, List[str]], estrategia_pooling: int)`
    - Retorna uma lista dos embeddings consolidados dos textos.
    - Parâmetros:
        - `texto`: Um texto ou uma lista de textos para obter os embeddings.
        - `estrategia_pooling`: Especifica a estratégia de pooling dos tokens do texto. Valores possívels 0 - MEAN ou 1 - MAX. Valor default 0(MEAN).

- `getEmbeddingSentenca(texto: Union[str, List[str]], estrategia_pooling: int)` 
    - Retorna uma lista dos embeddings consolidados das sentenças dos textos.    
    - Parâmetros:
        - `texto`: Um texto ou uma lista de textos para obter os embeddings.
        - `estrategia_pooling`: Especifica a estratégia de pooling dos tokens do texto. Valores possívels 0 - MEAN ou 1 - MAX. Valor default 0(MEAN).

- `getEmbeddingPalavra(texto: Union[str, List[str]], estrategia_pooling: int)` 
    - Retorna uma lista dos embeddings consolidados das palavras dos textos.
    - Parâmetros:
        - `texto`: Um texto ou uma lista de textos para obter os embeddings.
        - `estrategia_pooling`: Especifica a estratégia de pooling dos tokens do texto. Valores possívels 0 - MEAN ou 1 - MAX. Valor default 0(MEAN).

- `getEmbeddingToken(texto: Union[str, List[str]])` 
    - Retorna uma lista dos embeddings dos tokens dos textos.
    - Parâmetros:
        - `texto`: Um texto ou uma lista de textos para obter os embeddings.        

## Modelos Pré-treinados

A lista completa dos modelos de linguagem pré-treiandos podem ser consultados no site da [Huggingface](https://huggingface.co/models).

## Dependências

- transformers==4.26.1
- spacy==3.5.2
- tqdm==4.65.0

## Licença

Esse projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.