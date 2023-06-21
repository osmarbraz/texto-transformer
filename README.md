# Texto Transformer: Framework multilingual para processamento de textos utilizando modelos de linguagem baseados baseados em Transformer

Gera embeddings de textos, sentenças, palavras e tokens utilizando modelos contextualizados de linguagem multilingual baseados em Transformer.

## Instalação

Recomendamos **Python 3.6**, **[PyTorch 1.6.0](https://pytorch.org/get-started/locally/)** e **[transformers 4.26.1](https://github.com/huggingface/transformers)** e *[spaCy 3.5.2](https://spacy.io)**. 

**Instalação com pip**

Para instalar o pacote utilizando o **pip**, basta executar o comando abaixo:

<pre><code>$ pip install texto-transformer</code></pre>


**Instalação dos fontes**

Para instalar o pacote utilizando o **pip**, basta executar o comando abaixo:
Você também pode clonar a versão mais recente do [repositório](https://github.com/osmarbraz/texto-transformer.git) e instalá-la diretamente do código-fonte:

<pre><code>$ pip install -e .</code></pre>

## Exemplo simples de uso

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
for texto, embedding in zip(texto, embeddings_texto):
    print("Texto:", texto)
    print("Embedding:", embedding)
    print("")

````

## Métodos principais:

Métodos principais para recuperar embeddings de textos, sentenças, palavras e tokens. Os embeddings de textos, sentenças e palavras podem ser consolidados pelas estratégias de pooling média (MEAN) e máximo (MAX) dos embeddings de seys tokens.

- `getEmbeddingTexto(texto: Union[str, List[str]], estrategiaPooling)`
    - Retorna uma lista dos embeddings consolidados dos textos.
    - Parâmetros:
        - `texto`: Um texto ou uma lista de textos para obter os embedings.
        - `estrategiaPooling`: Especifica a estratégia de pooling dos tokens do texto. Valores possívels 0 - MEAN ou 1 - MAX. Valor default 0(MEAN).

- `getEmbeddingSentenca(texto: Union[str, List[str]], estrategiaPooling)` 
    - Retorna uma lista dos embeddings consolidados das sentenças dos textos.    
    - Parâmetros:
        - `texto`: Um texto ou uma lista de textos para obter os embedings.
        - `estrategiaPooling`: Especifica a estratégia de pooling dos tokens do texto. Valores possívels 0 - MEAN ou 1 - MAX. Valor default 0(MEAN).

- `getEmbeddingPalavra(texto: Union[str, List[str]], estrategiaPooling)` 
    - Retorna uma lista dos embeddings consolidados das palavras dos textos.
    - Parâmetros:
        - `texto`: Um texto ou uma lista de textos para obter os embedings.
        - `estrategiaPooling`: Especifica a estratégia de pooling dos tokens do texto. Valores possívels 0 - MEAN ou 1 - MAX. Valor default 0(MEAN).

- `getEmbeddingToken(texto: Union[str, List[str]])` 
    - Retorna uma lista dos embeddings dos tokens dos textos.
    - Parâmetros:
        - `texto`: Um texto ou uma lista de textos para obter os embedings.        

## Modelos Pré-treinados

- [Lista dos modelos pré-treinados](https://huggingface.co/models)

## Dependências
- transformers==4.26.1
- spacy==3.5.2
- tqdm==4.65.0

## Licença

Esse projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.