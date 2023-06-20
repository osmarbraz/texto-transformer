# Texto Transformer: Framework multilingual para processamento de textos utilizando modelos de linguagem baseados baseados em Transformer

Gera embeddings de textos, sentenças, palavras e tokens utilizando modelos contextualizados de linguagem multilingual baseados em Transformer.

## Instalação

Para instalar o pacote, basta executar o comando abaixo:

<pre><code>$ pip install texto-transformer</code></pre>

## Uso

Exemplo simples de uso do pacote:

````python
from textotransformer import TextoTransformer

modelo = TextoTransformer("neuralmind/bert-base-portuguese-cased")
````

## Dependências
- transformers==4.26.1
- spacy==3.5.2
- tqdm==4.65.0

## Licença

Esse projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.