# Import das bibliotecas.

 # Biblioteca de logging
import logging 

 # Biblioteca de dataclasses
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

logger = logging.getLogger(__name__)

# ============================
@dataclass
class ModeloArgumentos:        
    '''
    Classe(ModeloArgumentos) de definição dos parâmetros do modelo de linguagem.
    '''
 
    max_seq_len: Optional[int] = field(
                                       default=None,
                                       metadata={'help': 'max seq len'},
                                       )    
    pretrained_model_name_or_path: str = field(
                                               default='neuralmind/bert-base-portuguese-cased',
                                               metadata={'help': 'nome do modelo pré-treinado.'},
                                               )
    modelo_spacy: str = field(
                              default='pt_core_news_lg',
                              metadata={'help': 'nome do modelo do spaCy.'},
                              )  
               
    do_lower_case: bool = field(
                                default=False,
                                metadata={'help': 'define se o texto do modelo deve ser todo em minúsculo.'},
                                )  
    output_attentions: bool = field(
                                    default=False,
                                    metadata={'help': 'habilita se o modelo retorna os pesos de atenção.'},
                                    )
    output_hidden_states: bool = field(
                                       default=False,
                                       metadata={'help': 'habilita gerar as camadas ocultas do modelo.'},
                                       )      
    abordagem_extracao_embeddings_camadas: int = field(
                                    default=2, # 0-Primeira/1-Penúltima/2-Ùltima/3-Soma 4 últimas/4-Concat 4 últimas/5-Soma de todas
                                    metadata={'help': 'Especifica a abordagem para a extracao dos embeddings das camadas do transformer.'},
                                    )
    estrategia_pooling: int = field(
                                    default=0, # 0 - MEAN estratégia de pooling média / 1 - MAX  estratégia de pooling maior
                                    metadata={'help': 'Estratégia de pooling de padronização do embeddings das= palavras das sentenças.'},
                                    )
    palavra_relevante: int = field(
                                   default=0, # 0 - ALL Considera todas as palavras das sentenças / 1 - CLEAN desconsidera as stopwords / 2 - NOUN considera somente as palavras substantivas
                                   metadata={'help': 'Estratégia de relevância das palavras das sentenças para gerar os embeddings.'},
                                   )

