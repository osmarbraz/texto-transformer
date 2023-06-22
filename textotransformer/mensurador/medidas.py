# Import das bibliotecas.

# Biblioteca de logging
import logging  
# Biblioteca de aprendizado de máquina
import torch

# Biblioteca de cálculos de distância
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock

logger = logging.getLogger(__name__)

# ============================  
def similaridadeCoseno(texto1, texto2):
    '''
    Similaridade do cosseno dos embeddgins dos textos.
    
    Parâmetros:
    `texto1` - Um texto a ser medido.           
    `texto2` - Um texto a ser medido. 

    Retorno:
    A similaridade do cosseno entre os textos.                
    '''
    
    similaridade = 1 - cosine(texto1, texto2)
    
    return similaridade

def PytorchSimilaridadeCoseno(a: torch.Tensor, b: torch.Tensor):
    """
    Calcula a similaridade do cosseno cos_sim(a[i], b[j]) para todo i e j.
    :Retorna: Uma matriz com res[i][j] = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)

    return torch.mm(a_norm, b_norm.transpose(0, 1))

# ============================  
def similaridadeCoseno(texto1, texto2):
    '''
    Similaridade do cosseno dos embeddgins dos textos.
    
    Parâmetros:
    `texto1` - Um texto a ser medido.           
    `texto2` - Um texto a ser medido.

    Retorno:
    A similaridade do cosseno entre os textos.
    '''
    
    similaridade = 1 - cosine(texto1, texto2)
    
    return similaridade

# ============================  
def distanciaEuclidiana(texto1, texto2):
    '''
    Distância euclidiana entre os embeddings dos textos.
    Possui outros nomes como distância L2 ou norma L2.
    
    Parâmetros:
    `texto1` - Um texto a ser medido.           
    `texto2` - Um texto a ser medido.

    Retorno:
    A distância euclidiana entre os textos.
    '''
    
    distancia = euclidean(texto1, texto2)
    
    return distancia

# ============================  
def distanciaManhattan(texto1, texto2):
    '''
    Distância Manhattan entre os embeddings das sentenças. 
    Possui outros nomes como distância Cityblock, distância L1, norma L1 e métrica do táxi.
    
    Parâmetros:
    `texto1` - Um texto a ser medido.           
    `texto2` - Um texto a ser medido.

    Retorno:
    A distância Manhattan entre os textos.
    '''
    
    distancia = cityblock(texto1, texto2)

    return distancia
