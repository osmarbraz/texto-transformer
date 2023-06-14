# Import das bibliotecas.
import logging  # Biblioteca de logging
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cityblock

# ============================  
def similaridadeCoseno(texto1, texto2):
    '''
    Similaridade do cosseno dos embeddgins dos textos.
    
    Parâmetros:
    `texto1` - Um texto a ser medido.           
    `texto2` - Um texto a ser medido.                 
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
    '''
    
    distancia = cityblock(texto1, texto2)

    return distancia
