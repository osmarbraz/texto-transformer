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
def similaridadeCosseno(a: torch.Tensor, b: torch.Tensor):
   '''
   Calcula a similaridade do cosseno de dois vetores(embeddgins de 1 dimensão).
    
   Parâmetros:
      `a` - Um tensor na forma de vetor.
      `b` - Um tensor na forma de vetor.

   Retorno:
      A similaridade do cosseno entre os vetores.
   '''
    
   if not isinstance(a, torch.Tensor):      
      a = torch.tensor(a)

   if not isinstance(b, torch.Tensor):
      b = torch.tensor(b)
    
   similaridade = 1 - cosine(a, b)
    
   return similaridade

def similaridadeCossenoMatriz(a: torch.Tensor, b: torch.Tensor):
   '''
   Calcula a similaridade do cosseno entre duas matrizes com cos_sim(a[i], b[j]) para todo i e j.
    
   Parâmetros:
      `a` - Um tensor de forma (N, D)
      `b` - Um tensor de forma (M, D)    
    
   Retorno: 
      Uma matriz com res[i][j] = cos_sim(a[i], b[j])
   '''
   if not isinstance(a, torch.Tensor):
      a = torch.tensor(a)

   if not isinstance(b, torch.Tensor):
      b = torch.tensor(b)

   # Se for uma matriz de 1 dimensão
   if len(a.shape) == 1:
      # Adiciona uma dimensão
      a = a.unsqueeze(0)
   
   # Se for uma matriz de 1 dimensão
   if len(b.shape) == 1:
      # Adiciona uma dimensão
      b = b.unsqueeze(0)

   # Normaliza os vetores deixa os valores entre 0 e 1
   a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
   b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
   
   # Realiza o produto escalar entre os vetores
   sim = torch.mm(a_norm, b_norm.transpose(0, 1))
    
   # Se for uma matriz de 1 dimensão
   if sim.shape[0] == 1 and sim.shape[1] == 1:
      return sim.item()
   else: 
      return sim.numpy()

# ============================  
def distanciaEuclidiana(a: torch.Tensor, b: torch.Tensor):
   '''
   Calcula a distância euclidiana de dois vetores(embeddgins de 1 dimensão).
   Possui outros nomes como distância L2 ou norma L2.
    
   Parâmetros:
      `a` - Um tensor na forma de vetor.         
      `b` - Um tensor na forma de vetor.

   Retorno:
       A distância euclidiana entre os vetores.
   '''
   
   if not isinstance(a, torch.Tensor):      
      a = torch.tensor(a)

   if not isinstance(b, torch.Tensor):
      b = torch.tensor(b)
       
   distancia = euclidean(a, b)
    
   return distancia

# ============================  
def distanciaManhattan(a: torch.Tensor, b: torch.Tensor):
   '''
   Calcula a distância Manhattan de dois vetores(embeddgins de 1 dimensão).
   Possui outros nomes como distância Cityblock, distância L1, norma L1 e métrica do táxi.
    
   Parâmetros:
      `a` - Um tensor na forma de vetor.         
      `a` - Um tensor na forma de vetor.

   Retorno:
      A distância Manhattan entre os vetores.
   '''
   
   if not isinstance(a, torch.Tensor):      
      a = torch.tensor(a)

   if not isinstance(b, torch.Tensor):
      b = torch.tensor(b)
    
   distancia = cityblock(a, b)

   return distancia
