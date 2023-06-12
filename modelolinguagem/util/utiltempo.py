# Import das bibliotecas.
import logging  # Biblioteca de logging
from cmath import rect, phase
from math import radians, degrees
import datetime # Biblioteca de data e tempo
 
# ============================    
def formataTempo(tempo):
    '''
    Pega a tempo em segundos e retorna uma string hh:mm:ss
    '''
        
    # Arredonda para o segundo mais próximo.
    tempoArredondado = int(round((tempo)))
   
    # Formata como hh:mm:ss
    return str(datetime.timedelta(seconds=tempoArredondado))  

# ============================      
def mediaAngulo(deg):
    
    return degrees(phase(sum(rect(1, radians(d)) for d in deg) / len(deg)))
 
# ============================  
def mediaTempo(tempos):
    '''
    Calcula a média de uma lista de tempo string no formato hh:mm:ss
    '''
    
    t = (tempo.split(':') for tempo in tempos)
    # Converte para segundos
    segundos = ((float(s) + int(m) * 60 + int(h) * 3600) for h, m, s in t)
    # Verifica se deu algum dia
    dia = 24 * 60 * 60
    # Converte para angulos
    paraAngulos = [s * 360. / dia for s in segundos]
    # Calcula a média dos angulos
    mediaComoAngulo = mediaAngulo(paraAngulos)
    media_segundos = mediaComoAngulo * dia / 360.
    if media_segundos < 0:
        media_segundos += dia
    # Recupera as horas e os minutos  
    h, m = divmod(media_segundos, 3600)
    # Recupera os minutos e os segundos
    m, s = divmod(m, 60)  
    
    return '{:02d}:{:02d}:{:02d}'.format(int(h), int(m), int(s))    

# ============================    
def somaTempo(tempos):
    '''
    Calcula a soma de uma lista de tempo string no formato hh:mm:ss
    '''
    
    t = (tempo.split(':') for tempo in tempos)
    # Converte para segundos
    segundos = ((float(s) + int(m) * 60 + int(h) * 3600) for h, m, s in t)
    # Soma os segundos
    soma_segundos = sum([s * 1. for s in segundos])
    # Recupera as horas e os minutos   
    h, m = divmod(soma_segundos, 3600)
    # Recupera os minutos e os segundos
    m, s = divmod(m, 60) 
    
    return '{:02d}:{:02d}:{:02d}'.format(int(h), int(m), int(s))  
