# Import das bibliotecas.

# Biblioteca de logging
import logging  
import subprocess

# Bibliotecas próprias
from textotransformer.modelo.modeloarguments import ModeloArgumentos

logger = logging.getLogger(__name__)

class InstaladorModelo:
    ''' 
    Realiza a instalação do modelo spaCy.
     
    Parâmetros:
       `model_args` - Um objeto com os parâmetros do modelo.
    ''' 
    
    def __init__(self, model_args: ModeloArgumentos):
        #Atualiza os parâmetros
        self.model_args = model_args
        
        # Executa o processo de atualização e instalação do spaCy
        self.install_model_spacy()

    # ============================
    def __repr__(self):
        '''
        Retorna uma string com descrição do objeto
        '''

        return "Classe (\"{}\") de instalação do modelo \"{}\".".format(self.__class__.__name__,
                                                                        self.model_args.modelo_spacy.__class__.__name__)

    # ============================
    def install_model_spacy(self):
        '''
        Realiza a instalação do modelo spaCy.
        '''

        try:
            # Download do modelo de linguagem na linguagem solicitada
            subprocess.run(["python", "-m", "spacy", "download", self.model_args.modelo_spacy])
            logger.info("Download do modelo spaCy \"{}\" realizado!".format(self.model_args.modelo_spacy))
        except subprocess.CalledProcessError as e:
            logger.info("Falha em instalar modelo spaCy \"{}\". Erro: {}.".format(self.model_args.modelo_spacy, e))

