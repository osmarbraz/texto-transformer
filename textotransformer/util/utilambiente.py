# Import das bibliotecas.

# Biblioteca de logging
import logging  
import subprocess

logger = logging.getLogger(__name__)

class InstaladorModelo:
    ''' 
    Realiza a instalação do modelo spaCy.
     
    Parâmetros:
    `nome_pln` - argumentos passados para o instalador.
    ''' 
    def __init__(self, modelo_args):
        #Atualiza os parâmetros
        self.modelo_args = modelo_args
        
        # Executa o processo de atualização e instalação do spaCy
        self.install_model_spacy()

    # ============================
    def install_model_spacy(self):
        try:
            # Download do modelo de linguagem na linguagem solicitada
            subprocess.run(["python", "-m", "spacy", "download", self.modelo_args.modelo_spacy])
            logger.info("Download do modelo spaCy {} realizado!".format(self.modelo_args.modelo_spacy))    
        except subprocess.CalledProcessError as e:
            logger.info("Falha em instalar modelo spaCy {}. Erro: {}.".format(self.modelo_args.modelo_spacy, e))

