# Import das bibliotecas.
import logging  # Biblioteca de logging
import subprocess

class InstaladorSpacy:

    def __init__(self, modelo_args):
        #Atualiza os parâmetros
        self.modelo_args = modelo_args
        
        # Executa o processo de atualização e instalação do spaCy
        self.installspacy()

    def installspacy(self):
        self.install_setuptools()
        self.install_spacy()
        self.install_model_spacy()    
        self.logging.info("Instalação spacy versão {} realizada!".format(self.modelo_args.versao_spacy)) 


    def install_setuptools(self):
        try:
            subprocess.run(["pip", "-U", "install", "pip","setuptools", "wheel"])
            logging.info("Atualizado as instalações do pip, setuptools e wheell!")    
        except subprocess.CalledProcessError as e:
            logging.info("Falha em instalar setuptools. Erro: {}.".format(e))


    def install_spacy(self):
        try:
            subprocess.run(["pip", "install", "spacy={self.modelo_args.versao_spacy}"])
            logging.info("spaCy versão {} instalado!".format(self.modelo_args.versao_spacy))    
        except subprocess.CalledProcessError as e:
            logging.info("Falha em instalar spaCy versão {}. Erro: {}.".format(self.modelo_args.versao_spacy, e))    

    def install_model_spacy(self):
        try:
            # Download do modelo de linguagem na linguagem solicitada
            subprocess.run(["python", "-m", "spacy", "download", self.modelo_args.modelo_spacy])
            logging.info("Download do modelo {} realizado!".format(self.modelo_args.modelo_spacy))    
        except subprocess.CalledProcessError as e:
            logging.info("Falha em instalar modelo spaCy {}. Erro: {}.".format(self.modelo_args.modelo_spacy, e))
