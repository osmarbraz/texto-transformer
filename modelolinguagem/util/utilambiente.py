# Import das bibliotecas.
import logging  # Biblioteca de logging
import subprocess

def installspacy(modelo_args):
    install_setuptools()
    install_spacy(modelo_args)
    install_model_spacy(modelo_args)    
    logging.info("Instalação spacy versão {} realizada!".format(modelo_args.versao_spacy)) 


def install_setuptools():
    try:
        subprocess.run(["pip", "-U", "install", "pip","setuptools", "wheel"])
        logging.info("setuptools instalado!")    
    except subprocess.CalledProcessError as e:
        logging.info("Falha em instalar setuptools. Erro: {}.".format(e))


def install_spacy(modelo_args):
    try:
        subprocess.run(["pip", "-U", "install", "spacy={modelo_args.versao_spacy}"])
        logging.info("spaCy versão {} instalado!".format(modelo_args.versao_spacy))    
    except subprocess.CalledProcessError as e:
        logging.info("Falha em instalar spaCy versão {}. Erro: {}.".format(modelo_args.versao_spacy, e))    

def install_model_spacy(modelo_args):
    try:
        # Download do modelo de linguagem na linguagem solicitada
        subprocess.run(["python", "-m", "spacy", "download", modelo_args.modelo_spacy])
        logging.info("Download do modelo {} realizado!".format(modelo_args.modelo_spacy))    
    except subprocess.CalledProcessError as e:
        logging.info("Falha em instalar modelo spaCy {}. Erro: {}.".format(modelo_args.modelo_spacy, e))
