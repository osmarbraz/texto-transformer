from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="modelolinguagem",
    version="1.0.2",
    author="Osmar de Oliveira Braz Junior",
    author_email="osmar.braz@udesc.br",
    description="Multilingual text embeddings",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",    
    download_url="https://github.com/osmarbraz/modelolinguagem/",
    packages=find_packages(),
    python_requires=">=3.6.0",
    install_requires=[
        'transformers==4.26.1',
        'spacy==3.5.2'
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    keywords="Transformer embedding texto sentença palavra token"
)