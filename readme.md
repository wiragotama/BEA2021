# About
This repository contain the accompanying resources for [my BEA2021 paper](https://www.aclweb.org/anthology/2021.bea-1.10/).

> Jan Wira Gotama Putra, Simone Teufel, Takenobu Tokunaga. 2021. Parsing Argumentative Structure in English-as-Foreign-Language Essays. In the Proceedings of the Sixteenth Workshop on Innovative Use of NLP for Building Educational Applications, Association for Computational Linguistics, pages 97-109. 20 April 2021. 

# Dataset
Please check the [ICNALE-AS2R](https://github.com/wiragotama/ICNALE-AS2R) page for downloading the corpus.

# Data preparation
- ```data/``` folder contains the file structure to prepare the dataset
- ```data/dataset/``` folder contains the ICNALE dataset that has been annotated with argumentative structure and sentence reordering (see the download link above).
- ```data/dataset-SBERT/``` shows the folder organisation when preprocessing the data, i.e., converting sentences into vector forms using SBERT encoder.

# System
- Check the ```docker/``` folder to replicate our environment.
- The parsing process is divided into two stages: (1) sentence linking and (2) relation labelling. ```sentence linking/``` and ```rel labelling/``` folders contain the corresponding scripts. 
- Please refer to our paper for the experimental details. 



