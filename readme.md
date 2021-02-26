# About
This repository contain the accompanying resources for <span style="color:blue">[my BEA2021 paper]</span>.

> Jan Wira Gotama Putra, Simone Teufel, Takenobu Tokunaga. 2021. Parsing Argumentative Structure in English-as-Foreign-Language Essays. In the Proceedings of the Sixteenth Workshop on Innovative Use of NLP for Building Educational Applications, pages XX-YY. April XX 2021. Association for Computational Linguistics.  

# Dataset
- You can download the dataset at <span style="color:blue">[link to be prepared]</span>

# Data preparation
- ```data/``` folder contains the file structure to prepare the dataset
- ```data/dataset/``` folder contains the ICNALE dataset that has been annotated with argumentative structure and sentence reordering (see the download link above).
- ```data/dataset-SBERT/``` shows the folder organisation when preprocessing the data, i.e., converting sentences into vector forms using SBERT encoder.

# System
- Check the ```docker/``` folder to replicate our environment.
- The parsing process is divided into two stages: (1) sentence linking and (2) relation labelling. ```sentence linking/``` and ```rel labelling/``` folders contain the corresponding scripts. 
- Please refer to our paper for the experimental details. 



