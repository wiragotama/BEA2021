# Preprocessing

### Directory structure

| Folder / file | Description |
|---| ----|
|```BERTencoder.py``` | To encode a text into a vectors using BERT |
|```BERTserver.py``` |  |
|```SBERTencoder.py``` | To encode a text into a vectors using SBERT |
|```common_functions.py``` | Common functions shared across the preprocessing scripts |
|```corpus_stats_ICNALE.py``` | Corpus statistics if ICNALE dataset |
|```discourseunit.py``` | Data structure for essay |
|```generate_K_fold.py``` | To split dataset for cross-validation experiment |
|```icnale_to_tsv.py``` | Convert annotated ICNALE essays to ```.tsv``` files |
|```treebuilder.py```| Script to build a tree from linking predictions |
|```tsv_to_vector.py``` | Convert essays as a sequence of sentence vectors. There are many output files for each essay. Please read the script for details. |