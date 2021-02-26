# Preprocessing

### Directory structure

| Folder / file | Description |
|---| ----|
|```BERTencoder.py``` | To encode a text into a vectors |
|```BERTserver.py``` | For encoder |
|```common_functions.py``` | Common functions shared across the preprocessing scripts |
|```corpus_stats_ICNALE.py``` | Corpus statistics if ICNALE dataset |
|```discourseunit.py``` | Data structure for essay |
|```icnale_to_tsv.py``` | Convert annotated ICNALE essays to ```.tsv``` files |
|```tsv_to_vector.py``` | Convert essays (in ```.tsv```) as a sequence of sentence vectors. There are three output files for each essay: ```.vectors```, ```.rel_distances``` and ```.rel_labels``` |