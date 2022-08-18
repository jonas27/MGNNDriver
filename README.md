# graphDriver


#
Should I remove the data source and only mention them below?

From Meeting:
 - For more material: try to use gene expression from emogi in combination with 12 for one cancer in features to see if it improves
 - Try use another graph conv layer for one cancer (maybe with attention)
 - Try use MLP for preprocessing x before graph conv layer from here https://proceedings.neurips.cc/paper/2020/hash/c5c3d4fe6b2cc463c7d7ecba17cc9de7-Abstract.html
 - Use batchnormalization --> was not good
 
## Print to terminal and file
python cmd/hpo.py -b |& tee screen.log

## Datasets
I deleted the tmp data and datasets and they are not downloadable on gdc anymore. They were commited here:
https://github.com/jonas27/graphDriver/tree/285e94e267c16a2f669dbf23839b9ea89f023aff

To load the files, you also need the code at that commit. If not un-pickle could/should fail

## Development

### Code Style
This module tries to follow best practices for coding style defined in [here](https://google.github.io/styleguide/pyguide.html).  
New code should always follow this guidline or clearly explain why breaking it is necessary.  
The code should be orientated towards [The Zen of Python](https://www.python.org/dev/peps/pep-0020/).

### Add module to development env
For best practices see [here](https://docs.pytest.org/en/6.2.x/goodpractices.html)
In repo root dir use cmd `pip install -e .` or `conda develop .` . (For more detail see [here](https://stackoverflow.com/questions/37006114/anaconda-permanently-include-external-packages-like-in-pythonpath).)

### Conda and Dependencies
The conda encivonrment can be install from the environment file.  
Update the environemnt.yml file with `conda env update --prefix ./env --file environment.yml  --prune`.  
For more see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#updating-an-environment)

#### Pip Disk Size Error
When using pip inside a conda env the downloaded files are stored in the default tmp dir. 
To change that use the env var `TMPDIR` like so `export TMPDIR="/scratch/tmp/"


## load
This is done per tumor type. All related scripts (bash and python) are inside `process`.
<img src="https://docs.google.com/drawings/d/1_dEDg8iABWEWkyjAR7wKZHrhrgYfe9FI443HbtlObxw/export/png" alt="image alt text">

### Download TCGA Data
The GDC [website](https://portal.gdc.cancer.gov/) provides metadata files, called manifests, to download large datasets.
For more detail on the cancer mutation genes see [here](https://docs.gdc.cancer.gov/Data/File_Formats/MAF_Format/)

### Hugo Names

## Labels: Validated negative and positive Cancer Driver Genes
CGC data from [COSMIC](https://cancer.sanger.ac.uk/cosmic/census?tier=1)

### Gene Similarity Network
Calculates the Pearson Correlation Coefficient (PCC) as a measure of similarity between genes. This results in the feature matrix <img src="https://latex.codecogs.com/gif.latex?\phi_{i}" title="\phi_{i}" /> for CNN.
In graphNN we use the PCC as the edge weight.


## GCN
Pytorch Geometric models and supported operations [here].(https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html)  



## Training
The training is done with 5 balanced datasets. Each has the full number of driver-genes and a random, non-repeating downsampled sample of the passenger genes. The actually training per dataset is done via 10-fold cross-validation.


## HP Optimization
The project uses ray tune as optimization library and ASHA as optimization algorithm.

### Tensorboard
```bash
tensorboard --logdir ~/ray_results
```
## Comparison
precision_recall_curve and roc

## Problems

### CPU: If CUDA makes weird things
export CUDA_VISIBLE_DEVICES=""

## Data Sources

### DigSee
http://210.107.182.61/geneSearch/




## Meeting 04/03/2022

2 genes network, one from healthy (ppi) and one from cancer patients (gene corr).
If genes are different in networks they are cancer genes, if they are similar then they are passenger genes. 


## Ranking of Candidates
1. Emogi 1. how do they form negatives 2. Ratio of positive to negative set 3. What are the  metrics to measure the performance.
2. Test should include all remaining passenger genes --> score for positives should be high and passengers low. --> ranking score aka: q-score
--> but still only include q-score
3. Can we really induce that a cancer driver genes for other cancer types is a candidate for other cancers. 
4. Increase imbalance ratio to model real scenario.

## Meeting

1. combine gene expression data for all patients across all cancers. Then make network based on pcc
2. Node features: average over node featres of single cancers.
3. Train on all cancer drivers. 10-fold

Make negative set.

Double check how many negatives they use in single cancer study.


### Taking Ranking of driver genes (report mean and std).
Relative index

### Look at top k genes and report how many drivers
Absolute index

### scRNA data from tcga or other source
try for one cancer type.

### Perform HP
AUC Score & abs and relative score

### TODO
Use any features I want
Perforamnce on train val and test --> are we overfitting underfitting. What should be done
Is score correct --> check code again
Use their features
What is emogi using in terms of imbalance factor

Play with architecture
Check overfitting
Check underfitting
Try undirected and directed
Regul
dropout
early stopping (10 epochs   )



fix ppi, use directed

mutation rate

decrease learning instead of early stopping

Pretrain on pancancer (all others)


================================================

#### Conf(cancer='brca', network_type=['ppi'], outer_fold=9, gcnk=0, directed_genes=True, directed_ppi=True, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=0, num_genes_nodes=0, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=2, num_ppi_nodes=64, dropout=0.1, imbalance_factor=2, n_outer_folds=0, total_inner_folds=2, budget=0)
*** test results for cancer brca nt ['ppi'] is 0.3821271742280383  ; val is 0.29286293912903444  ***

--> increase imb factor to 5
#### Conf(cancer='brca', network_type=['ppi'], outer_fold=9, gcnk=0, directed_genes=True, directed_ppi=True, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=0, num_genes_nodes=0, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=2, num_ppi_nodes=64, dropout=0.1, imbalance_factor=5, n_outer_folds=0, total_inner_folds=2, budget=0)
*** test results for cancer brca nt ['ppi'] is 0.3595590056594481  ; val is 0.3931508294585426  ***

--> imb factor 1
#### Conf(cancer='brca', network_type=['ppi'], outer_fold=9, gcnk=0, directed_genes=True, directed_ppi=True, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=0, num_genes_nodes=0, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=2, num_ppi_nodes=64, dropout=0.1, imbalance_factor=1, n_outer_folds=0, total_inner_folds=2, budget=0)
*** test results for cancer brca nt ['ppi'] is 0.36435307506149006  ; val is 0.27830504082695035  ***

--> imb factor 2 and dropout 0
#### Conf(cancer='brca', network_type=['ppi'], outer_fold=9, gcnk=0, directed_genes=True, directed_ppi=True, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=0, num_genes_nodes=0, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=2, num_ppi_nodes=64, dropout=0.0, imbalance_factor=2, n_outer_folds=0, total_inner_folds=2, budget=0)
*** test results for cancer brca nt ['ppi'] is 0.3170498013752617  ; val is 0.27951178854677783  ***

--> dropout 0.3
#### Conf(cancer='brca', network_type=['ppi'], outer_fold=9, gcnk=0, directed_genes=True, directed_ppi=True, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=0, num_genes_nodes=0, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=2, num_ppi_nodes=64, dropout=0.3, imbalance_factor=2, n_outer_folds=0, total_inner_folds=2, budget=0)
*** test results for cancer brca nt ['ppi'] is 0.37073756684039727  ; val is 0.3805641446924808  ***


# Genes
=========================================================
#### Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=4, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.1, imbalance_factor=2, n_outer_folds=0, total_inner_folds=10, budget=0)
*** test results for cancer brca nt ['genes'] is 0.07771469851902219  ; val is 0.10219122950133372  ***

--> gene_corr_factor=0.9
#### Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=4, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.1, imbalance_factor=2, n_outer_folds=0, total_inner_folds=10, budget=0)
*** test results for cancer brca nt ['genes'] is 0.0707421968893093  ; val is 0.08331537870875447  ***

--> gene_corr_factor=0.5, dropout=0.5
#### Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=4, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.5, imbalance_factor=2, n_outer_folds=0, total_inner_folds=10, budget=0)
*** test results for cancer brca nt ['genes'] is 0.08568034148006717  ; val is 0.07184248895126993  ***

--> imb factor=10
#### Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=4, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.5, imbalance_factor=10, n_outer_folds=0, total_inner_folds=10, budget=0)
*** test results for cancer brca nt ['genes'] is 0.09746636422687263  ; val is 0.22296044928755604  ***

--> reset && total inner 2
#### Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=4, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.1, imbalance_factor=2, n_outer_folds=0, total_inner_folds=2, budget=0)
*** test results for cancer brca nt ['genes'] is 0.05931817942433517  ; val is 0.04457689280934897  ***

--> gene corr factor 0.9
#### Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=4, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.1, imbalance_factor=2, n_outer_folds=0, total_inner_folds=2, budget=0)
*** test results for cancer brca nt ['genes'] is 0.049946400018789826  ; val is 0.042240016441041174  ***

--> lr 0.0001
#### Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=4, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, lr=0.0001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.1, imbalance_factor=2, n_outer_folds=0, total_inner_folds=2, budget=0)
*** test results for cancer brca nt ['genes'] is 0.0572398149470013  ; val is 0.04306483792410322  ***

--> lr 0.1
ERROR, all zeros

--> lr 0.01
#### Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=4, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, lr=0.01, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.1, imbalance_factor=2, n_outer_folds=0, total_inner_folds=2, budget=0)
*** test results for cancer brca nt ['genes'] is 0.14177377864922341  ; val is 0.08595176641688418  ***

--> gcn nodes 4 (ie 2**4=16)
#### Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=4, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, lr=0.01, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=16, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.1, imbalance_factor=2, n_outer_folds=0, total_inner_folds=2, budget=0)
*** test results for cancer brca nt ['genes'] is 0.1415591968363717  ; val is 0.08397926502791608  ***

--> gene layers 3
#### Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=4, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, lr=0.01, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=3, num_genes_nodes=16, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.1, imbalance_factor=2, n_outer_folds=0, total_inner_folds=2, budget=0)
*** test results for cancer brca nt ['genes'] is 0.0823886480346809  ; val is 0.061371380190350155  ***

--> gene layers 1
#### Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=4, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, lr=0.01, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=1, num_genes_nodes=16, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.1, imbalance_factor=2, n_outer_folds=0, total_inner_folds=2, budget=0)
*** test results for cancer brca nt ['genes'] is 0.09550106343706158  ; val is 0.08835552526179496  ***

--> gene layers 2, undirected
#### Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=4, gene_corr_factor=0.5, directed_genes=False, directed_ppi=True, lr=0.01, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=1, num_genes_nodes=16, num_linear_layers=6, num_linear_nodes=128, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.1, imbalance_factor=2, n_outer_folds=0, total_inner_folds=2, budget=0)
*** test results for cancer brca nt ['genes'] is 0.05493536672394036  ; val is 0.07693396710436981  ***



## new try
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=1, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=3, budget=0)
*** test results for cancer brca nt ['genes'] is 0.15237155726799037 ***

---> undirected network
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=False, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=1, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=3, budget=0)
*** test results for cancer brca nt ['genes'] is 0.11255414176191822 ***

---> directed network, gene layers=2 and aggr='mean'
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=3, budget=0)
*** test results for cancer brca nt ['genes'] is 0.21823805520940515 ***

---> aggr='add
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=3, budget=0)
*** test results for cancer brca nt ['genes'] is 0.053297260731466865 ***'

---> aggr='max'
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=3, budget=0)
*** test results for cancer brca nt ['genes'] is 0.06291146406576022 ***

---> aggr='mean'
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=3, budget=0)
*** test results for cancer brca nt ['genes'] is 0.17978894426670552 ***

---> genes_layers=3
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=3, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=3, budget=0)
*** test results for cancer brca nt ['genes'] is 0.08632474653079261 ***

---> don't use val -> train 200 epochs
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=3, budget=0)
*** test results for cancer brca nt ['genes'] is 0.2082541787555733 ***

---> rerun
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=3, budget=0)
*** test results for cancer brca nt ['genes'] is 0.17723501963884555 ***

---> 400 epochs
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=3, budget=0)
*** test results for cancer brca nt ['genes'] is 0.18100971461212007 ***

---> increase dropout in genes layers to 0.5 from 0.3
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=3, budget=0)
*** test results for cancer brca nt ['genes'] is 0.1564023702290195 ***

---> use val again
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=3, budget=0)
*** test results for cancer brca nt ['genes'] is 0.23134955020292738 ***

---> total inner = 8
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.264341464469619 ***

---> don't use dropout in genes layers
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.28637663677743963 ***

---> old setup genes layer -> concat x outs
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.24232083764673548 ***

---> new setup and use val pr auc score as validation metric
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.23823847605111542 and val score is 0.09880166988836185***

---> new setup no val and 1:1 imb
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.21229810543353586 and val score is nan***

---> rerun 
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.19496252246535914 and val score is nan***

---> with val and same imb factor for train and val = 2
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.34440718306048135 and val score is nan***

---> rerun
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.25976867171625373 and val score is nan***

---> rerun
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=2, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.2864426822536873 and val score is nan***

---> imb factor =1
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=1, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.28167537187136577 and val score is nan***

---> rerun 
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=1, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.22175251553454195 and val score is 0.7986190476190476***

---> with gradient clipping
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=3, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.3637554321854234 and val score is 0.5174561606693959***

---> rerun
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=3, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.33489258125118065 and val score is 0.588543387168387***

---> rerun (use new model just to make sure its correct)
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=3, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.32162381704009146 and val score is 0.5458840299277605***

---> roc star loss
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=3, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.3170464745259677 and val score is 0.46643203202026734***

---> rerun 
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=3, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.2590767641554572 and val score is 0.5058276630490253***

---> remove star loss 
Conf(cancer='brca', network_type=['genes'], outer_fold=9, gcnk=7, min_genes_edges=3, gene_corr_factor=0.5, directed_genes=True, directed_ppi=True, use_genes_attr=False, lr=0.001, optimizer='AdamW', conv_layer='GraphConv', num_genes_layers=2, num_genes_nodes=64, num_linear_layers=3, num_linear_nodes=64, num_ppi_layers=0, num_ppi_nodes=0, dropout=0.2, imbalance_factor=3, imbalance_factor_val=0, n_outer_folds=0, total_inner_folds=8, budget=0)
*** test results for cancer brca nt ['genes'] is 0.3515771015447176 and val score is 0.5666725177986788***