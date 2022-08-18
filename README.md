# MGNNdriver



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

