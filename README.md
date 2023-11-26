# SpaGFT modified CAMPA

CAMPA (Conditional Autoencoder for Multiplexed Pixel Analysis) is a framework for quantiative analysis of subcellular multi-channel imaging data.
We use SpaGFT to modify it and achieve a better performance.


## How to install CAMPA_SpaGFT?

The virtual environment is recommended before installing CAMPA and SpaGFT. Users can configure the virtual environment by anaconda. 
SpaGCN_SpaGFT is based on CAMPA, however the environments of CAMPA and SpaGFT are not compatible in one conda virtual environment, so you need create two virtual environments and keep the environment path of SpaGFT, which will be used in following Modified-CAMPA training process.

### Install SpaGFT

```bash
conda create -n spagft_env python==3.8.0
conda activate spagft_env
pip install SpaGFT
```

Find path of `spagft_env`
```bash
conda env list
```

### Install Modified CAMPA requirements

CAMPA was developed for Python 3.9 and it envirment can be installed from `requirements.txt`:
```bash
conda deactivate
conda create -n modified_campa_env python==3.9
git clone https://github.com/jiangyi01/SpaGFTModifiedCAMPA.git
cd SpaGFTModifiedCAMPA
# when running CAMPA_SpaGFT, the environment is required
conda activate modified_campa_env 
pip install -r requirements.txt
```
We recommend [jupyter](https://jupyter.org/) for interactive usage under `modified_campa_env`. It can be installed and configured by
```bash
conda install jupyter
python -m ipykernel install --user --name=modified_campa_env --display-name=modified_campa_env
```


## Usage and analysis:

The training and predicting process and relevant analysis are in the `SpaGFT_modified_training_cluster.ipynb`
- Download and initialize datasets
- Training process
- Visualization and analysis

After training process is done and checkpoints weighting is saved, you can use the same analysis API with initial CAMPA. For getting more tutorial, please check [the tutorial of initial CAMPA](https://campa.readthedocs.io/en/latest/index.html).