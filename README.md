# ACLNK_main
Adversarial contrastive with leveraging negative knowledge for point of interest sequence learning (ACLNK)

# Requirements
* python >= 3.8
* torch == 2.1
* numpy == 1.24.4
* pandas == 2.0.3

# Datasets
1. Download raw data from following sources:
   (https://drive.google.com/drive/folders/1hSfZKnDMDXkwzeRoort4GCjJq7wx4IbX?usp=drive_link)
   Please note! When downloading the data, please use a VPN or proxy; otherwise, the link may appear to be invalid.
2. Extract the `new_datasets.zip` file. Create a new folder named `data/new_datasets`, then copy all files and directories to `data/new_datasets`
3. Copy `glove.twitter.27B.50d.pkl` to `data/`

# Configuration
1. Extract the `config.zip` file. Create a new folder named `config`, then copy all files and directories to `config/`
2. For example:
   TUL task for the NYC dataset, see `config/ACLNK_nyc_TUL.conf`

# Run code
   1. Run `train_ACLNK.py` directly.  
   **Note:** You can directly modify the config file path in line 24 of `train_ACLNK.py`; the file corresponds to those in `./config/`.

   2. Via the command line:
   For example, Train model on NYC of TUL task: `python train_ACLNK.py --config config/CACSR_gow_TUL.conf --dataroot data/`

## Acknowledgements
We thank everyone who has helped our work.
Shanghai, China. 


