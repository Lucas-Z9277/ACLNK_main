# ACLNK_main
Adversarial contrastive with leveraging negative knowledge for point of interest sequence learning (ACLNK)

# Requirements
* python >= 3.8
* torch == 2.1
* numpy == 1.24.4
* pandas == 2.0.3

# Datasets
1. Download raw data from following sources:
   [http:](https://drive.google.com/)
2. Copy all files and directories to data/new_datasets
3. Copy `glove.twitter.27B.50d.pkl` to data/

# Configuration
1. Download parameter settings from following sources:
   [http:](https://drive.google.com/)

2. Copy all files and directories to ./config/
   
3. For example:
   TUL task for the NYC dataset, see ./config/ACLNK_nyc_TUL.conf

# Run code
   For example, Train model on NYC of TUL task: 
   python train_ACLNK.py --config config/CACSR_nyc_TUL.conf --dataroot data/

## Acknowledgements
We thank everyone who has helped our work.
Shanghai, China. 


