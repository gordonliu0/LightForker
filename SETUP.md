# Install Requirements

1. Clone this repository.
2. Make sure Python is updated to version 3.12.6, checking using `python3 --version`
3. Set up a virtual environment using `python3 -m venv .venv` and `source .venv/bin/activate`
4. Update pip using `pip install --upgrade pip`
5. Install dependencies: `pip install -r requirements.txt`

# Download Datasets

LightFormer uses two datasets:

- Bosch Small Traffic Lights Dataset: https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset
- LISA Traffic Light Dataset: https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset

# Setup Datasets

1. Install datasets from links above.
2. Unzip and place in desired directory. I like to put it in the dataset folder, as indicated by the current generate_config.py settings.
3. For LISA (called Kaggle here), you can remove the `/Annotations` folder as we have custom LightFormer annotations in `/data_label/data/Kaggle_Dataset`. For Bosch,
4. In `configs/generate_config.py` update sample_database_folder paths for training, testing, and validation.
5. Run `python3 configs/generate_config.py` to generate `configs/Light_Former_config.json`.
6. Note that if you are keeping this on github make sure to .gitignore so you don't upload it!
