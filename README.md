# LUNA16 CNN Training Pipeline

This repository contains the code and notebooks for preprocessing, training, and evaluating six Convolutional Neural Network models. Three 2D models and three 3D models designed for the LUNA16 dataset. 

## Project Structure

Below is an overview of the main files and their purposes in this project:

### Models & Evaluation
* **`models_2D.py`**: Contains the architectures for three distinct 2D CNN models.
* **`models_3D.py`**: Contains the architectures for three distinct 3D CNN models.
* **`froc_eval.py`**: Contains the evaluation metrics, focused on calculating the FROC score and plotting it.

### Notebooks & Workflow
* **`preprocessing_compression.ipynb`**: Handles the data preprocessing.
* ** `cube_cutter_c`**: Cutting out patches/cubes from the compressed data. Normalizes and splits into training and validation sets.
* **`train_c.ipynb`**: The main training pipeline. Imports the models, executes the training loop, and runs the evaluation. 
* **`visualizing_masked_scans.ipynb`**: Used for exploring and visualizing the masked CT scan data examples.


## Data and Weights
To keep the repository lightweight, the `.gitignore` is configured to exclude:
* Raw and processed data (`data/`, `*.npz`)
* Saved model weights (`*.keras`, `*.h5`)
* Training outputs and generated plots (`training_history/`, `plots/`)

