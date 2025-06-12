# FC4NO

This repository provides a **fair comparison of neural operator architectures** for 3D real-world engineering problems.

---

## Project Structure

- **datascripts/**: Data processing and loading scripts for different datasets (e.g., `drivAer.py`, `jeb.py`, `bracket_time.py`, etc.).
- **models/**
  - **Point_models/**: Point-based neural operator models (e.g., `gnot.py`, `transolver.py`, `pointnet.py`).
  - **Grid_models/**: Grid-based models and supporting modules.
  - **Graph_models/**: Graph-based neural operator models (e.g., `gno.py`, `eagno.py`, `GraphUNet.py`).
  - **Branch_trunk_models/**: Branch-trunk neural operator models (e.g., `deepOnet.py`, `SDON.py`, `gano.py`, etc.).
- **utils/**: Utility scripts for training, data handling, and plotting.
- **configs/**
  - **data/**: YAML files specifying dataset configurations.
  - **model/**: YAML files specifying model configurations.
- **res/**
  - **trained_model/**: Pretrained model checkpoints.
  - **visualizations/**: Output visualizations.
- **Jobscripts/**: Shell scripts to run experiments for different datasets/models.
- **runs/**: Main experiment runner (`main.py`) and setup utilities.

---

## Getting Started

### 1. Data

- Download datasets from: https://dataverse.harvard.edu/dataverse/FC4NO
- Update the `data_loc_dict` in `runs/setup.py` if your data paths differ.

### 2. Environment Setup

- **Recommended:** Run experiments on a cluster due to dataset size and computational requirements.
- Build the container in the cluster:
  ```bash
  apptainer pull modulus.sif docker://weihengz/nvidia:latest
  ```
- Start the container:
  ```bash
  apptainer run --nv --writable-tmpfs --no-home --bind $PWD$:/FC modulus.sif
  ```
- For graph models, install additional dependencies:
  ```bash
  pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
  ```

### 3. Running Experiments

- Use the provided shell scripts in `Jobscripts/` to launch experiments for different datasets/models:
  ```bash
  bash Jobscripts/driver.sh
  bash Jobscripts/jet.sh
  # etc.
  ```
- Or run experiments directly with:
  ```bash
  cd runs
  python main.py --model <MODEL_NAME> --data <DATASET_NAME> --phase <train|test|plot>
  ```
  - Example:
    ```bash
    python main.py --model GNO --data bracket --phase train
    ```

### 4. Configuration

- **Model and data configurations** are in `configs/model/` and `configs/data/` as YAML files.
- You can customize model architectures and dataset parameters by editing these files.

### 5. Results

- Trained models are saved in `res/trained_model/`.
- Visualizations and outputs are in `res/visualizations/`.

---

## Available Models

- **Point-based:** GNOT, Transolver, PointNet
- **Grid-based:** figconv, gifno, vt, etc.
- **Graph-based:** GNO, EAGNO, GUNet
- **Branch-trunk:** DeepONet, SDON, DCON, GANO, geomDON

---

## Utilities

- **utils/utils_data.py**: Data loading and preprocessing
- **utils/utils_train.py**: Training and evaluation routines
- **utils/utils_plot.py**: Plotting and visualization

---

## Notes

- Make sure to adjust paths and dependencies as needed for your environment.
- For more details on each script or model, refer to the respective Python files.




