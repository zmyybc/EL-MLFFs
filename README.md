# EL-MLFFs: Ensemble Learning Force Fields Repository

Welcome to the EL-MLFFs repository, your comprehensive source for ensemble learning force fields. This repository is dedicated to the development and sharing of advanced models combining multiple machine learning force fields (MLFFs) to enhance prediction accuracy and reliability in molecular simulations.

## Dependencies

To set up your environment for using this repository, please create a conda environment with the provided `environment.yml` file:

```
conda env create -f environment.yml
```

## File Structure

- `el-mlffs`: Contains code for preparing data and training our ensemble models.
- `mlps`: Training work directory for individual MLFFs.
- `heatmap`: Stores visual data related to heatmaps for analysis.

## Usage

### Training Single Machine Learning Force Fields

Training scripts for single MLFFs are provided in the `mlps` directory. For detailed instructions on training individual force fields, please refer to the GitHub websites of [deepmd](https://github.com/deepmodeling/deepmd-kit.git), [painn](https://github.com/atomistic-machine-learning/schnetpack.git), [nep](https://github.com/brucefan1983/GPUMD.git), and [schnet](https://github.com/atomistic-machine-learning/schnetpack.git).

### Training Ensemble Force Fields

To train ensemble force fields:

1. Navigate to the `el-mlffs` directory.
2. For the deepmd model, place the model checkpoint in the `data` directory. For other models (painn, nep, schnet), generate a CSV file in the `data` directory using `get_csv.py` found in the `mlps` directory.
3. Modify the input section in `prepare.py` as follows:

```python
if __name__ == '__main__':
    preprocess_data(system_dirs = ["./data/train/"], 
                    model_paths = ["./data/1.pb", "./data/2.pb", "./data/3.pb"], 
                    predicted_forces_files = ["./data/predicted_forces_train_nep.csv",
                                              "./data/predicted_forces_train_painn.csv",
                                              "./data/predicted_forces_train_schnet.csv",
                                              "./data/predicted_forces_train_schnet1.csv",
                                              "./data/predicted_forces_train_schnet2.csv"], 
                    output_file = "./raw/dataset_all_train.pckl")
```
Follow the same procedure for test data.

4. Start the training process:
```
python main_stream.py
```
This command will automatically train all possible ensemble models formed by random combinations of the provided MLFFs. For instance, with 8 individual MLFFs, a total of 255 ensemble models will be generated.
5. To evaluate your models, run:
```
python test_all.py
```
## License

This project is licensed under the MIT License.

## Contribution
We welcome any suggestions and exploration! Feel free to expand our MLFFs family and experiment with different combinations to discover interesting results. Your contributions can help advance the field of molecular simulations.
