# Neural-architecture-search-challenge
Neural architecture search challenge on unseen data 2024
# Contents
The starting kit contains the following:
* `main.py`: The main pipeline. This will load each dataset, pass it through pipeline, and then produce test predictions
* `nas.py` : The file which build the pytorch model
* `trainer.py` : This file trains the model
* `data_processor.py` : This filee will process the dataset
* `helper.py` : This file will containes the extra function to keep the code clean in mian files


# Datasets
The pipeline and DataLoaders are expecting each dataset to be contained in its own folder with six NumPy files for the training, validation, and testing data, split between images and labels. Furthermore, a `metadata` file is expected containing the input shape, codename, benchmark, and number of classes. See the datasets are created (linked below), for the appropriate structure.
- AddNIST: [https://doi.org/10.25405/data.ncl.24574354.v1](https://doi.org/10.25405/data.ncl.24574354.v1)
- Language: [https://doi.org/10.25405/data.ncl.24574729.v1](https://doi.org/10.25405/data.ncl.24574729.v1)
- MultNIST: [https://doi.org/10.25405/data.ncl.24574678.v1](https://doi.org/10.25405/data.ncl.24574678.v1)
- CIFARTile: [https://doi.org/10.25405/data.ncl.24551539.v1](https://doi.org/10.25405/data.ncl.24551539.v1)
- Gutenberg: [https://doi.org/10.25405/data.ncl.24574753.v1](https://doi.org/10.25405/data.ncl.24574753.v1)
- GeoClassing: [https://doi.org/10.25405/data.ncl.24050256.v3](https://doi.org/10.25405/data.ncl.24050256.v3)
- Chesseract: [https://doi.org/10.25405/data.ncl.24118743.v2](https://doi.org/10.25405/data.ncl.24118743.v2)

