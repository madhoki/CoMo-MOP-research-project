Computational chemistry project looking at metal-organic polyhedra.
All data should be imported via the provided FinalData folder.

The folders 'Developing Code' and 'random stuff' should be ignored. They just contain unstructured attempts at a variety of different problems.

The folder 'FinalisedCode' contains everything relating to processing xyz datasets:
- The file 'MOPgeoDataGeneration_final.py' should be run to generate such a dataset, when provided with a folder of xyz files.
- The file 'functions.py' contains the logic behind how all calculations are performed
- The file 'centroidoptimisation.py' demonstrates which method of calculating centroids is preferred

The folder 'TestingCGbind' contains attempts at automated MOP assembly since all attempts failed due to the limited capabilities of cgbind, the code is redundant.

The folder 'ML_model_larger_datasets' contains machine learning attempts on the database of tetrahedral MOPs:
- All files are various different attempts, including ngboost, Gaussian process regression, and XGBoost. DeepLearning with Pytorch is also attempted, although the dataset is not large enough for this to be useful.
- The files 'Parsingstructures.py' and 'EnrichingDataset.py' should be run in that order to generate data for the models. They must be run after the dataset of geometric features is generated from 'MOPgeoDataGeneration_final.py'

The folder 'MLmodel_smaller_datasets' contains machine learning attempts on the database of MOPs named twa_mop_cavity_data.





