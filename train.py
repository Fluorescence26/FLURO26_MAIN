from train_func import train_models
import shutil
import os
import pandas as pd

# Define the path to the folder containing subfolders
path_to_descriptors = "/config/workspace/Descriptors"
results_folder = "results"
if os.path.exists(results_folder):
    shutil.rmtree(results_folder)



# Loop over each subfolder
for subfolder_name in os.listdir(path_to_descriptors):
    
    # Define the path to the X.csv and Y.csv files for the current subfolder
    path_to_X = os.path.join(path_to_descriptors, subfolder_name, "X.csv")
    path_to_Y = os.path.join(path_to_descriptors, subfolder_name, "Y.csv")
    
    # Load the X.csv and Y.csv files as dataframes
    X = pd.read_csv(path_to_X)
    Y = pd.read_csv(path_to_Y)
    
    # Call the train_models function with X, Y, and the current subfolder name as the PATH argument
    train_models(X, Y, f"{results_folder}/{subfolder_name}")
