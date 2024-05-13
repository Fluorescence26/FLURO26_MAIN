
import pandas as pd
from category_encoders import TargetEncoder
from imblearn.under_sampling import RandomUnderSampler
from utils import encode_targets

def preprocess_dataframe(df):
    # Select only SMILES and Target columns
    df = df[['Smiles', 'Fluorescent labeling']]

    # Drop duplicates
    df = df.drop_duplicates()

    # Target encoding
    df = encode_targets(df)
    

    # Undersampling
    
    # df = undersampler(df)

    return df

def preprocess_dataframe_for_regression(df):
    # Select only SMILES and Target columns
    df = df[['Smiles', 'Fluorescent labeling']]

    # Drop duplicates
    df = df.drop_duplicates()

    # Target encoding
    df = encode_targets(df)
    

    # Undersampling
    
    # df = undersampler(df)

    return df