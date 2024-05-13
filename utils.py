import numpy as np
import pandas as pd
def encode_targets(df):
    df['target'] = [ 1 if typ == 'Yes' else 0 for typ in df['Fluorescent labeling']]
    active = len(df[df['Fluorescent labeling'] == "Yes"])
    inactive = df[df['Fluorescent labeling'] == "No "].index
    random_indices = np.random.choice(inactive,active, replace=False)
    active_indices = df[df['Fluorescent labeling'] == "Yes"].index
    under_sample_indices = np.concatenate([active_indices,random_indices])
    df = df.loc[under_sample_indices]
    df= df[['Smiles',"target"]]
    return df