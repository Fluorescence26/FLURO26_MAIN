# Import libraries
import streamlit as st
import deepchem as dc
# import numpy as np
# import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
import joblib

# Load model 
model_path = "best_models/best_classifier.joblib"
model = joblib.load(model_path)

#  calculate Morgan fingerprints from SMILES string
def smiles_to_morgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp  = dc.feat.CircularFingerprint(radius=3, size=1024)
    return fp.featurize([mol])

# predict fluorescence from Morgan fingerprints
def predict_fluorescence(fp):
    pred = model.predict(fp)
    return pred

#  draw structure
def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return MolToImage(mol)

# title
st.title("Fluorescence Predictor")

# input 
smiles = st.text_input("Enter a SMILES string:")

# Check if the input is valid and not empty
if smiles:
    try:
        # Calculate Morgan fingerprints from SMILES string
        fp = smiles_to_morgan(smiles)

        # Predict fluorescence from Morgan fingerprints
        fluorescence = predict_fluorescence(fp)

        # Draw molecule structure from SMILES string
        image = draw_molecule(smiles)

        # Display the molecule structure and the prediction result on the app
        st.image(image, caption="Molecule Structure:") 
        st.write(f"{'Fluorescent' if fluorescence== 1 else 'Non-fluorescence'}")
        st.warning("Applicability Domain module is yet to be added, Predictions are less reliable if input molecule is not simmilar to Model Training molecules")

    except:
        # Display an error message if the input is invalid or cannot be processed 
        st.error("Invalid input or unable to process. Please enter a valid SMILES string.")
