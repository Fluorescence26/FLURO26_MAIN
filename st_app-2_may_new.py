import streamlit as st
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage
import joblib
import pandas as pd
import deepchem as dc

# Load models
try:
    model_fluorescence = joblib.load("best_classifier.joblib")
    model_regression = joblib.load("new_best_regressor.joblib")
    model_emission = joblib.load("best_regressor_emission.joblib")
except Exception as e:
    st.error(f"Error loading models: {e}")

# Helper function to calculate Morgan fingerprints
def smiles_to_morgan(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
        fp = featurizer.featurize([mol])
        return fp[0]
    except Exception as e:
        st.error(f"Error in fingerprint calculation: {e}")
        return None

# Helper function to calculate descriptors
def smiles_to_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        featurizer = dc.feat.MACCSKeysFingerprint()
        descriptors = featurizer.featurize([mol])
        return pd.DataFrame(data=descriptors)
    except Exception as e:
        st.error(f"Error generating descriptors: {e}")
        return None

# Prediction functions
def predict_fluorescence(model, fp):
    try:
        pred = model.predict([fp])[0]
        return pred
    except Exception as e:
        st.error(f"Error in fluorescence prediction: {e}")
        return None

def predict_absorption_max(model, smiles, solvent):
    try:
        smiles_desc = smiles_to_descriptors(smiles)
        solvent_desc = smiles_to_descriptors(solvent)
        X = pd.concat([smiles_desc, solvent_desc], axis=1)
        y_pred = model.predict(X)
        return y_pred[0]
    except Exception as e:
        st.error(f"Error in absorption max prediction: {e}")
        return None

def predict_emission_max(model, smiles, solvent):
    try:
        smiles_desc = smiles_to_descriptors(smiles)
        solvent_desc = smiles_to_descriptors(solvent)
        X = pd.concat([smiles_desc, solvent_desc], axis=1)
        y_pred = model.predict(X)
        return y_pred[0]
    except Exception as e:
        st.error(f"Error in emission max prediction: {e}")
        return None

# Draw molecule from SMILES
def draw_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return MolToImage(mol)
    except Exception as e:
        st.error(f"Error drawing molecule: {e}")
        return None

# Streamlit UI
st.title("Chemical Properties Prediction")

# Sidebar for model selection
model_selector = st.sidebar.selectbox(
    "Select Model",
    ["Fluorescence Classification", "Absorption Max Prediction", "Emission Max Prediction"]
)

if model_selector == "Fluorescence Classification":
    st.header("Fluorescence Prediction")
    smiles_input = st.text_input("Enter a SMILES string:")
    if smiles_input:
        fp = smiles_to_morgan(smiles_input)
        if fp is not None:
            prediction = predict_fluorescence(model_fluorescence, fp)
            if prediction is not None:
                st.image(draw_molecule(smiles_input), caption="Molecule Structure")
                st.write(f"Prediction: {'Fluorescent' if prediction == 1 else 'Non-Fluorescent'}")

elif model_selector == "Absorption Max Prediction":
    st.header("Absorption Max Prediction")
    smiles_input = st.text_input("Enter a SMILES string for the molecule:")
    solvent_input = st.text_input("Enter a SMILES string for the solvent:")
    if smiles_input and solvent_input:
        prediction = predict_absorption_max(model_regression, smiles_input, solvent_input)
        if prediction is not None:
            st.image(draw_molecule(smiles_input), caption="Molecule Structure")
            st.write(f"Predicted Absorption Max: {round(prediction, 2)} nm")

elif model_selector == "Emission Max Prediction":
    st.header("Emission Max Prediction")
    smiles_input = st.text_input("Enter a SMILES string for the molecule:")
    solvent_input = st.text_input("Enter a SMILES string for the solvent:")
    if smiles_input and solvent_input:
        prediction = predict_emission_max(model_emission, smiles_input, solvent_input)
        if prediction is not None:
            st.image(draw_molecule(smiles_input), caption="Molecule Structure")
            st.write(f"Predicted Emission Max: {round(prediction, 2)} nm")
