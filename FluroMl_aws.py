# Import libraries
from flask import Flask, request, jsonify
try:
    from rdkit import Chem
    from rdkit.Chem.Draw import MolToImage
    import deepchem as dc
except ImportError as e:
    raise ImportError("Ensure RDKit and DeepChem are installed and accessible in your environment.") from e
import joblib
import base64
import pandas as pd

# Load models
model_path_fluorescence = "./models/best_classifier.joblib"
model_fluorescence = joblib.load(model_path_fluorescence)

model_path_regression = "./models/new_best_regressor.joblib"
model_regression = joblib.load(model_path_regression)

model_path_emission = "./models/best_regressor.joblib"
model_emission = joblib.load(model_path_emission)

# Initialize Flask app
app = Flask(__name__)

# Helper functions

def smiles_to_morgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
    fp = featurizer.featurize([mol])
    return fp[0]  # Take the features for the first (and only) sample

def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    featurizer = dc.feat.MACCSKeysFingerprint()
    descriptors = featurizer.featurize([mol])
    return pd.DataFrame(data=descriptors)

def predict_absorption_max(model, smiles, solvent):
    smiles_desc = smiles_to_descriptors(smiles)
    solvent_desc = smiles_to_descriptors(solvent)
    X = pd.concat([smiles_desc, solvent_desc], axis=1)
    y_pred = model.predict(X)
    return y_pred[0]

def predict_emission_max(model, smiles, solvent):
    smiles_desc = smiles_to_descriptors(smiles)
    solvent_desc = smiles_to_descriptors(solvent)
    X = pd.concat([smiles_desc, solvent_desc], axis=1)
    emission_max_pred = model.predict(X)
    return emission_max_pred[0]

def predict(model, fp):
    return model.predict([fp])[0]

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    img = MolToImage(mol)
    img_bytes = img.tobytes()
    encoded_img = base64.b64encode(img_bytes).decode('utf-8')
    return encoded_img

# Flask routes
@app.route('/predict_emission', methods=['POST'])
def emission_prediction():
    data = request.json
    smiles = data.get('smiles')
    solvent = data.get('solvent')
    try:
        result = predict_emission_max(model_emission, smiles, solvent)
        image = draw_molecule(smiles)
        return jsonify({"emission_max": result, "molecule_image": image})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict_fluorescence', methods=['POST'])
def fluorescence_prediction():
    data = request.json
    smiles = data.get('smiles')
    try:
        fp = smiles_to_morgan(smiles)
        result = predict(model_fluorescence, fp)
        image = draw_molecule(smiles)
        prediction = "Fluorescent Molecule" if result == 1 else "Non-fluorescent Molecule"
        return jsonify({"prediction": prediction, "molecule_image": image})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict_absorption', methods=['POST'])
def absorption_prediction():
    data = request.json
    smiles = data.get('smiles')
    solvent = data.get('solvent')
    try:
        result = predict_absorption_max(model_regression, smiles, solvent)
        image = draw_molecule(smiles)
        return jsonify({"absorption_max": result, "molecule_image": image})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
