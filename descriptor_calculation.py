
import deepchem as dc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



def descriptor_cal(df,desc,TARGET):
  """
  Input: cleaned dataframe and featurization method name
  Output: - dataframe x which contains FP/Descriptors
          - datafeame y which contains target values
  """
  try:
      dataset_original=df
      
      # morgan
      if desc == "Morgan fingerprints": 
          smiles = dataset_original['Smiles'].tolist()
          featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
          discriptors = featurizer.featurize(smiles)
          x = pd.DataFrame(data=discriptors)

          y = dataset_original[TARGET]

          return x, y
      # rdkit

      elif desc == 'Mordred descriptors':
          smiles = dataset_original['Smiles'].tolist()
          featurizer = dc.feat.MordredDescriptors(ignore_3D = True)
          discriptors = featurizer.featurize(smiles)
          x = pd.DataFrame(data=discriptors)
          
          y = dataset_original[TARGET]

          return x, y

      elif desc == 'MACCSKeysFingerprint':
          smiles = dataset_original['Smiles'].tolist()
          featurizer = dc.feat.MACCSKeysFingerprint()
          discriptors = featurizer.featurize(smiles)
          x = pd.DataFrame(data=discriptors)
          y = dataset_original[TARGET]

          return x, y

      elif desc == 'RDKitDescriptors':
          smiles = dataset_original['Smiles'].tolist()
          featurizer = dc.feat.RDKitDescriptors(use_fragment=True, ipc_avg=True)
          discriptors = featurizer.featurize(smiles)
          x = pd.DataFrame(data=discriptors)
          x.replace([np.inf, -np.inf], np.nan, inplace=True)
          x.fillna(0, inplace=True)
          scaler = StandardScaler()
          x = pd.DataFrame(data=scaler.fit_transform(x))
          y = dataset_original[TARGET]

          return x, y

      elif desc == 'PubChemFingerprint':
          smiles = dataset_original['Smiles'].tolist()
          featurizer = dc.feat.PubChemFingerprint()
          discriptors = featurizer.featurize(smiles)
          x = pd.DataFrame(data=discriptors)
          y = dataset_original[TARGET]

          return x, y

      else:
          pass

  except Exception as e:
      raise e

def smiles_solvent_descriptors(df,desc,TARGET):
  try:
      dataset_original=df
      
      # morgan
      if desc == "Morgan fingerprints": 
          smiles = dataset_original['Smiles'].tolist()
          featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
          discriptors = featurizer.featurize(smiles)
          smiles_desc = pd.DataFrame(data=discriptors)

          solvent = dataset_original['Solvent'].tolist()
          discriptors_solvent = featurizer.featurize(solvent)
          solvent_desc = pd.DataFrame(data=discriptors_solvent)

          x = pd.concat([smiles_desc, solvent_desc], axis=1)

          y = dataset_original[TARGET]

          return x, y
      # rdkit

      elif desc == 'Mordred descriptors':
          smiles = dataset_original['Smiles'].tolist()
          featurizer = dc.feat.MordredDescriptors(ignore_3D = True)
          discriptors = featurizer.featurize(smiles)
          smiles_desc = pd.DataFrame(data=discriptors)

          solvent = dataset_original['Solvent'].tolist()
          discriptors_solvent = featurizer.featurize(solvent)
          solvent_desc = pd.DataFrame(data=discriptors_solvent)

          x = pd.concat([smiles_desc, solvent_desc], axis=1)
          
          y = dataset_original[TARGET]

          return x, y

      elif desc == 'MACCSKeysFingerprint':
          smiles = dataset_original['Smiles'].tolist()
          featurizer = dc.feat.MACCSKeysFingerprint()
          discriptors = featurizer.featurize(smiles)
          smiles_desc = pd.DataFrame(data=discriptors)

          solvent = dataset_original['Solvent'].tolist()
          discriptors_solvent = featurizer.featurize(solvent)
          solvent_desc = pd.DataFrame(data=discriptors_solvent)

          x = pd.concat([smiles_desc, solvent_desc], axis=1)
          y = dataset_original[TARGET]

          return x, y

      elif desc == 'RDKitDescriptors':
          smiles = dataset_original['Smiles'].tolist()
          featurizer = dc.feat.RDKitDescriptors(use_fragment=True, ipc_avg=True)
          discriptors = featurizer.featurize(smiles)
          smiles_desc = pd.DataFrame(data=discriptors)

          solvent = dataset_original['Solvent'].tolist()
          discriptors_solvent = featurizer.featurize(solvent)
          solvent_desc = pd.DataFrame(data=discriptors_solvent)

          x = pd.concat([smiles_desc, solvent_desc], axis=1)
          x.replace([np.inf, -np.inf], np.nan, inplace=True)
          x.fillna(0, inplace=True)
          scaler = StandardScaler()
          x = pd.DataFrame(data=scaler.fit_transform(x))
          y = dataset_original[TARGET]

          return x, y

      elif desc == 'PubChemFingerprint':
          smiles = dataset_original['Smiles'].tolist()
          featurizer = dc.feat.PubChemFingerprint()
          discriptors = featurizer.featurize(smiles)
          smiles_desc = pd.DataFrame(data=discriptors)

          solvent = dataset_original['Solvent'].tolist()
          discriptors_solvent = featurizer.featurize(solvent)
          solvent_desc = pd.DataFrame(data=discriptors_solvent)

          x = pd.concat([smiles_desc, solvent_desc], axis=1)
          y = dataset_original[TARGET]

          return x, y

      else:
          pass

  except Exception as e:
      raise e


