from tdc.benchmark_group import dti_dg_group
import pandas as pd 
from rdkit import Chem


def is_kekulizable(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        return True
    except:
        return False
        


# --- Load Benchmark ---
group = dti_dg_group(path='./TDC_data/')

benchmark = group.get('BindingDB_Patent')
name = benchmark['name']
train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=0)
test = benchmark['test']

valid_mask = train["smiles"].apply(is_kekulizable)
train = train[valid_mask].reset_index(drop=True)


df = train[["Drug_ID", "Target_ID"]].copy()
# Convert Drug_ID to numeric
df["Drug_ID"], drug_id_unique = pd.factorize(df["Drug_ID"])
# Convert Target_ID to numeric
df["Target_ID"], target_id_unique = pd.factorize(df["Target_ID"])


train.to_csv("train.csv",index=False)
test.to_csv("test.csv",index=False)
valid.to_csv("val.csv",index=False)


df.to_csv("ref.txt", sep="\t", index=True,header=None)

