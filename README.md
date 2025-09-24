# FIRM-DTI: Geometry-Aware Drug–Target Interaction Prediction


**FIRM-DTI** is a lightweight framework for drug–target binding affinity prediction and DTI classification.
Unlike conventional concatenation models, FIRM-DTI conditions molecular embeddings on protein embeddings via a **FiLM layer** and enforces **metric structure** with a **triplet loss**.
An **RBF regression head** maps embedding distances to smooth, interpretable affinity values, achieving strong out-of-domain performance on the Therapeutics Data Commons **DTI-DG** benchmark.

---


## Requirements

* Python ≥ 3.9
* PyTorch ≥ 2.0
* Hugging Face `transformers` for ESM2
* RDKit (for molecule preprocessing)

Pretrained encoders used:

* **MolE** (GuacaMol checkpoint):
  [MolE\_GuacaMol\_27113.ckpt](https://codeocean.com/capsule/2105466/tree/v1)
* **ESM2** (Facebook AI):
  [`esm2_t12_35M_UR50D`](https://huggingface.co/facebook/esm2_t12_35M_UR50D)

---

##  Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/EESI/Firm-DTI.git
cd Firm-DTI

# 2. (Optional) create a virtual environment and install dependencies
pip install -r requirements.txt

# 3. Download MolE GuacaMol checkpoint
https://codeocean.com/capsule/2105466/tree/v1

# 4. Prepare the patent-year split dataset
mkdir data_patent
cd data_patent
python ../prepare_dataset.py
cd ..

# 5. Train FIRM-DTI
python -u trainer.py \
  --input "./data_patent" \
  --output "./output/model_1" \
  --batch_size 16 \
  --batch_hard False
```






