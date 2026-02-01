#  Multifaceted E(3)-Equivariant Graph Auto-Encoder 

we propose a Multifaceted E(3)-Equivariant Graph Auto-Encoder that effectively learns and generates transmembrane protein binding domain fingerprints by integrating physicochemical and geometric features.

![图片描述](https://github.com/YantingTong/EGNN/blob/main/figure/Figure1.png)
---

## 📂 Document Structure
```text
EGNN/
├─ Figure/               
├─ model/                 # Model Structure Code
├─ model_weight/          # Trained model weights
├─ IEProtLib.zip          # Relevant dependency files
├─ README.md              # Project Description
├─ ae_train_list.txt      # List of training samples for the autoencoder
├─ create_cnndata.py      # Generate fingerprint feature matrix
├─ create_hdf5.py         # Construct a graph by binding domains
├─ test_model.py          # Model testing script
├─ test_resnet.py         # ResNet test script
└─ train_model.py         # Model training script
```

## ⚙️ Install dependencies
We recommend using conda to create a virtual environment:
```bash
conda create -n egnn python=3.10
conda activate egnn
```
Install the other environment dependencies
```bash
pip install -r requirements.txt
```
## 📥 Data Acquisition
The dataset is hosted on Hugging Face:  
[Click here to download the binding-domain dataset](https://huggingface.co/datasets/12Yan/binding-domain)

## 🚀 Quick Start
**1️⃣ Train AE**
```bash
python train_model.py
```
2️⃣ Test AE
```bash
python test_model.py
```
3️⃣ Binding domain recognition
```bash
python test_resnet.py
```



















