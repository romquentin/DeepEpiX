# Tuto 6: Add your own model

This tutorial explains how to integrate a **custom model** into the Dash app.  
Models are executed in **separate environments** to avoid dependency conflicts and to keep the app lightweight.


You need to install the app locally because it requires code adjustments to make your custom model pipeline work. 
[↪️ Go to Developer Guide > Setup & Run](../../dev/setup.md)

---

## Folder Structure

- **Dash app environment**:  
  `.dashenv` → Runs the Dash UI and callbacks.  

- **Model environments**:  
    `.tfenv` → TensorFlow models (`.keras`, `.h5`)  
    `.torchenv` → PyTorch models (`.pth`)  

- **Model folder**:  
  Place your model inside `models/` with the correct extension.

---

## How It Works

When you run a model:

1. The app detects the model type based on the file extension.
2. It selects the corresponding Python binary (`.tfenv` or `.torchenv`).
3. A subprocess is started to execute the model script (`model_pipeline/run_model.py`).
4. Results are saved to the cache directory for later retrieval.

---

## 🚀 Adding Your Model

**1. Place Your Model**

Add your trained model file to the `models/` directory:

- TensorFlow → `.keras` or `.h5`  
- PyTorch → `.pth`

**2. If needed, update `config.py`**

Define the paths to your virtual environments:
```python
TENSORFLOW_ENV = Path("/path/to/.tfenv")
TORCH_ENV = Path("/path/to/.torchenv")
```
If you created your environments following the setup tutorial, you don’t need to modify anything here.
You may, however, need to install additional packages inside the TensorFlow or Pytorch environment depending on your model requirements.

**3. Inference Pipeline**

The app launches inference depending on whether the model is PyTorch or TensorFlow.
Originally, it uses our preprocessing functions (`save_data_matrices`, `create_windows`, `generate_database`) before running inference.
By default, inference is delegated to `test_model_dash` from the respective backend.

If your model requires a custom pipeline, you can extend `run_model_pipeline` by adding conditions such as:

```python
if model_name == "path/to/your/custom/model":
    return test_model_custom(...)
```

argument it can takes and format final output

---

## ✅ Checklist Before Running

* Model file is in `models/`.
* Virtual environments are correctly defined in `config.py`.
* `run_model_pipeline` can handle your model type, or has been extended with your custom logic.
* Script writes results (CSV, logs, etc.) to the cache directory.


---


