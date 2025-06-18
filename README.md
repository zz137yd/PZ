# PZ
This is a recorded Korean voice that can convert speech to text and identify the speaker's age, gender, and accent.

However, due to the poor quality of the dataset, the current model is not very accurate.

Next, the training process and use of the user interface are explained.

---

## 1. Model

You can run xlsr.ipynb or xlsr.py.

If use xlsr.py, you can modify the epoch number manually.

```python
python3 [-u] xlsr.py [> test.log]
```

The best model file and parameter record document will be placed in the generated saved_model directory.

---

## 2. User interface
