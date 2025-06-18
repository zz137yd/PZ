# PZ
This is a recorded Korean voice that can convert speech to text and identify the speaker's age, gender, and accent.

However, due to the poor quality of the dataset, the current model is not very accurate.

Next, the training process and use of the user interface are explained.


## Model

You can run xlsr.ipynb or xlsr.py.

Before that, you need to install the dependent libraries first.

```python
pip install jamo jiwer jamotools
```

```python
pip install --upgrade transformers
```

Then download dataset.

[Dataset](https://drive.google.com/drive/folders/1VdgGLuVcL4A62MgwGV063MR4IbPzR1Pm?usp=sharing)</br>
[The csv file containing audio file information](https://drive.google.com/file/d/1bfFR-8cpNiQmxc1v145nyZiSVUxmgrzX/view?usp=sharing)

If use xlsr.py, you can modify the epoch number manually.</br>
And modify the dataset path as well as csv file path to yours.

```python
python3 [-u] xlsr.py [> test.log]
```

The best model file and parameter record file will be placed in the generated saved_model directory.

## User interface

After training is completed and there is a .pt file
Change the path of .pt file in the streamlit_app.py under the app directory to the path where the model is actually stored.

Requirement : 

```python
pip install jamo jamotools pydub
```

And then run the interface in terminal window

```bash
streamlit run streamlit_app.py
```

or python environment

```python
python -m streamlit run streamlit_app.py
```
