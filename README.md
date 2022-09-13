# Vessel Collision Risk Assessment (VCRA)
### Official Python implementation of the MLP-VCRA model from the paper "Vessel Collision Risk Assessment using AIS Data: A Machine Learning Approach"


# Installation 
In order to use MLP-VCRA in your project, download all necessary modules in your directory of choice via pip or conda, and install their corresponding dependencies, as the following commands suggest:

```Python
# Using pip/virtualenv
pip install −r requirements.txt

# Using conda
conda install --file requirements.txt
```

To reproduce the experimental study of the paper (MLP-VCRA vs. related work), please install the ```scikit-rvm``` (https://github.com/JamesRitchie/scikit-rvm) library


# Usage

In the ```cri_calc.py``` and ```cri_helper.py``` scripts you will find all necessary mathematical formulae regarding CRI calculation, as well as a usage example. 

The jupyter notebook ```4-vcra-equations-v2-model-trimming.ipynb``` contains all necessary code in order to create the training dataset (sample included at ```./data/```), and train the MLP-VCRA model.

Finally, the jupyter notebooks ```5-comparing-with-related-work.ipynb```, ```6-vcra-mlp-variants.ipynb``` present the results of our experimental study, and in ```7-exploitation-and-discussion``` we illustrate some visualizations regarding potential use-cases of MLP-VCRA.


# Contributors
Andreas Tritsarolis, Eva Chondrodima, and Yannis Theodoridis; Department of Informatics, University of Piraeus

Nikos Pelekis; Department of Statistics & Insurance Science, University of Piraeus


# Acknowledgement
This work was supported in part by EU Horizon 2020 R\&I Programme under Grant Agreement No 957237 (project VesselAI, https://vessel-ai.eu), and EU Horizon 2020 R\&I Programme under Marie Sklodowska-Curie Grant Agreement No 777695 (project MASTER, http://www.master-project-h2020.eu).
