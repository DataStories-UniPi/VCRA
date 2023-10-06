# Vessel Collision Risk Assessment and Forecasting (VCRA/F)
### Official Python implementation of the VCRA/F model from the paper "Collision Risk Assessment and Forecasting on Maritime Data"


# Installation 
In order to use VCRA/F in your project, download all necessary modules in your directory of choice via pip or conda, and install their corresponding dependencies, as the following commands suggest:

```Python
# Using pip/virtualenv
pip install −r requirements.txt

# Using conda
conda install --file requirements.txt
```

To reproduce the experimental study of the paper (VCRA/F vs. related work), please install the ```sklearn-rvm``` library (sklearn-rvm.readthedocs.io).


# Usage


In order to discover encountering vessels and create the training dataset for VCRA, please consult the code in the notebook ```4-encountering-vessels.ipynb```

In order to train the VCRA model, please run the script ```5-comparing-with-related-work.py``` using the following arguments:

```
python 5-comparing-with-related-work.py --model mlp
```

To reproduce the experimental study of the paper, please execute the code in the notebooks ```5-comparing-with-related-work``` (quality of VCRA vs. related work), ```6-comparing-models-latency``` (response time of VCRA vs. related work), and ```7-model-transparency``` (VCRA model explainability).

In order to assess the collision risk of future encountering processes, one must first train a VRF model for predicting vessels' locations. Since VRF models can be deployed in a plug and play mode, one can use whatever VRF model he/she finds fit (e.g., [https://github.com/DataStories-UniPi/VLF_VRF](https://github.com/DataStories-UniPi/VLF_VRF)). Afterwards, for the predicted locations, run the code in the notebook ```8-vcrf-encountering-vessels-arrnn.ipynb``` in order to discover future encountering processes (along with their corresponding CRI) and reproduce the experimental study of VCRF as well.


# Contributors
Andreas Tritsarolis; Department of Informatics, University of Piraeus

Brian Murray; Department of Energy & Transport, SINTEF Ocean

Nikos Pelekis; Department of Statistics & Insurance Science, University of Piraeus

Yannis Theodoridis; Department of Informatics, University of Piraeus


# Citation
If you use VCRA/F in your project, we would appreciate citations to the following paper:

> Andreas Tritsarolis, Brian Murray, Nikos Pelekis, and Yannis Theodoridis. 2023. Collision Risk Assessment and Forecasting on Maritime Data (Industrial Paper). In proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL). https://doi.org/10.1145/3589132.3625573


# Acknowledgement
This work was supported in part by EU Horizon 2020 R\&I Programme under Grant Agreement No 957237 (project VesselAI, https://vessel-ai.eu).
