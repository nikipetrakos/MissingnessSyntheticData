# MissingnessSyntheticData
This is the code for the simulation studies in *Incorporating Missingness in the Generation of Realistic Synthetic Trial Data*, preprint available [here](https://arxiv.org/abs/2512.00183). Note that this work is an extension of previous work that investigated methods for generating realistic synthetic RCT data, for which the code can be found [here](https://github.com/nikipetrakos/SyntheticDataGeneration/tree/main).

The `ACTGData_RCode_ToShare.R` file includes the code for the simulations using the original, primary data set (ACTG 175); the `optData_RCode_ToShare.R` file includes the code for the simulations using the second, additional data set (opt, from the Obstetrics and Periodontal Study). In both cases, the files are organized as follows: loading necessary libraries, data preparation (data are publicly available), helper functions (both for data generation and metrics), and finally the simulations. 

Any questions can be directed to niki.petrakos@mail.mcgill.ca.
