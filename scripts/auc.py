import numpy as np
import pandas as pd
import os
from scipy.integrate import trapz

directory = '/workspace/eval_results_final'

results = []

# Iterate through all .dat files
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".dat"):
        data = pd.read_csv(os.path.join(directory, filename), sep='\s+', names=['perc', 'nds'], skiprows=1)

        data = data[pd.to_numeric(data['perc'], errors='coerce').notnull()]
        data['perc'] = data['perc'].astype(int)
        data['nds'] = data['nds'].apply(pd.to_numeric, errors='coerce')

        # Extract values and calculate AUC
        x_values = data['perc'].values
        y_values = data['nds'].values
        auc = trapz(y_values, x_values)

        results.append([filename, auc])

# Write results to file
with open('results.txt', 'w') as f:
    for item in results:
        f.write("%s %f\n" % (item[0], item[1]))