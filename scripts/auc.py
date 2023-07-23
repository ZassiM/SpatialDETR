import numpy as np
import numpy as np
import pandas as pd
from scipy.integrate import trapz

# Load data from text file
data = pd.read_csv('/workspace/neg_pert/gradroll.txt', sep='\s+', names=['Step', 'NDS'], skiprows=1)

# Convert Step values to integer and extract y values
x_values = data['Step'].astype(int).values
y_values = data['NDS'].values

# Calculate AUC using trapezoidal rule
auc = trapz(y_values, x_values)

print("The area under the curve is:", auc)
