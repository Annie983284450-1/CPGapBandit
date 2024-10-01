import pandas as pd
import numpy as np

sepsis_full_48 = pd.read_csv('fully_imputed_8windowed_max48_updated.csv')
sepsis_full_48.loc[sepsis_full_48['hours2sepsis'] == 49, 'hours2sepsis'] = 480
# Optional: Reset the index if needed
sepsis_full_48.reset_index(drop=True, inplace=True)
# Save the updated DataFrame to a new CSV file
sepsis_full_48.to_csv('fully_imputed_8windowed_max480_updated.csv', index=False)

