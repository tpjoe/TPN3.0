import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_csv('data/processed/autoregressive.csv', low_memory=False)

df = df.drop_duplicates('person_id', keep='first')
df.loc[df.dated_max_chole_TPNEHR==1, :].drop_duplicates('person_id').day_since_birth.describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
df.drop_duplicates('person_id').max_chole_TPNEHR.value_counts()