import pandas as pd

df = pd.read_csv('l3_results.csv')

tabel = df.pivot_table(
    index=['D', 'func'], 
    columns='alg', 
    values='error', 
    aggfunc='median'
)

tabel = tabel[['PRS', 'LRS', 'SW']]

print(tabel.to_string(float_format=lambda x: f"{x:.3g}"))
