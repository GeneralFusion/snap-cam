import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('pi3b_summary.csv')
df.fillna(0, inplace=True)
snap_cols = [c for c in df.columns if c.startswith('Cap')]
snap_cols.append('RailGap.png')
shots = df['shot'].values

for snap_col in snap_cols:
    total_num_bad_images = np.cumsum(df[snap_col].values)
    label = f'{snap_col} ({int(total_num_bad_images[-1])} / {int(len(shots))} bad)'
    plt.plot(shots, total_num_bad_images, label=label)
plt.legend()
plt.grid()
plt.xlabel('shot number')
plt.ylabel('# of bad images')
plt.title('Cumulative count of corrupted images on M3V3 snap cams')
plt.savefig("test.png")
