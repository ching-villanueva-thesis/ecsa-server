import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fmin_values = pd.read_csv('./src/simulator/results/F10_fmin_values_20241110_181712.csv')

csa_fmin = fmin_values['CSA']
ecsa_fmin = fmin_values['ECSA']

labels = ['Cuckoo Search Algorithm', 'Enhanced Cuckoo Search Algorithm']
colors = ['lightblue', 'orange']

fig, ax = plt.subplots()
ax.set_ylabel('Values of best fmin')
ax.set_yscale('log')

bplot = ax.boxplot([csa_fmin,ecsa_fmin],
                #    patch_artist=True,  # fill with color
                   tick_labels=labels)  # will be used to label x-ticks

# fill with colors
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

plt.show()