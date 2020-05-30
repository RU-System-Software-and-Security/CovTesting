import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_excel(io='correlation.xlsx', sheet_name='ALL_ALL')
    data = data.set_index(' ')

    # create hot picture in Seaborn
    f, ax = plt.subplots(figsize=(10, 11))
    ax.set_xticklabels(data, rotation='horizontal')

    heatmap = sns.heatmap(data,
                          square=True,
                          cmap='coolwarm',
                          cbar_kws={'shrink': 0.7, 'ticks': [-1, -.5, 0, 0.5, 1]},
                          vmin=-1,
                          vmax=1,
                          annot=True,
                          annot_kws={'size': 16})

    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation = 45)

    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation = 45)


    # cb = heatmap.figure.colorbar(heatmap.collections[0])
    # cb.ax.tick_params(length = 0.001, width=2,  labelsize=10)

    # sns.set_style({'yticklabels.top': True})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('ALL_ALL.eps', format='eps', bbox_inches='tight')

    plt.show()


