import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.offsetbox import AnchoredText

from src.models.johnson_cook_model import JohnsonCookModel
from src.models.khan_huang_liang_model import KhanHuangLiangModel

from src.models.material_model import MaterialModel
from src.models.modified_johnson_cook_model import ModifiedJohnsonCookModel
from src.models.zerilli_armstrong_bcc_model import ZerilliArmstrongBCCModel
from src.models.zerilli_armstrong_fcc_model import ZerilliArmstrongFCCModel


class Plotter:
    pass


labels = {
    JohnsonCookModel: "JC model to data fitness",
    ModifiedJohnsonCookModel: "MJC model to data fitness",
    KhanHuangLiangModel: "KHL model to data fitness",
    ZerilliArmstrongBCCModel: "ZA-BCC model to data fitness",
    ZerilliArmstrongFCCModel: "ZA-FCC model to data fitness"
}


def plot(df: pd.DataFrame, model: MaterialModel, filename: str):
    dfc = df.copy()
    dfc = dfc.sort_values(by="strain")
    cls = model.__class__

    dfc_c = dfc.copy()
    dfc_c['stress'] = np.nan
    dfc_c['strain'] = dfc_c['strain'] * 4 / 3

    dfc['comp_stress'] = dfc.apply(lambda row: model(row['strain'], row['strain_rate'], row['temperature']), axis=1)
    dfc_c['comp_stress'] = dfc_c.apply(lambda row: model(row['strain'], row['strain_rate'], row['temperature']), axis=1)

    dfc['stress'] = dfc['stress'] / 1e6
    dfc['comp_stress'] = dfc['comp_stress'] / 1e6
    dfc_c['stress'] = dfc_c['stress'] / 1e6
    dfc_c['comp_stress'] = dfc_c['comp_stress'] / 1e6

    grouped = dfc.groupby(['strain_rate', 'temperature'])
    grouped_c = dfc_c.groupby(['strain_rate', 'temperature'])

    ncols = 4
    nrows = int(np.ceil(grouped.ngroups / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(24, 18), sharey=False, facecolor='1.0')

    flattened_axes = axes.flatten()
    for ax in flattened_axes:
        ax.set_visible(False)

    for (key, ax) in zip(grouped.groups.keys(), axes.flatten()):
        ax.set_visible(True)
        grouped_c.get_group(key).plot(x='strain', y='comp_stress', c='red', ax=ax, label='Model')
        grouped.get_group(key).plot(x='strain', y='stress', ax=ax, label='Experiment')
        ax.set_title(labels[cls])
        ax.set_xlabel('True strain')
        ax.set_ylabel('True stress [MPa]')

        box_text = 'Strain rate: {}s-1\nTemperature: {}K'.format(*key)
        text_box = AnchoredText(box_text, frameon=True, loc='lower left', pad=0.1)
        plt.setp(text_box.patch, facecolor='white', alpha=0.85)
        ax.add_artist(text_box)
        ax.legend(loc='lower right', ncol=1, framealpha=0.85, edgecolor='black', fancybox=False)

        plt.subplots_adjust(wspace=0.3, hspace=0.35)
    plt.savefig(filename)
    plt.close(fig)

