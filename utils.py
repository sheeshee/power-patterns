import os
import pandas as pd
import numpy as np
import scipy.ndimage
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt



DATA_PATH = os.getcwd()+'/data'
ENERGY_COL_NAME = 'out.electricity.total.energy_consumption'
N_BINS = 30


def load_data() -> pd.DataFrame:
    data = {}
    for filepath in tqdm(_file_list()):
        df = pd.read_parquet(filepath, engine='pyarrow')
        df = df.reset_index(drop=True).set_index('timestamp')
        bu_id = _get_building_name(filepath)
        data[bu_id] = df[ENERGY_COL_NAME]
    return pd.DataFrame(data)


def make_heatmap(energy: pd.Series, n_bins: int = N_BINS, time_step='1h') -> pd.DataFrame:
    grouped = group_by_time(energy, time_step)
    output = pd.DataFrame()
    for key, group in grouped:
        values = group['energy'].values
        values = values - values.min()
        values = values / values.max()
        density, bins = np.histogram(
            values,
            n_bins,
            range=(0, 1),
            density=True
        )
        output[str(key)] = density
    return output

def group_by_time(energy, time_step):
    energy = energy.resample(time_step).mean()
    df = pd.DataFrame(energy.rename('energy'))
    df['time'] = df.index.time
    grouped = df.groupby('time')
    return grouped


def smooth_heatmap(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    heatmap = scipy.ndimage.uniform_filter(df.values, size=window)
    heatmap = heatmap / df.max().max()
    return pd.DataFrame(heatmap, index=df.index, columns=df.columns)


def plot_heatmap(df: pd.DataFrame):
    ax = sns.heatmap(df, cmap='viridis')
    ax.invert_yaxis()
    plt.show()


def _file_list() -> list[str]:
    files = [f'{DATA_PATH}/{obj}' for obj in os.listdir(DATA_PATH)]
    return files


def _get_building_name(filepath):
    return os.path.basename(filepath).split('.')[0]
