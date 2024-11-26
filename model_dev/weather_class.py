import xarray as xr
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import cartopy.crs as ccrs

from torch.utils.data import Dataset

from IPython.display import HTML

import torch

from typing import Tuple

class WeatherData(Dataset):
    def __init__(self, dataset: xr.Dataset, window_size: int = 24, steps: int = 3, auto: bool = True, coarsen: int = 1, use_forcings: bool = True, data_split: str = 'train'):
        
        self.dataset = dataset
        self.window_size = window_size
        self.steps = steps

        self.min_value = self.dataset.wind_speed.min().item()
        self.max_value = self.dataset.wind_speed.max().item()

        self.mean_value = self.dataset.wind_speed.mean().item()
        self.std_value = self.dataset.wind_speed.std().item()

        self.land_sea_mask = np.load('C:/Users/23603526/Documents/GitHub/WeatherForecasting/data/land_sea_mask.npy')

        self.use_forcings = use_forcings

        self.input_size = self.window_size * self.dataset.latitude.size * self.dataset.longitude.size
        self.forcing_size = 2  
        self.output_size = 1 * self.dataset.latitude.size * self.dataset.longitude.size 

        self.data_split = data_split

        if auto:
            if coarsen > 1:
                self.subset_data(coarsen)
                
            self.split_data()
            self.normalize_data()


    def __len__(self) -> int:

        if self.data_split == 'train':
            dataset_length = len(self.X_train)
        elif self.data_split == 'val':
            dataset_length = len(self.X_val)
        elif self.data_split == 'test':
            dataset_length = len(self.X_test)
        else:
            raise ValueError("data_split must be 'train', 'val', or 'test'")
        
        total_window_size = (self.window_size + self.steps)
        num_windows = dataset_length - total_window_size
        
        return max(0, num_windows)  


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        M = torch.tensor(self.land_sea_mask).float()

        if self.data_split == 'train':
            x = self.X_train_t[idx:idx + self.window_size]
            F = self.F_train_t[idx + self.window_size]
            y = self.X_train_t[idx + self.window_size:idx + self.window_size + self.steps] 
            return x, F, M, y
        elif self.data_split == 'val':
            x = self.X_val_t[idx:idx + self.window_size]
            F = self.F_val_t[idx + self.window_size]
            y = self.X_val_t[idx + self.window_size:idx + self.window_size + self.steps] 
            return x, F, M, y
        elif self.data_split == 'test':
            x = self.X_test_t[idx:idx + self.window_size]
            F = self.F_test_t[idx + self.window_size]
            y = self.X_test_t[idx + self.window_size:idx + self.window_size + self.steps] 
            return x, F, M, y
        else:
            raise ValueError("data_split must be 'train', 'val', or 'test'")


    def subset_data(self, coarsen: int = 1) -> None:

        if coarsen > 1:
            lat_slice = slice(1, 33, coarsen)
            lon_slice = slice(3, 67, coarsen)

        self.dataset = self.dataset.isel(latitude=lat_slice, longitude=lon_slice)


    def split_data(self, test_size: float = 0.1, val_size: float = 0.2, random_state: int = 42) -> None:
        
        data = self.dataset.wind_speed.values.squeeze()
        forcings = np.stack([self.dataset.time.dt.hour.values, self.dataset.time.dt.month.values], axis=-1)
        time_values = self.dataset.time.values


        self.X_train, self.X_test, self.F_train, self.F_test, self.T_train, self.T_test = train_test_split(data, forcings, time_values, test_size=test_size, shuffle=False)

        self.X_train, self.X_val, self.F_train, self.F_val, self.T_train, self.T_val = train_test_split(self.X_train, self.F_train, self.T_train, test_size=val_size, shuffle=False)


    def normalize_data(self, method: str = 'min_max') -> None:

        self.X_train_t = self.normalize(self.X_train, method)
        self.X_val_t = self.normalize(self.X_val, method)
        self.X_test_t = self.normalize(self.X_test, method)

        self.X_train_t = torch.tensor(self.X_train_t).float()

        self.X_val_t = torch.tensor(self.X_val_t).float()

        self.X_test_t = torch.tensor(self.X_test_t).float()

        self.F_train_t = torch.tensor(self.F_train).float()
        self.F_val_t = torch.tensor(self.F_val).float()
        self.F_test_t = torch.tensor(self.F_test).float()


    def normalize(self, data: np.ndarray, method: str = 'avg_std') -> np.ndarray:

        if method == 'min_max':
            return (data - self.min_value) / (self.max_value - self.min_value)
        else:
            return (data - self.mean_value) / self.std_value


    def unnormalize(self, data: np.ndarray, method: str = 'avg_std') -> np.ndarray:
            
            if method == 'min_max':
                return data * (self.max_value - self.min_value) + self.min_value
            else:
                return data * self.std_value + self.mean_value
            

    def plot_from_data(self, seed: int = 0, frame_rate: int = 16, levels: int = 10) -> HTML:
        
        bounds = [self.dataset.longitude.min().item(), self.dataset.longitude.max().item(), self.dataset.latitude.min().item(), self.dataset.latitude.max().item()]

        features = self.X_test[seed:seed + self.window_size]
        targets = self.X_test[seed + self.window_size:seed + self.window_size + self.steps]
        
        time_features = self.T_test[seed:seed + self.window_size]
        time_targets = self.T_test[seed + self.window_size:seed + self.window_size + self.steps]

        time_features = pd.to_datetime(time_features)
        time_targets = pd.to_datetime(time_targets)

        fig, axs = plt.subplots(1, 2, figsize=(21, 7), subplot_kw={'projection': ccrs.PlateCarree()})

        vmin = min(features.min().item(), targets.min().item())
        vmax = max(features.max().item(), targets.max().item())

        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)

        print('Features:', features.shape, '\nTargets:', targets.shape)

        for ax in axs:
            ax.set_extent(bounds, crs=ccrs.PlateCarree())
            ax.coastlines()


        feat = axs[0].contourf(self.dataset.longitude, self.dataset.latitude, features[0], levels=levels, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())
        tar = axs[1].contourf(self.dataset.longitude, self.dataset.latitude, targets[0], levels=levels, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())
        axs[1].set_title('Target')

        fig.colorbar(feat, ax=axs[0], orientation='vertical', label='Wind Speed (m/s)')
        fig.colorbar(tar, ax=axs[1], orientation='vertical', label='Wind Speed (m/s)')

        def animate(i):
            axs[0].clear()
            axs[0].coastlines()

            axs[0].contourf(self.dataset.longitude, self.dataset.latitude, features[i], levels=levels, vmin=vmin, vmax = vmax)

            axs[0].set_title(f'Window {i} - {time_features[i].strftime("%Y-%m-%d %H:%M:%S")}')
            if self.steps > 1:
                axs[1].contourf(self.dataset.longitude, self.dataset.latitude, targets[i % self.steps], levels=levels, vmin=vmin, vmax = vmax)
                axs[1].set_title(f'Target - {time_targets[i % self.steps].strftime("%Y-%m-%d %H:%M:%S")}')
            # return pcm

            
        frames = features.shape[0]

        interval = 1000 / frame_rate

        ani = FuncAnimation(fig, animate, frames=frames, interval=interval)

        plt.close(fig)

        return HTML(ani.to_jshtml())


    def test_class(self) -> None:
        print('self.X_train:', self.X_train.shape, 'self.X_val:', self.X_val.shape, 'self.X_test:', self.X_test.shape)
        print('self.F_train:', self.F_train.shape, 'self.F_val:', self.F_val.shape, 'self.F_test:', self.F_test.shape)

        print('self.X_train_t:', self.X_train_t.shape, 'self.X_val_t:', self.X_val_t.shape, 'self.X_test_t:', self.X_test_t.shape)
        print('self.F_train_t:', self.F_train_t.shape, 'self.F_val_t:', self.F_val_t.shape, 'self.F_test_t:', self.F_test_t.shape)

        print('self.input_size:', self.input_size, 'self.forcing_size:', self.forcing_size, 'self.output_size:', self.output_size)

   