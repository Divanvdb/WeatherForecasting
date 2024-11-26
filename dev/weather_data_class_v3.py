import xarray as xr
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import cartopy.crs as ccrs

import torch.nn as nn
from torch.utils.data import Dataset

from IPython.display import HTML

import torch

from typing import Tuple

class WeatherData(Dataset):

    """
    A dataset class for preparing wind speed data for machine learning models.

    Attributes:
        dataset (xr.Dataset): The xarray dataset containing wind speed data.
        window_size (int): The size of the window for creating features.
        steps (int): The number of forecasting steps.
        use_forcings (bool): Flag to indicate whether to use forcings.
        features (np.ndarray): Array of feature data.
        targets (np.ndarray): Array of target data.
        forcings (np.ndarray): Array of forcing data.
        time_values (np.ndarray): Array of time values corresponding to features.
        min_value (float): Minimum wind speed value for normalization.
        max_value (float): Maximum wind speed value for normalization.
        mean_value (float): Mean wind speed value for normalization.
        std_value (float): Standard deviation of wind speed for normalization.
        X_train (np.ndarray): Training features.
        X_test (np.ndarray): Testing features.
        y_train (np.ndarray): Training targets.
        y_test (np.ndarray): Testing targets.
        F_train (np.ndarray): Training forcings.
        F_test (np.ndarray): Testing forcings.
        X_train_t (torch.Tensor): Normalized training features as tensors.
        y_train_t (torch.Tensor): Normalized training targets as tensors.
        X_test_t (torch.Tensor): Normalized testing features as tensors.
        y_test_t (torch.Tensor): Normalized testing targets as tensors.
        F_train_t (torch.Tensor): Training forcings as tensors.
        F_test_t (torch.Tensor): Testing forcings as tensors.
    """

    def __init__(self, dataset: xr.Dataset, window_size: int = 24, steps: int = 3, auto: bool = True, use_forcings: bool = True, intervals: int = 1, data_split: str = 'train'):

        """
        Initializes the WeatherData object.

        Args:
            dataset (xr.Dataset): The xarray dataset containing wind speed data.
            window_size (int): The size of the window for creating features. Default is 24.
            steps (int): The number of forecasting steps. Default is 3.
            auto (bool): Flag to automatically window and normalize data. Default is False.
            use_forcings (bool): Flag to indicate whether to use forcings. Default is False.
        """
        
        self.dataset = dataset
        self.window_size = window_size
        self.steps = steps
        self.calculate_wind_speed()
        self.dataset = self.dataset.sortby('latitude')

        self.min_value = self.dataset.wspd.min().item()
        self.max_value = self.dataset.wspd.max().item()

        self.mean_value = self.dataset.wspd.mean().item()
        self.std_value = self.dataset.wspd.std().item()

        self.use_forcings = use_forcings

        self.model = None

        # MLP input size
        self.input_size = self.window_size * self.dataset.latitude.size * self.dataset.longitude.size
        self.forcing_size = 2  
        self.output_size = 1 * self.dataset.latitude.size * self.dataset.longitude.size 

        self.data_split = data_split

        self.intervals = intervals

        # if intervals > 1:
        #         self.time_intervals(intervals)
        #         print('Time intervals applied')
                

        if auto:
            self.split_data()    
            print('Data split')
            self.normalize_data()    
            print('Data normalized')

    def __len__(self) -> int:
        """
        Returns the number of samples based on how many windows of size `window_size + steps`
        can fit into the dataset for the specified split.

        Returns:
            int: The number of valid windows that can fit into the specified dataset split.
        """

        if self.data_split == 'train':
            dataset_length = len(self.X_train)
        elif self.data_split == 'val':
            dataset_length = len(self.X_val)
        elif self.data_split == 'test':
            dataset_length = len(self.X_test)
        else:
            raise ValueError("data_split must be 'train', 'val', or 'test'")
        
        total_window_size = (self.window_size + self.steps)  * self.intervals
        num_windows = dataset_length - total_window_size + self.intervals  
        
        return max(0, num_windows)  

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample from the specified dataset split.

        Args:
            idx (int): The index of the sample to retrieve.
            data_split (str): The dataset split ('train', 'val', 'test'). Default is 'train'.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing features, forcings, and target.
        """

        if self.data_split == 'train':
            x = self.X_train_t[idx:idx + self.window_size * self.intervals:self.intervals]
            F = self.F_train_t[idx + self.window_size * self.intervals]
            y = self.X_train_t[idx + self.window_size * self.intervals:idx + (self.window_size + self.steps) * self.intervals: self.intervals] 
            return x, F, y
        elif self.data_split == 'val':
            x = self.X_val_t[idx:idx + self.window_size * self.intervals:self.intervals]
            F = self.F_val_t[idx + self.window_size * self.intervals]
            y = self.X_val_t[idx + self.window_size * self.intervals:idx + (self.window_size + self.steps) * self.intervals: self.intervals] 
            return x, F, y
        elif self.data_split == 'test':
            x = self.X_test_t[idx:idx + self.window_size * self.intervals:self.intervals]
            F = self.F_test_t[idx + self.window_size * self.intervals]
            y = self.X_test_t[idx + self.window_size * self.intervals:idx + (self.window_size + self.steps) * self.intervals: self.intervals] 
            return x, F, y
        else:
            raise ValueError("data_split must be 'train', 'val', or 'test'")

    def subset_data(self, coarsen: int = 1) -> None:

        """
        Subsets the dataset based on the specified coarsening factor. Only happens once and then dataset is saved as a .nc file.

        Args:
            coarsen (int): The coarsening factor for subsetting. Default is 1.

        Returns:
            None: Updates the dataset in place.
        """

        if coarsen > 1:
            lat_slice = slice(1, 33, coarsen)
            lon_slice = slice(3, 67, coarsen)
        else:
            lat_slice = slice(1, 33)  
            lon_slice = slice(3, 67)

        self.dataset = self.dataset.isel(latitude=lat_slice, longitude=lon_slice)

    def calculate_wind_speed(self) -> None:
        """
        Calculates wind speed from u and v components and adds it to the dataset.

        Returns:
            None: Updates the dataset in place with the wind speed variable.
        """

        self.dataset['wspd'] = np.sqrt(self.dataset.u**2 + self.dataset.v**2).astype(np.float32)
        self.dataset.attrs['wspd_units'] = 'm/s'
        # self.dataset['wdir'] = np.arctan2(self.dataset.v, self.dataset.u) * 180 / np.pi
        # self.dataset.attrs['wdir_units'] = 'degrees'

    def split_data(self, test_size: float = 0.1, val_size: float = 0.2, random_state: int = 42) -> None:
        """
        Splits the data into training, validation, and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
            val_size (float): Proportion of the dataset to include in the validation split from the training set. Default is 0.1.
            random_state (int): Random seed for reproducibility. Default is 42.

        Returns:
            None: Updates the instance attributes with training, validation, and testing sets.
        """
        
        
        # Create a numpy array of the dataset
        data = self.dataset.wspd.values
        forcings = np.stack([self.dataset.time.dt.hour.values, self.dataset.time.dt.month.values], axis=-1)
        time_values = self.dataset.time.values

        # Split the data into train, validation, and test sets

        self.X_train, self.X_test, self.F_train, self.F_test, self.T_train, self.T_test = train_test_split(data, forcings, time_values, test_size=test_size, shuffle=False)

        self.X_train, self.X_val, self.F_train, self.F_val, self.T_train, self.T_val = train_test_split(self.X_train, self.F_train, self.T_train, test_size=val_size, shuffle=False)

    def normalize_data(self, method: str = 'avg_std') -> None:
        """
        Normalizes the training, validation, and testing data using mean and standard deviation.

        Returns:
            None: Updates the instance attributes with normalized data as tensors.
        """

        self.X_train_t = self.normalize(self.X_train, method)
        self.X_val_t = self.normalize(self.X_val, method)
        self.X_test_t = self.normalize(self.X_test, method)

        # Convert to tensors
        self.X_train_t = torch.tensor(self.X_train_t).float()

        self.X_val_t = torch.tensor(self.X_val_t).float()

        self.X_test_t = torch.tensor(self.X_test_t).float()

        # Convert forcings to tensors
        self.F_train_t = torch.tensor(self.F_train).float()
        self.F_val_t = torch.tensor(self.F_val).float()
        self.F_test_t = torch.tensor(self.F_test).float()

    def normalize(self, data: np.ndarray, method: str = 'avg_std') -> np.ndarray:
        """
        Normalizes the given data using min-max normalization.

        Args:
            data (np.ndarray): The data to normalize.

        Returns:
            np.ndarray: The normalized data.
        """

        if method == 'min_max':
            return (data - self.min_value) / (self.max_value - self.min_value)
        else:
            return (data - self.mean_value) / self.std_value

    def plot_from_data(self, seed: int = 0, frame_rate: int = 16, levels: int = 10) -> HTML:
        """
        Plots features and targets from the windowed arrays for visualization.

        Args:
            seed (int): Seed for reproducibility in selecting samples. Default is 0.
            frame_rate (int): The frame rate for the animation. Default is 16.
            levels (int): Number of contour levels for the plot. Default is 10.

        Returns:
            HTML: An HTML object representing the animation.
        """
        bounds = [self.dataset.longitude.min().item(), self.dataset.longitude.max().item(), self.dataset.latitude.min().item(), self.dataset.latitude.max().item()]

        features = self.X_test[seed:seed + self.window_size * self.intervals:self.intervals]
        targets = self.X_test[seed + self.window_size * self.intervals:seed + (self.window_size + self.steps) * self.intervals: self.intervals]
        
        time_features = self.T_test[seed:seed + self.window_size * self.intervals:self.intervals]
        time_targets = self.T_test[seed + self.window_size * self.intervals:seed + (self.window_size + self.steps) * self.intervals: self.intervals]

        time_features = pd.to_datetime(time_features)
        time_targets = pd.to_datetime(time_targets)

        fig, axs = plt.subplots(1, 2, figsize=(21, 7), subplot_kw={'projection': ccrs.PlateCarree()})

        vmin = min(features.min().item(), targets.min().item())
        vmax = max(features.max().item(), targets.max().item())

        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)

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

    def assign_model(self, model: nn.Module) -> None:
        """
        Assign a model to the class instance.

        Args:
            model (nn.Module): A PyTorch model to assign for training and prediction.
        """
        self.model = model

    def load_model(self, file_path: str) -> None:
        """
        Load a model from a file.

        Args:
            file_path (str): Path to load the model from.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.load_state_dict(torch.load(file_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, X: torch.Tensor, F: torch.Tensor) -> np.ndarray:
        """
        Predict output based on input data.

        Args:
            X (torch.Tensor): Input data for prediction.
            F (torch.Tensor): Forcings data, such as hour and month (if used).

        Returns:
            np.ndarray: Model predictions.
        """

        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X).float()
            F = torch.tensor(F).float()
            if self.use_forcings:
                return self.model(X, F).numpy()
            else:
                return self.model(X).numpy()

    def autoregressive_predict(self, X: torch.Tensor, F: torch.Tensor, rollout_steps: int, unnormalize: bool = True, verbose: bool = False) -> np.ndarray:
        """
        Perform autoregressive predictions for multiple time steps.

        Args:
            X (torch.Tensor): Input data for prediction.
            F (torch.Tensor): Forcings data, such as hour and month.
            rollout_steps (int): Number of future steps to predict.
            unnormalize (bool): Whether to unnormalize the predictions. Default is True.
            verbose (bool): Whether to print intermediate shapes for debugging. Default is False.

        Returns:
            np.ndarray: Predictions for each time step.
        """

        self.model.eval()
        with torch.no_grad():
            
            # X = torch.tensor(X).float()
            F = torch.tensor(F).float()
            
            predictions = []

            current_input = X#.to(self.device)
            current_F = F#.to(self.device)
            
            for step in range(rollout_steps):
                
                if self.use_forcings:
                    next_pred = self.model(current_input, current_F).cpu().numpy()
                else:
                    try:
                        next_pred = self.model(current_input).cpu().numpy()
                    except:
                        next_pred = self.model(current_input).numpy()
                
                predictions.append(next_pred)
                
                next_pred_tensor = torch.tensor(next_pred).float()#.to(self.device) 

                if verbose:
                    print(current_input.shape, next_pred_tensor.shape)

                current_input = torch.cat((current_input[:, 1:], next_pred_tensor), dim=1)#.to(self.device)

                hour = current_F[0, 0].item()  # Extract the hour
                month = current_F[0, 1].item()  # Extract the month
                
                hour += 1
                if hour == 24:
                    hour = 0
                
                current_F = torch.tensor([[hour, month]]).float()#.to(self.device)

            predictions = np.array(predictions).reshape(rollout_steps, self.dataset.sizes['latitude'], self.dataset.sizes['longitude'])

            # Unnromalize the predictions
            if unnormalize:
                predictions = predictions * self.std_value + self.mean_value
            
            return predictions
        
    def plot_pred_target(self, seed: int = 0, frame_rate: int = 16, levels: int = 10) -> HTML:
        """
        Plot the predictions and targets with animations.

        Args:
            seed (int): Seed to select the test data for plotting. Default is 0.
            frame_rate (int): Frame rate for animation. Default is 16.
            levels (int): Number of contour levels for plots. Default is 10.

        Returns:
            HTML: An HTML object containing the animation of predictions and targets.
        """

        bounds = [self.dataset.longitude.min().item(), self.dataset.longitude.max().item(), self.dataset.latitude.min().item(), self.dataset.latitude.max().item()]
        targets = self.X_test[seed + self.window_size * self.intervals:seed + (self.window_size + self.steps) * self.intervals: self.intervals]
        time_values = self.T_test[seed + self.window_size * self.intervals:seed + (self.window_size + self.steps) * self.intervals: self.intervals]

        time_values = pd.to_datetime(time_values)

        predictions = self.autoregressive_predict(self.X_test_t[seed:seed + self.window_size * self.intervals:self.intervals].unsqueeze(0), self.F_test_t[seed + self.window_size * self.intervals].unsqueeze(0), self.steps)

        fig, axs = plt.subplots(2, 3, figsize=(21, 7), subplot_kw={'projection': ccrs.PlateCarree()})

        vmin = min(predictions.min().item(), targets.min().item())
        vmax = max(predictions.max().item(), targets.max().item())

        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.2)

        for ax in axs.flatten()[:-1]:
            ax.set_extent(bounds, crs=ccrs.PlateCarree())
            ax.coastlines()

        ax_last = fig.add_subplot(2, 3, 6)

        pred = axs[0, 0].contourf(self.dataset.longitude, self.dataset.latitude, predictions[0], levels=levels, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())
        tar = axs[0, 1].contourf(self.dataset.longitude, self.dataset.latitude, targets[0], levels=levels, vmin=vmin, vmax = vmax, transform=ccrs.PlateCarree())

        error = (predictions[0] - targets[0,0].squeeze()) 

        err = axs[0, 2].contourf(self.dataset.longitude, self.dataset.latitude, error.squeeze(), levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')

        perc_error = error / targets[0,0].squeeze() * 100
        perc_error = np.clip(perc_error, -100, 100)
        rmse = np.sqrt(error**2)

        perr = axs[1, 0].contourf(self.dataset.longitude, self.dataset.latitude, perc_error, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
        rms = axs[1, 1].contourf(self.dataset.longitude, self.dataset.latitude, rmse, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
        ax_last.scatter(targets[0].flatten(), predictions[0].flatten(), c=error, cmap='coolwarm')

        fig.colorbar(pred, ax=axs[0, 0], orientation='vertical', label='Wind Speed (m/s)')
        fig.colorbar(tar, ax=axs[0, 1], orientation='vertical', label='Wind Speed (m/s)')
        fig.colorbar(err, ax=axs[0, 2], orientation='vertical', label='Percentage Error (%)')
        fig.colorbar(perr, ax=axs[1, 0], orientation='vertical', label='Percentage Error (%)')
        fig.colorbar(rms, ax=axs[1, 1], orientation='vertical', label='Root Mean Squared Error (m/s)')

        ax_last.set_xlabel("Observed Wind Speed (m/s)")
        ax_last.set_ylabel("Forecasted Wind Speed (m/s)")

        def animate(i):
            for ax in axs.flatten()[:-1]:
                ax.clear()
                ax.coastlines()
            
            ax_last.clear()
            ax_last.set_xlabel("Observed Wind Speed (m/s)")
            ax_last.set_ylabel("Forecasted Wind Speed (m/s)")

            axs[0, 0].contourf(self.dataset.longitude, self.dataset.latitude, predictions[i], levels=levels, vmin=vmin, vmax = vmax)
            axs[0, 1].contourf(self.dataset.longitude, self.dataset.latitude, targets[i], levels=levels, vmin=vmin, vmax = vmax)
            
            error =  (predictions[i] - targets[i].squeeze())
            axs[0, 2].contourf(self.dataset.longitude, self.dataset.latitude, error, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
            
            perc_error = error / targets[i % self.steps].squeeze() * 100
            perc_error = np.clip(perc_error, -100, 100)
            rmse = np.sqrt(error**2)

            axs[1, 0].contourf(self.dataset.longitude, self.dataset.latitude, perc_error, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
            axs[1, 1].contourf(self.dataset.longitude, self.dataset.latitude, rmse, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm')
            ax_last.scatter(targets[i].flatten(), predictions[i].flatten(), c=error, cmap='coolwarm')

            axs[0, 0].set_title(f'Prediction {i} - {time_values[i].strftime("%Y-%m-%d %H:%M:%S")}')  
            axs[0, 1].set_title(f'Target - {time_values[i].strftime("%Y-%m-%d %H:%M:%S")}')
            axs[0, 2].set_title(f'Error - {time_values[i].strftime("%Y-%m-%d %H:%M:%S")}')
            axs[1, 0].set_title(f'Percentage Error - {time_values[i].strftime("%Y-%m-%d %H:%M:%S")}')
            axs[1, 1].set_title(f'Root Mean Squared Error - {time_values[i].strftime("%Y-%m-%d %H:%M:%S")}')
            ax_last.set_title(f'Error Scatter Plot - {time_values[i].strftime("%Y-%m-%d %H:%M:%S")}')

        frames = predictions.shape[0]

        interval = 1000 / frame_rate

        ani = FuncAnimation(fig, animate, frames=frames, interval=interval)

        plt.close(fig)

        return HTML(ani.to_jshtml())
