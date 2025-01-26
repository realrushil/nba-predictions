import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import os


data_dir = 'team_csvs'
features = ['EFG', 'FTR', 'OREB_PCT', 'TOV_PCT', 'OFF_RTG', 'DEF_RTG']

def create_game_windows(data, window_size=10):
    windows = []
    for i in range(len(data) - window_size + 1):
        window = data.iloc[i:i + window_size][features].values.flatten()
        windows.append(window)
    return np.array(windows)

all_data = []
scaler = StandardScaler()
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        team_data = pd.read_csv(os.path.join(data_dir, file))
        team_data[features] = scaler.fit_transform(team_data[features])
        team_windows = create_game_windows(team_data)
        all_data.append(team_windows)

all_data = np.vstack(all_data)
model = models.Sequential([
    layers.InputLayer(input_shape=(all_data.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(60, activation='linear'),
])

model.compile(optimizer='adam', loss='mse')
model.fit(all_data, all_data, epochs=16, batch_size=32)
embeddings = model.predict(all_data)

np.save('team_embeddings.npy', embeddings)