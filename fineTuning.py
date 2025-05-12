import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.model import ViTime
from scipy import interpolate
from sklearn.model_selection import train_test_split
from argparse import Namespace
import torch.serialization


def interpolate_to_512(original_sequence):
  n = len(original_sequence)
  if n == 512:
    return original_sequence
  x_original = np.linspace(0, 1, n)
  x_interpolated = np.linspace(0, 1, 512)
  f = interpolate.interp1d(x_original, original_sequence)
  return f(x_interpolated)


def preparar_dados(dados, input_len, pred_len):
  X, y = [], []
  for i in range(len(dados) - input_len - pred_len):
      X.append(interpolate_to_512(dados[i:i + input_len]))
      y.append(dados[i + input_len:i + input_len + pred_len])
  return np.array(X), np.array(y)


def train(model, optimizer, criterion, X_train, y_train, X_val, y_val, args, len_valid_data, max_epochs=50, patience=5):
    print(1)
    device = args.device
    best_val_loss = float('inf')
    epochs_no_improve = 0
    print(2)

    for epoch in range(max_epochs):
      print(3)
      model.train()
      print(4)
      permutation = torch.randperm(X_train.size(0))
      epoch_loss = 0
      print(5)

      for i in range(0, X_train.size(0), args.batch_size):
        print(6)
        indices = permutation[i:i+args.batch_size]
        print(indices)
        print(7)
        print(X_train[indices])
        batch_x, batch_y = X_train[indices], y_train[indices]
        print(8)

        optimizer.zero_grad()
        print(9)
        outputs = model(batch_x)
        print(10)
        loss = criterion(outputs, batch_y)
        print(11)
        loss.backward()
        print(12)
        optimizer.step()
        print(13)
        epoch_loss += loss.item()
        print(14)

      avg_train_loss = epoch_loss / (X_train.size(0) // args.batch_size)
      print(15)

      # Validação
      model.eval()
      with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()

      print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

      if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save({'model': model.state_dict(), 'args': args}, f"ViTime_finetuned{len_valid_data}.pth")
      else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
          print("Early stopping.")
          break


def main():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  df = pd.read_csv('/content/drive/MyDrive/ViTime/BD_AC_CPUTemp.csv', sep=';')
  data = df['CPUTemp'].dropna().values
  valid_data = data[: 4992]
  
  input_len = 512
  pred_len = 720

  X, y = preparar_dados(valid_data, input_len, pred_len)
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

  X_train = torch.tensor(X_train, dtype=torch.float32).to(device).reshape(-1, 1, 4, 128)
  print(f"X_train shape: {X_train.shape}")
  y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
  print(f"y_train shape: {y_train.shape}")
  X_val = torch.tensor(X_val, dtype=torch.float32).to(device).reshape(-1, 1, 4, 128)
  print(f"X_val shape: {X_val.shape}")
  y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
  print(f"y_val shape: {y_val.shape}")
  print()

  checkpoint = torch.load('/content/drive/MyDrive/ViTime/ViTime_V1.pth', map_location=device)
  args = checkpoint['args']
  print(vars(args))
  print()
  args.device = device
  args.flag = 'train'
  args.upscal = True
  args.batch_size = 16
  # args.h = 1
  # args.size = [212, 0 , 300]
  print(vars(args))
  print()

  model = ViTime(args).to(device)
  model.load_state_dict(checkpoint['model'])
  model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.MSELoss()

  train(model, optimizer, criterion, X_train, y_train, X_val, y_val, args, len(valid_data))


if __name__ == "__main__":
  main()









