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
      # X.append(interpolate_to_512(dados[i:i + input_len]))
      X.append(dados[i:i + input_len])
      y.append(dados[i + input_len:i + input_len + pred_len])
  return np.array(X), np.array(y)


def train_epoch(model, optimizer, criterion, X_train, y_train, args):
  model.train()
  total_loss = 0
  print(5)

  """
  permutation = torch.randperm(X_train.size(0))

    for i in range(0, X_train.size(0), args.batch_size):
      indices = permutation[i:i+args.batch_size]
      print(indices)
      print(X_train[indices])
      batch_x, batch_y = X_train[indices], y_train[indices]

      optimizer.zero_grad()
      outputs = model(batch_x)
      loss = criterion(outputs, batch_y)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

  return total_loss / (X_train.size(0) // args.batch_size)
  """

  for i in range(0, len(X_train), args.batch_size):
    print(f"i {i}")
    batch_x = X_train[i:i+args.batch_size]
    batch_y = y_train[i:i+args.batch_size]
    print(6)
    batch_x_cpu = batch_x.cpu()
    batch_y_cpu = batch_y.cpu()

    print(f"batch_x.shape {batch_x.shape}")
    print(f"len(batch_x.shape) {len(batch_x.shape)}")
    if len(batch_x_cpu.shape) == 2:
      batch_x_cpu = batch_x_cpu.unsqueeze(-1)
    
    print(7)
    outputs = model.customTrain(batch_x_cpu)
    # outputs = torch.from_numpy(outputs).squeeze()
    print(8)
    
    batch_y_cpu,d,mu,std=model.dataTool.dataTransformationBatch(batch_y_cpu)
    
    print(outputs.shape)
    print(batch_y.shape)
    loss = criterion(outputs.cpu(), batch_y_cpu)
    print(9)
    optimizer.zero_grad()
    print(10)
    loss.backward()
    print(11)
    optimizer.step()
    print(12)
    total_loss += loss.item()

  print(13)
  return total_loss / (len(X_train) / args.batch_size)


def evaluate(model, criterion, X_val, y_val, args):
  model.eval()
  total_loss = 0
  print(15)
    
  with torch.no_grad():  
    """
    val_outputs = model(X_val)
    val_loss = criterion(val_outputs, y_val).item()
  
  return val_loss
  """
    for i in range(0, len(X_val), args.batch_size):
      print(f"i {i}")
      batch_x = X_val[i:i+args.batch_size]
      batch_y = y_val[i:i+args.batch_size]
      print(16)
      batch_x_cpu = batch_x.cpu()

      print(f"batch_x.shape {batch_x.shape}")
      print(f"len(batch_x.shape) {len(batch_x.shape)}")
      if len(batch_x_cpu.shape) == 2:
        batch_x_cpu = batch_x_cpu.unsqueeze(-1)
      
      print(17)
      outputs = model.inference(batch_x_cpu)
      print(18)
      loss = criterion(outputs, batch_y)
      print(19)
      total_loss += loss.item()
  
  print(20)
  return total_loss / (len(X_val) / args.batch_size)


def train(model, optimizer, criterion, X_train, y_train, X_val, y_val, args, len_valid_data, max_epochs=50, patience=5):
    print(1)
    # device = args.device
    best_val_loss = float('inf')
    epochs_no_improve = 0
    print(2)

    for epoch in range(max_epochs):
      print(f"epoch {epoch}")
      print(3)
      avg_train_loss = train_epoch(model, optimizer, criterion, X_train, y_train, args)

      print(14)
      avg_val_loss = evaluate(model, criterion, X_val, y_val, args)

      print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

      if avg_val_loss < best_val_loss:
        print(21)
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save({'model': model.state_dict(), 'args': args}, f"ViTime_finetuned{len_valid_data}.pth")
        print(22)
      else:
        print(21)
        epochs_no_improve += 1
        print(f"epochs with no improve {epochs_no_improve}")
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

  # X_train = torch.tensor(X_train, dtype=torch.float32).to(device).reshape(-1, 1, 4, 128)
  X_train = torch.tensor(X_train, dtype=torch.float32).cpu().unsqueeze(-1)
  print(f"X_train shape: {X_train.shape}")
  y_train = torch.tensor(y_train, dtype=torch.float32).cpu()
  print(f"y_train shape: {y_train.shape}")
  # X_val = torch.tensor(X_val, dtype=torch.float32).to(device).reshape(-1, 1, 4, 128)
  X_val = torch.tensor(X_val, dtype=torch.float32).cpu().unsqueeze(-1)
  print(f"X_val shape: {X_val.shape}")
  y_val = torch.tensor(y_val, dtype=torch.float32).cpu()
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

  # model = ViTime(args).to(device)
  model = ViTime(args).cpu()
  model.dataTool.device = device
  model.load_state_dict(checkpoint['model'])
  # model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.MSELoss()

  train(model, optimizer, criterion, X_train, y_train, X_val, y_val, args, len(valid_data))


if __name__ == "__main__":
  main()









