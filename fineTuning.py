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
      X.append(dados[i:i + input_len])
      y.append(dados[i + input_len:i + input_len + pred_len])
  return np.array(X), np.array(y)


def train_epoch(model, optimizer, criterion, X_train, y_train, args):
  model.train()
  total_loss = 0

  for i in range(0, len(X_train), args.batch_size):
    batch_x = X_train[i:i+args.batch_size]
    batch_y = y_train[i:i+args.batch_size]
    
    batch_x = batch_x.to(args.device) #.cpu()
    if len(batch_x.shape) == 2:
      batch_x = batch_x.unsqueeze(-1)

    outputs = model.customTrain(batch_x)

    if len(batch_y.shape) == 2:
      batch_y = batch_y.unsqueeze(-1)
    batch_y,d,mu,std=model.dataTool.dataTransformationBatch(batch_y)
    batch_y = batch_y.to(args.device)
    
    loss = criterion(outputs, batch_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

  return total_loss / (len(X_train) / args.batch_size)


def evaluate(model, criterion, X_val, y_val, args):
  model.eval()
  total_loss = 0
    
  with torch.no_grad():  
    for i in range(0, len(X_val), args.batch_size):
      batch_x = X_val[i:i+args.batch_size]
      batch_y = y_val[i:i+args.batch_size]
      
      batch_x = batch_x.to(args.device)
      if len(batch_x.shape) == 2:
        batch_x = batch_x.unsqueeze(-1)
      
      outputs = model.customTrain(batch_x)
      
      if len(batch_y.shape) == 2:
        batch_y = batch_y.unsqueeze(-1)    
      batch_y,d,mu,std=model.dataTool.dataTransformationBatch(batch_y)
      batch_y = batch_y.to(args.device)

      loss = criterion(outputs, batch_y)
      
      total_loss += loss.item()
  
  return total_loss / (len(X_val) / args.batch_size)


def train(model, optimizer, criterion, X_train, y_train, X_val, y_val, args, len_valid_data, max_epochs=50, patience=5):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(max_epochs):
      avg_train_loss = train_epoch(model, optimizer, criterion, X_train, y_train, args)
      avg_val_loss = evaluate(model, criterion, X_val, y_val, args)

      print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

      if avg_val_loss < best_val_loss:
        print(21)
        """best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save({'model': model.state_dict(), 'args': args}, f"ViTime_finetuned{len_valid_data}.pth")
        print(22)"""
      else:
        print(23)
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
  pred_len = 512
  X, y = preparar_dados(valid_data, input_len, pred_len)
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

  X_train = torch.tensor(X_train, dtype=torch.float32).cpu().unsqueeze(-1)
  y_train = torch.tensor(y_train, dtype=torch.float32).cpu()
  X_val = torch.tensor(X_val, dtype=torch.float32).cpu().unsqueeze(-1)
  y_val = torch.tensor(y_val, dtype=torch.float32).cpu()

  checkpoint = torch.load('/content/drive/MyDrive/ViTime/ViTime_V1.pth', map_location=device)
  args = checkpoint['args']
  args.device = device
  args.flag = 'train'
  args.upscal = True
  args.batch_size = 2

  model = ViTime(args).to(device)

  model.dataTool.device = 'cpu'
  model.load_state_dict(checkpoint['model'])

  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.MSELoss()

  train(model, optimizer, criterion, X_train, y_train, X_val, y_val, args, len(valid_data))


if __name__ == "__main__":
  main()









