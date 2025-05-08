import argparse
import matplotlib.pyplot as plt
from model.model import ViTime
import numpy as np
import pandas as pd
import torch
from scipy import interpolate


def interpolate_to_512(original_sequence):
  n = len(original_sequence)
  x_original = np.linspace(0, 1, n)
  x_interpolated = np.linspace(0, 1, 512)
  f = interpolate.interp1d(x_original, original_sequence)
  return f(x_interpolated)


def inverse_interpolate(processed_sequence, original_length):
  z = int(original_length * 720 / 512)
  x_processed = np.linspace(0, 1, len(processed_sequence))
  x_inverse = np.linspace(0, 1, z)
  f_inverse = interpolate.interp1d(x_processed, processed_sequence)
  return f_inverse(x_inverse)
  
def carregar_modelo(modelpath):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  checkpoint = torch.load(modelpath, map_location=device)
  args = checkpoint['args']
  args.device = device
  args.flag = 'test'

  # Set upscaling parameters
  args.upscal = True  # True: max input length = 512, max prediction length = 720
                      # False: max input length = 1024, max prediction length = 1440
  model = ViTime(args=args)
  model.load_state_dict(checkpoint['model'])
  model.to(device)
  model.eval()
  return model, args


def executar_previsao(entrada, alvo, model, args):
  interpolated_input = interpolate_to_512(entrada)
  args.realInputLength = len(interpolated_input)
  predicted = model.inference(interpolated_input).flatten()
  predicted = inverse_interpolate(predicted, len(entrada)).flatten()
  predicted = predicted[:len(alvo)]
  return predicted


def plotar_grafico(dados, previstos, inicio_previsao, titulo, caminho_grafico):
  plt.figure(figsize=(20, 8))
  plt.plot(dados, label='Real', color='blue')
  posicao_previstos = range(inicio_previsao, inicio_previsao + len(previstos))
  plt.plot(posicao_previstos, previstos, label='Previsão', color='red')
  plt.title(titulo)
  plt.legend()
  plt.grid(True)
  plt.savefig(caminho_grafico, dpi=300)
  plt.close()


def salvar_previsao_csv(alvo, previstos, caminho_csv):
  df = pd.DataFrame({'Real': alvo, 'Previsto': previstos})
  df.to_csv(caminho_csv, index=False)


def main(modelpath, qnt_alvo):
  model, args = carregar_modelo(modelpath)

  # Leitura do CSV
  df = pd.read_csv('/content/drive/MyDrive/ViTime/BD_AC_CPUTemp.csv', sep=';')
  data = df['CPUTemp'].dropna().values
  total_len = len(data)

  entrada = data[:total_len - qnt_alvo]
  alvo = data[-qnt_alvo:]
  previstos = executar_previsao(entrada, alvo, model, args)
  plotar_grafico(data, previstos, len(entrada), f"Previsão das Últimas {qnt_alvo} Leituras", f"plot_qnt{qnt_alvo}.png")
  salvar_previsao_csv(alvo, previstos, f"previsao_qnt{qnt_alvo}.csv")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='ViTime model inference')
  parser.add_argument('--modelpath', type=str, required=True, help='Path to the model checkpoint file')
  parser.add_argument('--qnt_alvo', type=int, default=100, help='Number of target values to predict (default: 100)')
  args = parser.parse_args()
  main(args.modelpath, args.qnt_alvo)
