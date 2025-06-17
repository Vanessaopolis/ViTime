import json
import time
import torch
import argparse
import tracemalloc
import numpy as np
import pandas as pd

from scipy import interpolate
from model.model import ViTime
from sklearn.metrics import mean_squared_error


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


def calcular_mse_rmse(real, previsto):
  if (len(real) != len(previsto)):
    print("Os dados não têm a mesma quantidade de leituras")
    print(f"real: {len(real)}, previsto: {len(previsto)}")
    return
  
  mse = mean_squared_error(real, previsto)
  rmse = np.sqrt(mse)
  return mse, rmse


def tempo_de_leituras(n):
  segundos = n * 5
  
  if segundos < 60:
    return f"{segundos}seg"

  minutos = segundos // 60
  if minutos < 60:
    return f"{minutos}min"

  horas = minutos // 60
  return f"{horas}hrs"


def formata_lista(lista):
  return json.dumps(lista.tolist() if hasattr(lista, 'tolist') else lista)


def salvar_run_csv(nome_modelo, real, previsto, entrada, mse, rmse, tempo, pico_memoria, caminho_csv):
  df = pd.DataFrame({
    'nome_modelo': [nome_modelo],
    'real': [formata_lista(real)],
    'previsto': [formata_lista(previsto)],
    'entrada': [formata_lista(entrada)],
    'mse': [mse],
    'rmse': [rmse],
    'tempo_consumo (s)': [tempo],
    'memoria_consumo (mb)': [pico_memoria],
  })
    
  try:
    df_existente = pd.read_csv(caminho_csv)
    df_existente = pd.concat([df_existente, df], ignore_index=True)
    df_existente.to_csv(caminho_csv, index=False)
  except FileNotFoundError:
    df.to_csv(caminho_csv, index=False)


def main(modelpath, qnt_alvo, input_type):
  model, args = carregar_modelo(modelpath)

  df = pd.read_csv('/content/drive/MyDrive/ViTime/BD_AC_CPUTemp.csv', sep=';')
  data = df['CPUTemp'].dropna().values
  total_len = len(data)
  print(f"Tamanho do dataset {total_len}")

  if qnt_alvo > total_len:
    print(f"Quantidade de valores alvo maior que o tamanho do dataset.")
    return

  if input_type == "full":
    # todo o banco de dados, menos os N últimos
    inicio = 0
  
  elif input_type == "fixed":
    # 512 leituras anteriores às inferências
    inicio = total_len - qnt_alvo - 512
  
  elif input_type == "moving":
    # entrada adaptativa
    tamanho_entrada = max(512, 10 * qnt_alvo)
    inicio = max(0, total_len - qnt_alvo - tamanho_entrada)
  
  else:
    print("Tipo de entrada inválido.")
    return

  entrada = data[inicio : total_len - qnt_alvo]
  print(f"Tamanho da entrada {len(entrada)}")
  alvo = data[-qnt_alvo:]

  repeticoes = 30
  for i in range(repeticoes):
    nome_modelo = f"ViTime_{input_type}_{tempo_de_leituras(qnt_alvo)}_run{i+1}"
    tracemalloc.start()

    inicio = time.perf_counter_ns()
    previstos = executar_previsao(entrada, alvo, model, args)
    fim = time.perf_counter_ns()

    atual, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tempo = (fim - inicio) / (10 ** 9)
    pico_memoria = (pico / (1024 ** 2))

    mse, rmse = calcular_mse_rmse(alvo, previstos)

    print(f"n = {qnt_alvo}, modelo = {nome_modelo}, mse = {mse:.5f}, rmse = {rmse:.5f}, t = {tempo:.5f}, mb = {pico_memoria:.5f}")
    salvar_run_csv(nome_modelo=nome_modelo, 
                    real=alvo, 
                    previsto=previstos, 
                    entrada=entrada, 
                    mse=mse,
                    rmse=rmse,
                    tempo=tempo,
                    pico_memoria=pico_memoria,
                    caminho_csv="resultado_inferencia.csv")
  print()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='ViTime model inference')
  parser.add_argument('--modelpath', type=str, required=True, help='Path to the model checkpoint file')
  parser.add_argument('--qnt_alvo', type=int, default=100, help='Number of target values to predict (default: 100)')
  parser.add_argument('--input_type', type=str, default='full', choices=['full', 'fixed', 'moving'], help='Input type the model should consider')
  args = parser.parse_args()
  main(args.modelpath, args.qnt_alvo, args.input_type)
