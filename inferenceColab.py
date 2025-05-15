import argparse
import matplotlib.pyplot as plt
from model.model import ViTime
import numpy as np
import pandas as pd
import torch
from scipy import interpolate
from sklearn.metrics import mean_squared_error
import time
import tracemalloc


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


def plotar_grafico(dados, previstos, titulo, caminho_grafico):
  plt.figure(figsize=(20, 8))
  plt.plot(dados, label='Real', color='blue')

  inicio_previsao = len(dados) - len(previstos)
  posicao_previstos = range(inicio_previsao, inicio_previsao + len(previstos))
  
  plt.plot(posicao_previstos, previstos, label='Previsão', color='red')
  plt.title(titulo, fontsize=24)
  plt.legend(fontsize=16)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.grid(True)
  plt.savefig(caminho_grafico, dpi=300)
  plt.close()


def calcular_mse_rmse(real, previsto):
  if (len(real) != len(previsto)):
    print("Os dados não têm a mesma quantidade de leituras")
    return
  
  mse = mean_squared_error(real, previsto)
  rmse = np.sqrt(mse)
  return mse, rmse


def tempo_de_leituras(n):
  total_segundos = n * 5
  
  if total_segundos < 60:
    return f"{total_segundos}_seg"
  
  minutos = total_segundos // 60
  return f"{minutos}_min"


def salvar_metricas_csv(nome_modelo, mse, rmse, tempos, picos_memoria, repeticoes, caminho_csv):
  tempo_medio = sum(tempos) / repeticoes
  std_tempo = np.std(tempos)

  memoria_media = sum(picos_memoria) / repeticoes
  memoria_max = max(picos_memoria)
  std_memoria = np.std(picos_memoria)

  df = pd.DataFrame({
    'nome_modelo': [nome_modelo],
    'mse': [mse],
    'rmse': [rmse],
    'tempo_consumo (ns)': [tempo_medio],
    'desvio_tempo': [std_tempo],
    'memoria_consumo (mb)': [memoria_media],
    'memoria_max (mb)': [memoria_max],
    'desvio_memoria': [std_memoria]
  })
    
  try:
    df_existente = pd.read_csv(caminho_csv)
    df_existente = pd.concat([df_existente, df], ignore_index=True)
    df_existente.to_csv(caminho_csv, index=False)
  except FileNotFoundError:
    df.to_csv(caminho_csv, index=False)
  

def salvar_previsao_csv(nome_modelo, alvo, previstos, caminho_csv):
  df = pd.DataFrame({'Real': alvo, 'Previsto': previstos})
  df.insert(0, 'nome_modelo', '')
  df.at[0, 'nome_modelo'] = nome_modelo
  df.to_csv(caminho_csv, index=False)


def main(modelpath, qnt_alvo, input_type):
  model, args = carregar_modelo(modelpath)

  df = pd.read_csv('/content/drive/MyDrive/ViTime/BD_AC_CPUTemp.csv', sep=';')
  data = df['CPUTemp'].dropna().values
  total_len = len(data)

  if qnt_alvo > total_len:
    print(f"Quantidade de valores alvo maior que o tamanho do dataset.")
    return

  if input_type == "full":
    # todo o banco de dados, menos os N últimos
    entrada = data[: total_len - qnt_alvo]
  
  elif input_type == "fixed":
    # 512 leituras anteriores às inferências
    entrada = data[total_len - qnt_alvo - 512 : total_len - qnt_alvo] 
  
  elif input_type == "moving":
    # entrada adaptativa
    tamanho_entrada = max(512, 10 * qnt_alvo)
    entrada = data[total_len - qnt_alvo - tamanho_entrada : total_len - qnt_alvo]
  
  else:
    print("Tipo de entrada inválido.")
    return

  print(f"tamanho da entrada = {len(entrada)}")
  dados_plotagem = data[-(4*qnt_alvo):]
  alvo = data[-qnt_alvo:]

  repeticoes = 30
  tempos = []
  picos_memoria = []
  mse = 0
  rmse = 0
  
  for i in range(repeticoes):
    tracemalloc.start()

    inicio = time.perf_counter_ns()
    previstos = executar_previsao(entrada, alvo, model, args)
    fim = time.perf_counter_ns()

    atual, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tempos.append(fim - inicio)
    picos_memoria.append(pico / (1024 ** 2))

    mse_i, rmse_i = calcular_mse_rmse(alvo, previstos)
    mse += mse_i
    rmse += rmse_i

    print(f"n = {qnt_alvo}, i = {i}, t = {tempos[-1]}, mb = {picos_memoria[-1]:.5f}, mse = {mse_i}, rmse = {rmse_i}")
  print()

  nome_modelo = f"ViTime_{tempo_de_leituras(qnt_alvo)}_{input_type}"
  mse_medio = mse / repeticoes
  rmse_medio = rmse / repeticoes

  plotar_grafico(dados_plotagem, previstos, f"Inferência das Últimas {qnt_alvo} Leituras (zeroshot) - DB inteiro", f"{nome_modelo}.png")
  salvar_metricas_csv(nome_modelo, mse_medio, rmse_medio, tempos, picos_memoria, repeticoes, f"metricas_{input_type}.csv")
  salvar_previsao_csv(nome_modelo, alvo, previstos, f"{nome_modelo}.csv")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='ViTime model inference')
  parser.add_argument('--modelpath', type=str, required=True, help='Path to the model checkpoint file')
  parser.add_argument('--qnt_alvo', type=int, default=100, help='Number of target values to predict (default: 100)')
  parser.add_argument('--input_type', type=str, default='full', choices=['full', 'fixed', 'moving'], help='Input type the model should consider')
  args = parser.parse_args()
  main(args.modelpath, args.qnt_alvo, args.input_type)
