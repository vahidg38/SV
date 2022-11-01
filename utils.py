import scipy.io as sio
from math import ceil
from statistics import mean, stdev
import pandas as pd
import numpy as np

def fault_generation(df_real, type='bias', sensor='PM25', magnitude=0, start=0, stop=100):
  if (start < 0) or (stop > len(df_real)) or (start == stop):
    raise Exception("Inappropriate boundries.")
  #faulty = df_real[sensor].values
  faulty = []
  if sensor == 'PM25':
    faulty = df_real.PM25.values
  elif sensor == 'PM10':
    faulty = df_real.PM10.values
  elif sensor == 'CO2':
    faulty = df_real.CO2.values
  elif sensor == 'Temp':
    faulty = df_real.Temp.values
  elif sensor == 'Humidity':
    faulty = df_real.Humidity.values
  else:
    raise Exception("Inappropriate sensor name.")

  if (type == 'bias'):
    for i in range(start, stop):
      faulty[i] += magnitude

  elif (type == 'Complete_failure'):
    for i in range(start, stop):
      faulty[i] = magnitude

  elif (type == 'Drift'):
    m1 = mean(df_real[sensor])
    s1 = stdev(df_real[sensor])

    m2 = m1 + magnitude
    s2 = s1 + magnitude

    for i in range(start, stop):
      faulty[i] = m2 + (faulty[i] - m1) * (s2 / s1)

  elif (type == 'Degradation'):
    mu, sigma = 0, 0.1
    noise = np.random.normal(mu, sigma, [stop - start, ])
    faulty += magnitude * noise

  else:
    raise Exception("Inappropriate failure type.")
  df_real.drop(sensor, axis=1, inplace=True)
  df_real[sensor] = faulty

  return df_real

def data_loading(mat_file='IAQ_2month_Vah.mat', train_test_ratio=4):
  d = sio.loadmat(mat_file)
  d = d["Platfrom_C"]

  train_data=[]
  test_data=[]

  t= train_test_ratio/(train_test_ratio+1)

  for i in range(ceil(d.shape[0]*t)):
    train_data.append(d[i])
  for j in range(ceil(d.shape[0]*t), d.shape[0]):
    test_data.append(d[i])

  return train_data, test_data