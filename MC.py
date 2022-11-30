import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats        
from scalecast.Forecaster import Forecaster
from scalecast.AnomalyDetector import AnomalyDetector
from scalecast.SeriesTransformer import SeriesTransformer
#https://towardsdatascience.com/anomaly-detection-for-time-series-with-monte-carlo-simulations-e43c77ba53c

sns.set(rc={'figure.figsize':(12,8)})

df = pdr.get_data_fred(
    'HOUSTNSA',
    start='1959-01-01',
    end='2022-05-01',
).reset_index()


f = Forecaster(
    y=df['HOUSTNSA'],
    current_dates=df['DATE']
)

f.plot()
plt.title('Original Series',size=16)
plt.show()


