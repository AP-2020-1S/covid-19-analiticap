print("Estoy ejecutando")

import requests
from pandas.io.json import json_normalize
import pandas as pd

url="https://www.datos.gov.co/resource/gt2j-8ykr.json?$limit=1000000000"
Datos=pd.read_json(url,convert_dates=['fecha_de_notificaci_n','fecha_de_muerte','fis','fecha_diagnostico','fecha_recuperado','fecha_reporte_web'])

Datos.rename(columns={'id_de_caso':'id_caso','fecha_de_notificaci_n':'fecha_notificacion','c_digo_divipola':'codigo_municipio','ciudad_de_ubicaci_n':'ciudad',
                      'atenci_n':'atencion','tipo':'tipo_contagio','estado':'estado','pa_s_de_procedencia':'pais_procedencia','fis':'fecha_sintomas',
                      'tipo_recuperaci_n':'tipo_recuperacion','fecha_de_muerte':'fecha_muerte'},inplace=True)

# filtrando ciudades
Datos1=Datos[Datos['ciudad'].isin(['Medellín'])]
Datos1.reset_index(inplace=True,drop=True)

print("Leí datos")

print("Calculando casos nuevos reportados")
# tabla con ciudades filtradas para modelar casos nuevos
tabla_nuevos=pd.pivot_table(Datos1,index=['fecha_reporte_web'],values=['id_caso'],aggfunc='count')
tabla_nuevos=pd.DataFrame(tabla_nuevos.to_records())
tabla_nuevos.rename(columns={'id_caso':'casos_nuevos_reportados'},inplace=True)
tabla_nuevos.loc[:,'casos_confirmados']=tabla_nuevos['casos_nuevos_reportados'].cumsum()

tabla_nuevos.plot(x='fecha_reporte_web',y='casos_confirmados',figsize=(10, 5)).figure.savefig('./images/Med7.png')

from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Serie normal
M2Nuevos=tabla_nuevos.set_index('fecha_reporte_web')
M2Nuevos=M2Nuevos.drop(['casos_confirmados'],axis=1)
plt.rcParams["figure.figsize"] = (10,5)

temp=M2Nuevos[-7:].values.reshape(-1,1)
sc=MinMaxScaler(feature_range=(0, 1))
temp=sc.fit_transform(temp)

from tensorflow.keras.models import load_model
model = load_model('./modelos/ModMedellinNuevos.h5')

a=([[[float(temp[i])] for i in range(0,len(temp))]])

b=[]
for i in range(1,15):
    pred = model.predict(a)
    b.append([float(pred)])
    a[0].pop(0)
    a[0].append([float(pred)])

MedellinNuevos=sc.inverse_transform(b)

import datetime
index=[M2Nuevos.index[-1]+datetime.timedelta(days=i) for i in range(1,15)]
MedellinNuevos=pd.DataFrame(data=MedellinNuevos, index=index, columns=['prediccion_casos_nuevos_reportados'])

M2Nuevos.plot().figure.savefig('./images/Med1.png')

total=pd.concat((M2Nuevos,MedellinNuevos),axis = 0)
total.plot()
plt.axvline(x=M2Nuevos.index[-1],color='y',linestyle='--')
plt.savefig('./images/Med2.png')

M2Confirmados=tabla_nuevos.set_index('fecha_reporte_web')
M2Confirmados=M2Confirmados.drop(['casos_nuevos_reportados'],axis=1)
MedellinNuevos.loc[:,'prediccion_casos_confirmados']=MedellinNuevos['prediccion_casos_nuevos_reportados'].cumsum()
MedellinConfirmados=MedellinNuevos.drop(['prediccion_casos_nuevos_reportados'],axis=1)
MedellinConfirmados['prediccion_casos_confirmados']=MedellinConfirmados['prediccion_casos_confirmados'].map(lambda x: x+float(M2Confirmados.iloc[-1].values))
total=pd.concat((M2Confirmados,MedellinConfirmados),axis = 0)
total.plot()
plt.axvline(x=M2Confirmados.index[-1],color='y',linestyle='--')
plt.savefig('./images/Med8.png')

print("Terminé nuevos")

print("Calculando casos fallecidos")
# tabla con ciudades filtradas para modelar casos muerte
tabla_muerte=pd.pivot_table(Datos1,index=['fecha_muerte'],values=['id_caso'],aggfunc='count')
tabla_muerte=pd.DataFrame(tabla_muerte.to_records())
tabla_muerte.rename(columns={'id_caso':'casos_muertes'},inplace=True)

from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Serie normal
M2Muerte=tabla_muerte.set_index('fecha_muerte')
plt.rcParams["figure.figsize"] = (10,5)

temp=M2Muerte[-7:].values.reshape(-1,1)
sc=MinMaxScaler(feature_range=(0, 1))
temp=sc.fit_transform(temp)

model = load_model('./modelos/ModMedellinMuerte.h5')

a=([[[float(temp[i])] for i in range(0,len(temp))]])

b=[]
for i in range(1,15):
    pred = model.predict(a)
    b.append([float(pred)])
    a[0].pop(0)
    a[0].append([float(pred)])

MedellinMuerte=sc.inverse_transform(b)

import datetime
index=[M2Muerte.index[-1]+datetime.timedelta(days=i) for i in range(1,15)]
MedellinMuerte=pd.DataFrame(data=MedellinMuerte, index=index, columns=['prediccion_casos_muerte'])

M2Muerte.plot().figure.savefig('./images/Med3.png')

total=pd.concat((M2Muerte,MedellinMuerte),axis = 0)
total.plot()
plt.axvline(x=M2Muerte.index[-1],color='y',linestyle='--')
plt.savefig('./images/Med4.png')

print("Terminé fallecidos")

print("Calculando casos recuperados")
# tabla con ciudades filtradas para modelar casos recuperado
tabla_recuperado=pd.pivot_table(Datos1,index=['fecha_recuperado'],values=['id_caso'],aggfunc='count')
tabla_recuperado=pd.DataFrame(tabla_recuperado.to_records())
tabla_recuperado.rename(columns={'id_caso':'casos_recuperado'},inplace=True)

from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Serie normal
M2Recuperado=tabla_recuperado.set_index('fecha_recuperado')
plt.rcParams["figure.figsize"] = (10,5)

temp=M2Recuperado[-7:].values.reshape(-1,1)
sc=MinMaxScaler(feature_range=(0, 1))
temp=sc.fit_transform(temp)

model = load_model('./modelos/ModMedellinRecuperado.h5')

a=([[[float(temp[i])] for i in range(0,len(temp))]])

b=[]
for i in range(1,15):
    pred = model.predict(a)
    b.append([float(pred)])
    a[0].pop(0)
    a[0].append([float(pred)])

MedellinRecuperado=sc.inverse_transform(b)

import datetime
index=[M2Recuperado.index[-1]+datetime.timedelta(days=i) for i in range(1,15)]
MedellinRecuperado=pd.DataFrame(data=MedellinRecuperado, index=index, columns=['prediccion_casos_recuperado'])

M2Recuperado.plot().figure.savefig('./images/Med5.png')

total=pd.concat((M2Recuperado,MedellinRecuperado),axis = 0)
total.plot()
plt.axvline(x=M2Recuperado.index[-1],color='y',linestyle='--')
plt.savefig('./images/Med6.png')

print("Terminé recuperados")

print("Estoy calculando SIR")
#calcula el tiempo de latencia de la enfermedad a través de un promedio ponderado

Datos1['Latencia']=Datos1['fecha_recuperado'] - Datos1['fecha_reporte_web']
Calculo_inicial_latencia1=Datos1.groupby('Latencia')['id_caso'].count()
latencia1=pd.DataFrame()
latencia1['días']=Calculo_inicial_latencia1.keys()
latencia1['días']=latencia1['días'].map(lambda x: str(x).replace('days',''))
latencia1['días']=latencia1['días'].map(lambda x: str(x).replace('+00:00:00',''))
latencia1['días']=latencia1['días'].map(lambda x: str(x).replace('00:00:00',''))
latencia1['días']=pd.to_numeric(latencia1['días'])
latencia1['cantidad_casos']=[linea for linea in Calculo_inicial_latencia1]
latencia1=latencia1[latencia1['días']>0]
total_casos1=latencia1['cantidad_casos'].sum()
latencia1['ponderado']=(latencia1['cantidad_casos']/total_casos1)*latencia1['días']
tiempo_latencia1=latencia1['ponderado'].sum()
tiempo_latencia1

# cálculo del parámetro gamma
gamma1=1/tiempo_latencia1
gamma1

tabla_nuevos=pd.pivot_table(Datos1,index=['fecha_reporte_web'],values=['id_caso'],aggfunc='count')
tabla_nuevos=pd.DataFrame(tabla_nuevos.to_records())
tabla_nuevos.rename(columns={'id_caso':'casos_nuevos_reportados'},inplace=True)
tabla_nuevos.loc[:,'casos_confirmados']=tabla_nuevos['casos_nuevos_reportados'].cumsum()

np.mean(tabla_nuevos['casos_nuevos_reportados'][(tabla_nuevos['fecha_reporte_web']>'2020-08-01') & (tabla_nuevos['fecha_reporte_web']<'2020-08-07')])

#Escenario 2: Ro=            R0 del 2 al 6 de agosto, dia 136, promedio 1030.2 infectados  =    1.712
Ro_2=1.712
Beta1_2=gamma1*Ro_2         

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total population, N.
N = 2933094
# Initial number of infected and recovered individuals, I0 and R0.
I0, R0 = 1, 0
# Everyone else, S0, is susceptible to infection initially.
S0 = N - I0 - R0
# A grid of time points (in days)
t = np.linspace(0, 300, 300)

Beta=Beta1_2

# The SIR model differential equations.
def deriv(y, t, N, Beta, gamma1):
    S, I, R = y
    dSdt = -Beta * S * I/N 
    dIdt = Beta * S * I/N  - gamma1 * I
    dRdt = gamma1 * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, Beta, gamma1))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd')
ax.plot(t, S, 'b', label='Susceptible')
ax.plot(t, I, 'r', label='Infected')
ax.plot(t, R, 'g', label='Recovered with immunity')
ax.set_xlabel('Time /days')
ax.grid(b=True, which='major', c='w', ls='-')
legend = ax.legend()
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.savefig('./images/Med9.png')

import datetime
a=[x for x in tabla_nuevos['fecha_reporte_web'].values]
b=[tabla_nuevos['fecha_reporte_web'].values[-1].astype('M8[ms]').astype('O')+datetime.timedelta(days=i) for i in range(1,(len(S)-len(tabla_nuevos)+1))]
fechas=a+b
tabla2=pd.DataFrame({'Susceptibles':np.round(S),'Infectados':np.round(I),'Recuperados':np.round(R),'Fecha':fechas})

print('Estoy ponderando los modelos')

numero_infectados=len(Datos1)
df_fallecidos=Datos1[Datos1["atencion"]=='Fallecido']
numero_muertes = len(df_fallecidos)
tasa_letalidad=(numero_muertes/numero_infectados)*100
tasa_letalidad=(round(tasa_letalidad,3))/100

tabla2['CasosDia']=tabla2['Infectados'].diff().fillna(tabla2['Infectados'].iloc[0])
tabla2['MuertesDia']=tabla2['CasosDia'].map(lambda x: x*tasa_letalidad)
tabla2['RecuperadosDia']=tabla2['Recuperados'].diff().fillna(tabla2['Recuperados'].iloc[0])

MedellinNuevos['NuevosSIR']=tabla2['CasosDia'][tabla2['Fecha'].isin(MedellinNuevos.index)].values
MedellinNuevos['PesoCorto']=[x/100 for x in [100,90,90,90,90,90,80,80,80,70,70,20,10,0]]
MedellinNuevos['PesoLargo']=[x/100 for x in [0,10,10,10,10,10,20,20,20,30,30,80,90,100]]
MedellinNuevos['prediccion_nuevos_final']=(MedellinNuevos['prediccion_casos_nuevos_reportados']*MedellinNuevos['PesoCorto']+MedellinNuevos['NuevosSIR']*MedellinNuevos['PesoLargo'])/(MedellinNuevos['PesoCorto']+MedellinNuevos['PesoLargo'])
MedellinNuevos['prediccion_confirmados_final']=MedellinNuevos['prediccion_nuevos_final'].cumsum()
MedellinNuevos['prediccion_confirmados_final']=MedellinNuevos['prediccion_confirmados_final'].map(lambda x: x+float(M2Confirmados.iloc[-1].values))
MedellinNuevosFinal=MedellinNuevos[['prediccion_nuevos_final','prediccion_confirmados_final']]
a=tabla2[int(tabla2[tabla2['Fecha']==MedellinNuevosFinal.index[-1]].index.values)+1:][['Fecha','CasosDia']]
a=a.set_index('Fecha')
a.rename(columns={'CasosDia':'prediccion_nuevos_final'},inplace=True)
a['prediccion_confirmados_final']=a['prediccion_nuevos_final'].cumsum()
a['prediccion_confirmados_final']=a['prediccion_confirmados_final'].map(lambda x: x+MedellinNuevosFinal['prediccion_confirmados_final'].iloc[-1])
MedellinNuevosFinal=pd.concat((MedellinNuevosFinal, a), axis = 0)
MedellinNuevosFinal=MedellinNuevosFinal[:40]
total=pd.concat((M2Nuevos,MedellinNuevosFinal[['prediccion_nuevos_final']]),axis = 0)
total.plot()
plt.axvline(x=M2Confirmados.index[-1],color='y',linestyle='--')
plt.savefig('./images/Med10.png')

total=pd.concat((M2Confirmados,MedellinNuevosFinal[['prediccion_confirmados_final']]),axis = 0)
total.plot()
plt.axvline(x=M2Confirmados.index[-1],color='y',linestyle='--')
plt.savefig('./images/Med11.png')

MedellinRecuperado['RecuperadosSIR']=tabla2['RecuperadosDia'][tabla2['Fecha'].isin(MedellinRecuperado.index)].values
MedellinRecuperado['PesoCorto']=[x/100 for x in [100,90,90,90,90,90,80,80,80,70,70,20,10,0]]
MedellinRecuperado['PesoLargo']=[x/100 for x in [0,10,10,10,10,10,20,20,20,30,30,80,90,100]]
MedellinRecuperado['prediccion_recuperados_final']=(MedellinRecuperado['prediccion_casos_recuperado']*MedellinRecuperado['PesoCorto']+MedellinRecuperado['RecuperadosSIR']*MedellinRecuperado['PesoLargo'])/(MedellinRecuperado['PesoCorto']+MedellinRecuperado['PesoLargo'])
MedellinRecuperadoFinal=MedellinRecuperado[['prediccion_recuperados_final']]
a=tabla2[int(tabla2[tabla2['Fecha']==MedellinRecuperadoFinal.index[-1]].index.values)+1:][['Fecha','RecuperadosDia']]
a=a.set_index('Fecha')
a.rename(columns={'RecuperadosDia':'prediccion_recuperados_final'},inplace=True)
MedellinRecuperadoFinal=pd.concat((MedellinRecuperadoFinal, a), axis = 0)
MedellinRecuperadoFinal=MedellinRecuperadoFinal[:40]
total=pd.concat((M2Recuperado,MedellinRecuperadoFinal[['prediccion_recuperados_final']]),axis = 0)
total.plot()
plt.axvline(x=M2Recuperado.index[-1],color='y',linestyle='--')
plt.savefig('./images/Med12.png')

MedellinMuerte['MuertesSIR']=tabla2['CasosDia'][tabla2['Fecha'].isin(MedellinMuerte.index)].values*tasa_letalidad
MedellinMuerte['PesoCorto']=[x/100 for x in [100,90,90,90,90,90,80,80,80,70,70,20,10,0]]
MedellinMuerte['PesoLargo']=[x/100 for x in [0,10,10,10,10,10,20,20,20,30,30,80,90,100]]
MedellinMuerte['prediccion_muertes_final']=(MedellinMuerte['prediccion_casos_muerte']*MedellinMuerte['PesoCorto']+MedellinMuerte['MuertesSIR']*MedellinMuerte['PesoLargo'])/(MedellinMuerte['PesoCorto']+MedellinMuerte['PesoLargo'])
MedellinMuerteFinal=MedellinMuerte[['prediccion_muertes_final']]
a=tabla2[int(tabla2[tabla2['Fecha']==MedellinMuerteFinal.index[-1]].index.values)+1:][['Fecha','CasosDia']]
a['CasosDia']=a['CasosDia']*tasa_letalidad
a=a.set_index('Fecha')
a.rename(columns={'CasosDia':'prediccion_muertes_final'},inplace=True)
MedellinMuerteFinal=pd.concat((MedellinMuerteFinal, a), axis = 0)
MedellinMuerteFinal=MedellinMuerteFinal[:40]
total=pd.concat((M2Muerte,MedellinMuerteFinal[['prediccion_muertes_final']]),axis = 0)
total.plot()
plt.axvline(x=M2Muerte.index[-1],color='y',linestyle='--')
plt.savefig('./images/Med13.png')

print('Estoy exportando resultados')

import six

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax


MedellinNuevosFinal['Fecha'] = MedellinNuevosFinal.index
MedellinNuevosFinal=MedellinNuevosFinal[['Fecha','prediccion_nuevos_final','prediccion_confirmados_final']]
MedellinNuevosFinal=MedellinNuevosFinal.round({'prediccion_nuevos_final': 0, 'prediccion_confirmados_final': 0})

MedellinRecuperadoFinal['Fecha'] = MedellinRecuperadoFinal.index
MedellinRecuperadoFinal=MedellinRecuperadoFinal[['Fecha','prediccion_recuperados_final']]
MedellinRecuperadoFinal=MedellinRecuperadoFinal.round({'prediccion_recuperados_final': 0})

MedellinMuerteFinal['Fecha'] = MedellinMuerteFinal.index
MedellinMuerteFinal=MedellinMuerteFinal[['Fecha','prediccion_muertes_final']]
MedellinMuerteFinal=MedellinMuerteFinal.round({'prediccion_muertes_final': 0})

render_mpl_table(MedellinNuevosFinal, header_columns=0, col_width=5).figure.savefig('./images/Med14.png')
render_mpl_table(MedellinRecuperadoFinal, header_columns=0, col_width=5).figure.savefig('./images/Med15.png')
render_mpl_table(MedellinMuerteFinal, header_columns=0, col_width=5).figure.savefig('./images/Med16.png')

print('Terminé')