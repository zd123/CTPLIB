from CTP.LinearRegression import CTP_LinReg
import pandas as pd

df = pd.read_csv('data/baseball.csv')

iv = ['OBP', 'SLG']
dv = 'RS'

model = CTP_LinReg.CTP_LinReg(df, iv, dv)
model.run_all()