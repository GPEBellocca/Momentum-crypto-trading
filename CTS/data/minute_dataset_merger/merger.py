import pandas as pd


df1 = pd.DataFrame()
df1 = pd.read_csv("./LTCUSDm1.csv")
df2 = pd.DataFrame()
df2 = pd.read_csv("./LTCUSDm2.csv")



dfres = pd.DataFrame()
frames = [df1, df2]
dfres = pd.concat(frames, ignore_index= True)
dfres.to_csv("./LTCUSDm.csv", index=False)

df3 = pd.DataFrame()
df3 = pd.read_csv("./LTCUSDm.csv")

print(df3)


