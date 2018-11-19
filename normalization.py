import pandas as pd

from sklearn.preprocessing import MinMaxScaler


class Normalization:

 #   def normalize(self, dataFrame):
     #defne ecala - 0 e 1   scale = MinMaxScaler()
     #nao mexer dataf original, executa escala 0 e 1 (fit) com a escala   originDataframe = scale.fit_transform(df)

   #cria DF para exportar pra csv     exporting = pd.DataFrame.from_records(originDataframe)

        exporting.to_csv("train_normalized.csv")
        print("train_normalized.csv")

df = pd.read_csv("train_clustered.csv")

Normalization = Normalization()
df_normalized = Normalization.normalize(dataFrame=df)
