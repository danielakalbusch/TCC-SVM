from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import pandas as pd


class SupportVectorMachines:

    def assortment(self, data_frame):
        data_base = data_frame
        data = data_base.drop('0', axis=1)
        target = data_base["0"]

        data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.30, shuffle=True)

        print('Kernel = Linear')
        classifier = SVC(kernel='linear')

        print('Treinando pandas_to_predict')
        classifier.fit(data_train, target_train)

        array_to_predict = []

        print("Predict to train")
        print(data_train.shape)

        for i in range(data_test.shape[0]):
            result = classifier.predict(data_test.iloc[[i]])
            print(str(i) + "  " + str(result))
            array_to_predict.append(result)

        pandas_to_predict = pd.DataFrame(array_to_predict)
        print("Score: " + str(classifier.score(data_test, target_test)))
        print("Score: " + str(classifier.score(data_train, target_train)))
        data_base["result_SVM"] = pandas_to_predict

        data_base.rename(
            columns={0: 'target', 1: 'exp_motorista', 2: 'idade', 3: 'ano_car', 4: 'estado_civil', 5: 'sexo',
                     6: 'trabalho_car', 7: 'indc_roubo_furto_reg', 8: 'qualidade_vida_reg',
                     9: 'indc_roubo_modelo_car', 10: 'franquia_car', 11: 'infracoes', 12: 'garagem_car',
                     13: "K-classes"}, inplace=True)

        data_base.to_csv("result_to_SVM.csv")

        print("Finishing to svm")


df = pd.read_csv("train_normalized.csv")

SupportVectorMachines = SupportVectorMachines()
SupportVectorMachines.assortment(df)
