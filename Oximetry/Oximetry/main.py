from DataMerge import DataMerge
from ModelTrainer import ModelTrainer
from HeartRatePredictor import HRpredictor

import configparser
conf = configparser.ConfigParser()
conf.read('config.ini')
a =conf.get("Data_Merge", "PPG_time_up")
print(a)
if __name__ == "__main__":

    while True:
        print("Please select the function you want:")
        print("1. Prepare training data(Merge ECG and PPG)")
        print("2. Train model(Classifier or Predictor)")
        print("3. Predict heart rate with existing model")
        print("4. Exit\n")

        mode = input("Please type in the index(1, 2, 3, 4):")

        if mode == "1":
            # print("Merge data")
            DataMerge().run()
        elif mode == "2":
            print()
            print("Please select the model to be trained:")
            print("1. Classifier(Random Forest)")
            print("2. Predictor(Random Forest)")
            print("3. Predictor(Keras Neural Network)")
            mode = input("Please type in the index(1, 2, 3):")
            ModelTrainer(mode)
        elif mode == "3":
            # print("Predict HR")
            HRpredictor().run_dataset()
        elif mode == "4":
            print("Bye~")
            break
        else:
            print("Invalid input")

        print()
