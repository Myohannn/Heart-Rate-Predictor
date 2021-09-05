# Oximetry

Workshop report: https://docs.google.com/document/d/1dXQfEB2MrkhZBZW3rTBgUqv3nMeJxR4xu_ywu5ZfTGg/edit?usp=sharing

This is the source code of our heart rate predictor.
The program was coded in Python 3.9 and you can find the require package in “requirements.txt”.

There are three functions:
1.Preprocess training data
2.Build model (classifier&predictor)
3.Predict heart rate

Preprocess training data:
1.Edit ‘config.ini’: Modify the [Data_Merge] section according to the time schedule and the heart rate level bound
2.Run “main.py’
3.Type 1 in the console
4.Select the ECG and PPG file in the pop up window
5.The merged file will be saved in ‘separate_data’ and the separated files will 		be saved in 'dataset/trainingdata/data4regression’

Remark:
Make sure the time formats of ECG and PPG files and ‘config.ini’ are correct.

Build model:
1.Edit ‘config.ini’: Modify the parameters in [Model_Trainer] section.
2.Run “main.py’
3.Type 2 in the console
4.Select model type in the console
5.Select training dataset in the pop up window
6.The well trained model will be saved in ‘model/classifier’ or ‘model/predictor’

Remark: 
For keras predictor, the training results of all the parameters will be saved in 	‘dataset/training_result/model_training_result.xls’.

Predict heart rate:
1.Edit ‘config.ini’: Modify the parameters in [HR_Predictor] section.
2.Run “main.py’
3.Type 3 in the console
4.Select dataset to be predicted
5.Select models
6.The prediction output will be saved in ‘HR_prediction,xlsx’




 
