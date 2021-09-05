import pandas as pd
import win32ui

import configparser
conf = configparser.ConfigParser()
conf.read('config.ini')

class DataMerge:
    def __init__(self):
        # ID7 Treadmill
        # self.PPG_time_up = '11:09:00'
        # self.PPG_time_down = '11:12:00'
        # self.ECG_time_up = '2021-04-08 11:09:00'
        # self.ECG_time_down = '2021-04-08 11:12:00'
        self.df1 = None
        self.df2 = None
        self.df_merge = None
        self.PPG_time_up = conf.get("Data_Merge", "PPG_time_up")
        self.PPG_time_down = conf.get("Data_Merge", "PPG_time_down")
        self.ECG_time_up = conf.get("Data_Merge", "ECG_time_up")
        self.ECG_time_down = conf.get("Data_Merge", "ECG_time_down")
        # self.PPG_time_up = '11:47:00'
        # self.PPG_time_down = '11:53:00'
        # self.ECG_time_up = '2021-04-16 ' + self.PPG_time_up
        # self.ECG_time_down = '2021-04-16 ' + self.PPG_time_down


    def manage_ECG(self):
        time_up_index = self.df1[self.df1.Time == self.ECG_time_up].index.tolist()[0]
        time_down_index = self.df1[self.df1.Time == self.ECG_time_down].index.tolist()[0]
        ECG_data = self.df1[["Time", "record.heart_rate[bpm]"]].loc[time_up_index:time_down_index]
        ECG_data["Time"] = ECG_data["Time"].astype("str").map(lambda x: x.replace(x[:-8], ""))
        return ECG_data

    def manage_PPG(self):
        PPG_raw_data = self.df2[['Time', ' Green Count', ' Green2 Count', ' IR Count', ' Red Count']]
        time_up_index = self.df2[self.df2.Time == self.PPG_time_up].index.tolist()[0]
        time_down_index = self.df2[self.df2.Time == self.PPG_time_down].index.tolist()[-1]
        PPG_data = PPG_raw_data.loc[time_up_index:time_down_index]
        return PPG_data

    def getfile(self):
        def read_file(bit):
            dlg = win32ui.CreateFileDialog(1)
            # if bit == 1:
            #     dlg.SetOFNInitialDir(
            #         r'C:\Users\csdwhuang\Desktop\cheah-modified\Phase 2 Test 2 Records Data-Laser Diode Prototype\ECG')
            # elif bit == 2:
            #     dlg.SetOFNInitialDir(
            #         r'C:\Users\csdwhuang\Desktop\cheah-modified\Phase 2 Test 2 Records Data-Laser Diode Prototype\PPG')
            dlg.DoModal()
            filename = dlg.GetPathName()
            return filename

        print("Please select ECG file(.xlsx)")
        ECG_file_path = read_file(1)
        print("Selected：{}\n".format(ECG_file_path))

        print("Please select PPG file(.csv)")
        PPG_file_path = read_file(2)
        strArray1 = PPG_file_path.split("\\")
        strArray2 = (strArray1[-1]).split("_")
        output_filename = (strArray2[-2] + "_" + strArray2[-1]).replace("csv", "xlsx")
        print("Selected：{}\n".format(PPG_file_path))

        return ECG_file_path, PPG_file_path, output_filename

    def filter_data(self):
        title_list = [' Green Count', ' Green2 Count', ' IR Count', ' Red Count', 'record.heart_rate[bpm]']
        df = self.df_merge
        df = df[title_list]
        df = df.loc[(df[' Green Count'] >= 10000)]
        df = df.loc[(df[' Green2 Count'] >= 10000)]
        df = df.loc[(df[' IR Count'] >= 10000)]
        df = df.loc[(df[' Red Count'] >= 100000)]
        df = df.loc[(df['record.heart_rate[bpm]'] >= 50) & (df['record.heart_rate[bpm]'] <= 220)]
        df = df.sort_values(by='record.heart_rate[bpm]', ascending=True)
        self.df_merge = df

    def separate_data(self,output_filename):
        df1 = self.df_merge.loc[(self.df_merge['record.heart_rate[bpm]'] <= int(conf.get("Data_Merge", "HR_low")))]
        df2 = self.df_merge.loc[
            (self.df_merge['record.heart_rate[bpm]'] > int(conf.get("Data_Merge", "HR_low"))) & (self.df_merge['record.heart_rate[bpm]'] <= int(conf.get("Data_Merge", "HR_high")))]
        df3 = self.df_merge.loc[(self.df_merge['record.heart_rate[bpm]'] > int(conf.get("Data_Merge", "HR_high")))]

        df1.to_excel("dataset/training_data/data4regression/low/low_" + output_filename)
        df2.to_excel("dataset/training_data/data4regression/medium/medium_" + output_filename)
        df3.to_excel("dataset/training_data/data4regression/high/high_" + output_filename)
        print("File separated!")

    def run(self):

        ECG_path, PPG_path, output_filename = self.getfile()
        self.df1 = pd.read_excel(ECG_path)
        self.df2 = pd.read_csv(PPG_path)

        ECG_data = self.manage_ECG()
        PPG_data = self.manage_PPG()

        self.df_merge = pd.merge(left=PPG_data, right=ECG_data, left_on="Time", right_on="Time")

        self.filter_data()
        self.separate_data(output_filename)

        self.df_merge.to_excel("dataset/merged_data/" + output_filename, index=False)
        print(output_filename + " saved!")

# ID5 Bicycle
# ECG_time_up = '2021-04-15 15:52:00'
# ECG_time_down = '2021-04-15 15:58:00'
# PPG_time_up = '15:52:00'
# PPG_time_down = '15:58:00'

# ID10 Bicycle
# ECG_time_up = '2021-04-08 15:27:00'
# ECG_time_down = '2021-04-08 15:33:00'
# PPG_time_up = '15:27:00'
# PPG_time_down = '15:33:00'

# ID5T
# PPG_time_up = '16:07:40'
# PPG_time_down = '16:20:00'
# ECG_time_up = '2021-04-15 16:07:40'
# ECG_time_down = '2021-04-15 16:20:00'

# ID6T
# PPG_time_up = '14:38:00'
# PPG_time_down = '14:42:00'
# ECG_time_up = '2021-04-16 ' + PPG_time_up
# ECG_time_down = '2021-04-16 ' + PPG_time_down

# ID11T
# PPG_time_up = '15:26:00'
# PPG_time_down = '15:31:00'

# ECG_time_up = '2021-04-15 ' + PPG_time_up
# ECG_time_down = '2021-04-15 ' + PPG_time_down

# ID8T
# PPG_time_up = '12:26:00'
# PPG_time_down = '12:32:00'
# ECG_time_up = '2021-04-16 ' + PPG_time_up
# ECG_time_down = '2021-04-16 ' + PPG_time_down

# ID9T
# PPG_time_up = '11:26:37'
# PPG_time_down = '11:33:00'
# ECG_time_up = '2021-04-08 ' + PPG_time_up
# ECG_time_down = '2021-04-08 ' + PPG_time_down

# ID4T
# PPG_time_up = '11:20:00'
# PPG_time_down = '11:25:00'
# ECG_time_up = '2021-04-15 ' + PPG_time_up
# ECG_time_down = '2021-04-15 ' + PPG_time_down

# ID1T
# PPG_time_up = '11:11:13'
# PPG_time_down = '11:18:00'
# ECG_time_up = '2021-04-09 ' + PPG_time_up
# ECG_time_down = '2021-04-09 ' + PPG_time_down
