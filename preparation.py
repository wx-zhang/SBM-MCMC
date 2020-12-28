import joblib

truthfb = []
data_file = open('truthfb.csv')
data_file.readline()
for each_line in data_file:
    each_line = each_line.strip("\n")
    truthfb.append(int(each_line))
data_file.close()

joblib.dump(truthfb, 'truthfb.pkl')



