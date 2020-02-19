import pandas as pd


Total = pd.read_csv("../Datasets/Stack/Total.csv", usecols=["Tags_clean", "Title_clean"], nrows=20000)
line = ""
for index, row in Total.iterrows():
    line += row["Title_clean"]
    line += '\t'
    test = str(row["Tags_clean"])
    if test != "nan":
        tag_list = row["Tags_clean"].split(" ")
        for i in range(len(tag_list)):
            if i != len(tag_list) - 1:
                line += tag_list[i]
                line += ","
            else:
                line += tag_list[i]
        with open("data/sample.csv", 'a+') as f:
            f.write(line + "\n")
        line = ""
print("done")
