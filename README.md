# Output
training data, testing data and tree structure of the top k frequent labels
generated folders:
```angular2
data-->processed_text-->X_test.txt
                     -->X_train.txt
                     -->y_test.txt 
                     -->y_train.txt
    -->tree_img-->tree.png
```
# CSV format
sample.csv is a piece of sample data with the format: text\tl1,l2,l3,...\\
# How to use
put the .csv file at the same folder with get_tree.py and clean_data.py. 
Then run "python clean_data.py "sample", 10 2" where sample is the data file name, 10 is the number of top frequent labels
we need to keep, and 2 is the number of minimum labels per instance should have.\\
# requirement package
all list in requirements.txt\\