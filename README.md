# useage steps
1. run clean_data.py to clean data
2. run original.py to get the baseline result
3. run the edge.py to get the predict result for each edge
4. run the enumeration.py to get the predicted result and the accuracy. Our method improves the baseline result by adding the edge info to it,
so before running the enumeration.py, make sure you have ran the edge.py and original.py
# clean_data.py
## input
sample.csv is a piece of sample data with the format: text\tl1,l2,l3,...
## Output
training data, testing data and tree structure of the top k frequent labels
generated folders:
```angular2
data-->processed_text-->X_test.txt
                     -->X_train.txt
                     -->y_test.txt 
                     -->y_train.txt
    -->tree_img      -->tree.png
    -->store         -->lx_ly
                     -->lx_ly
                     -->...
                     -->original
```

## How to use
put the .csv file at the same folder with get_tree.py and clean_data.py. 
Then run "python clean_data.py "sample", 5 2" where sample is the data file name, 5 is the number of top frequent labels
we need to keep, and 2 is the number of minimum labels per instance should have. The generated tree structure is stored in the "data/tree_img" folder.
There are two imgs in the tree_img folder. Each label of the img with postfix ".digit" is a int number representing the index of the lable.
# edge.py
edge.py is a binary classifier for edges. 
There are three parameter need to be filled according to the training data info printed by the clean_data.py

"lstm_num" is the length of the output vector of the LSTM, normally shorter than the "max_length" but bigger than 1.

"max_length" is the max length of the input text we keep. For this parameter I always choose the value bigger than the average length of the training text printed by the clean_data.py

"batchsize" I always set 32 if the dataset is not very big. 64 or 128 is Ok if the dataset is very big.
# original.py
original.py is a One-vs-All multi-label model. There are 4 parameters you need to change according to your data.
"lstm_num", "max_length", and "batchsize" have the same definitions with that of the edge.py
"dense_num" is the dimensionality of the output of the multi-label classifier which equals to the number of the labels.
# enumeration.py
enumeration.py does the inference. There are two parameters you need to change according to the data.
"edge_list" is a list of the edges which are listed in your "data/store" folder. If your "data/store" folder has a list of edges: l0_l1, l0_l3, l1_l4, l3_l2,
then you can fill the edge_list with [(0, 1), (0, 3), (1, 4), (3, 2)].
"total_num" is the number of the test instances. 
# Requirement Package
all list in requirements.txt

