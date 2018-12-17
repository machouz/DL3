PART 3
======================
Training the model
---------------------

```
python bilstmTrain.py train_name repr model_file w2i_file id_label_file dev_name c2i_file
```
 
* `train_name` is the path to the training data
* `repr` is the model to train (can be `-a` `-b` `-c` `-d`)
* `model_file` is the name of the file of the trained model
* `w2i_file` is the name of the file of the word to index dictionary (for the second model it's the char to index one)
* `id_label_file` is the name of the file of the idex to label dictionary
* `dev_name` is the path to the dev data
* `c2i_file` is the name of the file of the char to index dictionary (only needed for the last model)

Using the model
---------------------
```
python bilstmTag.py repr modelFile inputFile w2i_file id_label_file dev_name c2i_file
```
 
* `train_name` is the path to the training data
* `repr` is the model to train (can be `-a` `-b` `-c` `-d`)
* `model_file` is the name of the file of the trained model
* `w2i_file` is the name of the file of the word to index dictionary (for the second model it's the char to index one)
* `id_label_file` is the name of the file of the idex to label dictionary
* `dev_name` is the path to the dev data
* `c2i_file` is the name of the file of the char to index dictionary (only needed for the last model)