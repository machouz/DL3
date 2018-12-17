* To train the model, you need to run the bellow command:

`python bilstmTrain.py repr trainFile modelFile w2i_file id2label_file`

repr : [1, 2, 3, 4] representation method
modelFile: path to save the model
w2i_file: path to save the w2i of the model
id2label_file: path to save the id2label of the model

* To use a trained model and predict on test file, you need to run the bellow command:

`python bilstmTag.py repr modelFile inputFile w2i_file id2label_file output_file

inputFile: path of blind test
output_file: path of the output result