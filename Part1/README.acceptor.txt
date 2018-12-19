To generate the positive and negative examples you need to run:
python gen_example.py LIMIT

LIMIT an optional parameter, it's the limit of the size of a sequence of number or character when you use + or * in a regex.
By default it is 10.
Ex : "a+" will generate a maximum of 10 "a" by default.

To run the acceptor:
python experiment.py