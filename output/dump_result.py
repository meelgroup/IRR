import pandas as pd
import numpy as np
import sys

if(len(sys.argv)<8):
	print("arg: <filename> <method> <dataset> <train time> <train accuracy> <test time> <test_accuracy>")
	exit()
# print(sys.argv)

file_input=open(sys.argv[1],"a")

for column in sys.argv[2:]:
	file_input.write(column+str(","))
file_input.write("\n")

file_input.close()
# print("writing complete")