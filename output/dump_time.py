import pandas as pd
import numpy as np
import sys

# if(len(sys.argv)<8):
# 	print("arg: <filename> <method> <dataset> <m>
# 	exit()
# print(sys.argv)

file_input=open(sys.argv[1],"r")
line=file_input.readline()
time=0
while(line):
    if(line.startswith("\tUser time (seconds): ")):
        # print(line[:-1].split(" "))
        time+=float(line[:-1].split(" ")[-1])
    if(line.startswith("\tSystem time (seconds): ")):
        time+=float(line[:-1].split(" ")[-1])

    line=file_input.readline()
# print(time)
# # time=float(time)



file_input.close()

fout=open("../output/time.csv","a")
for item in (sys.argv[2:]):
    fout.write(item+",")
# fout.write(str(datasets[int(sys.argv[3])])+",")
# fout.write(sys.argv[4]+",")
fout.write(str(time)+"\n")

fout.close()