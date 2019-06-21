
import pandas as pd
import numpy as np
import pylab 

markersize=20
fontsize=32
labelsize=20
pylab.rc('pdf',fonttype = 42)
pylab.rc('ps',fonttype = 42)
marker=['x','o',"v",'*','s',"p",'D','^']
x = [1,2,3]
df = pd.read_csv('vary_threshold',header=-1, sep='\s+',)
df.columns = ["index",   "lambda", "test accuracy",
              "train accuracy",    "time",  "rule size", "std test",  "sem rule size"]
show= False

y = df["test accuracy"]
yerr=df["std test"]
pylab.errorbar(x,y,yerr,linewidth=4,markersize=markersize,marker=marker[1])
pylab.xlabel("threshold, "+r"$\eta_l$",fontsize=fontsize)
pylab.ylabel("Test Acc "+r"$\%$",fontsize=fontsize)
pylab.xticks(x)
pylab.tick_params(labelsize=labelsize)
pylab.tight_layout()
if(show):
	pylab.show()
pylab.savefig("test_accuracy_vary_threshold.pdf")
pylab.clf()

y = df["train accuracy"]
# yerr=df["std train"]
# pylab.errorbar(x,y,yerr,linewidth=4,markersize=markersize,marker=marker[1])
pylab.plot(x, y, linewidth=4,
           markersize=markersize, marker=marker[1])
pylab.xlabel("threshold, "+r"$\eta_l$",fontsize=fontsize)
pylab.ylabel("Train Acc "+r"$\%$",fontsize=fontsize)
pylab.xticks(x)
pylab.tick_params(labelsize=labelsize)
pylab.tight_layout()
if(show):
	pylab.show()
pylab.savefig("train_accuracy_vary_threshold.pdf")
pylab.clf()

y = df["time"]
pylab.plot(x,y,linewidth=4,markersize=markersize,marker=marker[1])
pylab.xlabel("threshold, "+r"$\eta_l$",fontsize=fontsize)
pylab.ylabel("Time (s)",fontsize=fontsize)
pylab.xticks(x)
pylab.tick_params(labelsize=labelsize)
pylab.tight_layout()
if(show):
	pylab.show()
pylab.savefig("time_vary_threshold.pdf")
pylab.clf()


y = df["rule size"]
yerr = df["sem rule size"]
pylab.errorbar(x, y, yerr, linewidth=4, markersize=markersize, marker=marker[1])
pylab.xlabel("threshold, "+r"$\eta_l$",fontsize=fontsize)
pylab.ylabel("Rule Size",fontsize=fontsize)
pylab.xticks(x)
pylab.tick_params(labelsize=labelsize)
pylab.tight_layout()
if(show):
	pylab.show()
pylab.savefig("rule_size_vary_threshold.pdf")
pylab.clf()



