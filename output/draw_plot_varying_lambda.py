
import pandas as pd
import numpy as np
import pylab

markersize = 20
fontsize = 32
labelsize = 20
pylab.rc('pdf',fonttype = 42)
pylab.rc('ps',fonttype = 42)
marker = ['x', 'o', "v", '*', 's', "p", 'D', '^']
x = [1, 2, 3]
my_xticks = ['$1$', '$5$', '$10$']
df = pd.read_csv('vary_lambda', header=-1, sep='\s+',)
df.columns = ["index",   "lambda", "test accuracy",
              "train accuracy",    "time",  "rule size", "std test",  "sem rule size"]

show = not True

y = df["test accuracy"]
yerr = df["std test"]
pylab.errorbar(x, y, yerr, linewidth=4,
               markersize=markersize, marker=marker[1])
pylab.xlabel("fidelity, "+r"$\lambda$", fontsize=fontsize)
pylab.ylabel("Test Acc "+r"$\%$", fontsize=fontsize)
pylab.xticks(x, my_xticks)
pylab.tick_params(labelsize=labelsize)
pylab.tight_layout()
if(show):
    pylab.show()
pylab.savefig("test_accuracy_vary_lambda.pdf")
pylab.clf()

y = df["train accuracy"]
# yerr = df["std train"]
# pylab.errorbar(x, y, yerr, linewidth=4,
#                markersize=markersize, marker=marker[1])
pylab.plot(x, y, linewidth=4,
           markersize=markersize, marker=marker[1])

pylab.xlabel("fidelity, "+r"$\lambda$", fontsize=fontsize)
pylab.ylabel("Train Acc "+r"$\%$", fontsize=fontsize)
pylab.xticks(x, my_xticks)
# pylab.ylim(98,99.5)
pylab.tick_params(labelsize=labelsize)
pylab.tight_layout()
if(show):
    pylab.show()
pylab.savefig("train_accuracy_vary_lambda.pdf")
pylab.clf()

y = df["time"]
pylab.plot(x, y, linewidth=4, markersize=markersize, marker=marker[1])
pylab.xlabel("fidelity, "+r"$\lambda$", fontsize=fontsize)
pylab.ylabel("Time (s)", fontsize=fontsize)
pylab.xticks(x, my_xticks)
pylab.tick_params(labelsize=labelsize)
pylab.tight_layout()
if(show):
    pylab.show()
pylab.savefig("time_vary_lambda.pdf")
pylab.clf()


y = df["rule size"]
yerr = df["sem rule size"]
pylab.errorbar(x, y, yerr, linewidth=4, markersize=markersize, marker=marker[1])
pylab.xlabel("fidelity, "+r"$\lambda$", fontsize=fontsize)
pylab.ylabel("Rule Size", fontsize=fontsize)
pylab.xticks(x, my_xticks)
pylab.tick_params(labelsize=labelsize)
pylab.tight_layout()
if(show):
    pylab.show()
pylab.savefig("rule_size_vary_lambda.pdf")
pylab.clf()


# diff_set=["Rule Size","TR","VAL","Time(s)","TST"]
# marker=['x','o',"v",'*','s',"p",'D','^']
# rule_type=["DNF","CNF"]
# rule_count=0
# for item in range(3,6):
# 	for rule in range(2):
# 		x = df['x']
# 		cnt=0
# 		for i in (df.columns[2+item*10+rule*5:7+item*10+rule*5]):
# 			cnt+=1
# 			# if(cnt not in [5,6]):
# 			# 	continue
# 			print(cnt)
# 			if(cnt==1 or cnt==4):
# 				continue
# 			y = df[i]
# 			print(i)
# 			pylab.plot(x,y,linewidth=4,markersize=markersize,marker=marker[cnt-1],label=diff_set[cnt-1])
# 			# cnt+=1
# 		pylab.xlabel(r"$\lambda$",fontsize=fontsize)
# 		pylab.ylabel("Accuracy",fontsize=fontsize)
# 		# if(item in [1,2,4]):
# 		# 	pylab.ylim(60,100)
# 		# elif item in [1]:
# 		# pylab.ylim(60,100)
# 		pylab.xlim(0.8, 10.2)
# 		# pylab.ylim(top=100)

# 		# if(item in [1,2,4]):
# 		# 	pylab.ylim(bottom=70)
# 		# pylab.tick_params(labelsize=20)
# 		pylab.legend(loc='best',frameon=False ,fontsize=fontsize-8,ncol=1,columnspacing=0.5)
# 		# pylab.legend(frameon=False,loc='best',fontsize=fontsize-4,ncol=3,columnspacing=1)
# 		pylab.tick_params(labelsize=labelsize)
# 		pylab.tight_layout()
# 		print(("Accuracy_"+str(rule_type[rule_count/3])+"_"+str(rule_count%3+1)+"_lambda_varying.pdf"))
# 		pylab.savefig("Accuracy_"+str(rule_type[rule_count/3])+"_"+str(rule_count%3+1)+"_lambda_varying.pdf")
# 		# if(item in [0,2,4]):
# 		pylab.show()
# 		pylab.clf()
# 		rule_count+=1
