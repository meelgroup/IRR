## Dependencies
1. Install [Open-WBO](http://sat.inesc-id.pt/open-wbo/). After the installation is complete, add the path of the binary in the PATH variable.
  
2. Install [CPLEX](http://www-03.ibm.com/ibm/university/academic/pub/page/ban_ilog_programming) and set [python API of CPLEX](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.7.1/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html).
  
3. Python libraries: pandas, numpy, sklearn, pickle, Orange, Pulp.
 
4. Install [weka](https://www.cs.waikato.ac.nz/ml/weka/downloading.html) and add the path of weka.jar in the path variable. 

## To process data 
1. Go to `scripts/` and run     ``python data_ready.py``.

## To reproduce result
1. Run ``bash todo.sh`` from the root directory.
    
2. To view the result, go to `output/` and run ``python for_latex_table.py``. 

## View reported results in the paper
1. To view the reported results in the submitted paper, go to `output/` and run ``python for_latex_table.py``. 

## Additional materials
1. ``appendix.pdf`` contains the proof of succinctness of relaxed-CNF

2. ``results/rules.txt`` contains all the generated rules by IRR and inc-IRR.

3. ``results/figures/`` contains all the graphs of effect of individual parameters.