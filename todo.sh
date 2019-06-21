#!/bin/bash


# echo "Rank is: ${OMPI_COMM_WORLD_RANK}"

ulimit -t unlimited
shopt -s nullglob
# numthreads=$((OMPI_COMM_WORLD_SIZE))
# mythread=$((OMPI_COMM_WORLD_RANK))


numthreads=1
mythread=1



tlimit="2500"
#4GB mem limit
memlimit="14000000"
echo "$numthreads"
echo "$mythread"

#methods=(other_classifier Exact_LP Approx_LP)
methods=(IMLI Approx_LP other_classifier Exact_LP )



# parkinsons 190
# pima 760
# transfusion 740
# wdbc 560
# toms 28170
# ionosphere 350 
# twitter 49990
# credit 30000
# adult 32560
# iris 150
# tictactoe 959
# titanic 1309
# compas 7210
# heart 303
# ilpd 583

datasets=(iris parkinsons pima transfusion wdbc ionosphere tictactoe titanic compas heart ilpd adult credit toms twitter)
size=(150 190 760 740 560 350 959 1309 7210 303 583 32560 30000 28170 49990)





runIndexes=10
zero=0
cnt=1
levels=(1 2 3)
solvers=(cplex)
weight_regularization=(1  5  10)
partitions=(16384 8192 4096 2048 1024 512 256 128 64 32 16 8 4 2)
ms=(1 2 3)
classifiers=(rf logreg nn svc ripper)
params=(7 9 7 9 8)





#create individual todos



for dataset_index in ${!datasets[*]}
do
    dataset=${datasets[$dataset_index]}
    echo $dataset
    for method in ${methods[*]}
    do


        if [[ $method == other_classifier ]]
        then

            for(( runIndex=0;  runIndex < $runIndexes; runIndex++ ))
            do
                it_classifier=0
                for classifier in ${classifiers[*]}
                do
                    for param in `seq 0 ${params[$it_classifier]}`
                    do
#                        echo "$param"
                        # if (($mythread == $(($cnt%$numthreads))))
                        # then
                            rm -f todo_$mythread.sh
                            touch todo_$mythread.sh
#                            echo "#!/bin/bash" >> todo_$mythread.sh
                            echo "ulimit -t 1000" >> todo_$mythread.sh
                            if [[ $method == other_classifier ]]
                            then
                                echo "no virtual limit for ripper"
                            else
                                echo "ulimit -v $memlimit" >> todo_$mythread.sh
                            fi
                            echo "ulimit -c 0" >> todo_$mythread.sh
#                            echo "set -x" >> todo_$mythread.sh
                            

                            echo "cd scripts/" >>todo_$mythread.sh
                            echo "mkdir ../Benchmarks/tempfiles_$mythread" >>todo_$mythread.sh
                            echo "/usr/bin/time --verbose -o ../Benchmarks/tempfiles_$mythread/time.txt python other_classifier.py --dataset=$dataset --classifier=$classifier --node=$mythread --runIndex=$runIndex --param=$param --timeout=$tlimit" >>todo_$mythread.sh
                            echo "python ../output/dump_time.py ../Benchmarks/tempfiles_$mythread/time.txt $classifier $dataset  NAN NAN NAN NAN  NAN $runIndex $param" >> todo_$mythread.sh
#                            echo "python ../output/dump_time.py ../Benchmarks/tempfiles_$mythread/time.txt Approx_LP $dataset $level $lamda linprog NAN $partition $runIndex $m" >> todo_$mythread.sh

                            echo "rm -r ../Benchmarks/tempfiles_$mythread" >>todo_$mythread.sh
                            echo "cd .." >>todo_$mythread.sh


                            bash todo_$mythread.sh
                            rm -f todo_$mythread.sh

#                            echo "$it_classifier"
                        # fi
                        cnt=$((cnt +1))

                    done
                    it_classifier=$((it_classifier+1))
                done
            done
        elif [[ $method == IMLI ]]
        then

            for level in ${levels[*]}
            do

                for(( runIndex=0;  runIndex < $runIndexes; runIndex++ ))
                do

                    for lamda in ${weight_regularization[*]}
                    do



                        for partition in ${partitions[*]}
                        do
                            l1=$((${size[$dataset_index]}*9 ))
                            r1=$((320*$partition))
                            r2=$((5120*$partition))

                            if [ $l1 -ge $r1 ] && [ $l1 -le $r2 ];
                            then
    #                        echo "$param"
                                # if (($mythread == $(($cnt%$numthreads))))
                                # then
                                    rm -f todo_$mythread.sh
                                    touch todo_$mythread.sh
        #                            echo "#!/bin/bash" >> todo_$mythread.sh
                                    echo "ulimit -t 1000" >> todo_$mythread.sh
                                    echo "ulimit -v $memlimit" >> todo_$mythread.sh
                                    echo "ulimit -c 0" >> todo_$mythread.sh
        #                            echo "set -x" >> todo_$mythread.sh


                                    echo "cd scripts/" >>todo_$mythread.sh
                                    echo "mkdir ../Benchmarks/tempfiles_$mythread" >>todo_$mythread.sh
                                    echo "/usr/bin/time --verbose -o ../Benchmarks/tempfiles_$mythread/time.txt python imli.py --dataset=$dataset  --lamda=$lamda --runIndex=$runIndex  --node=$mythread --partition=$partition --level=$level;" >>todo_$mythread.sh
                                    echo "python ../output/dump_time.py ../Benchmarks/tempfiles_$mythread/time.txt Approx $dataset $level $lamda open-wbo NAN $partition $runIndex 1" >> todo_$mythread.sh
                                    echo "rm -r ../Benchmarks/tempfiles_$mythread" >>todo_$mythread.sh
                                    echo "cd .." >>todo_$mythread.sh


                                    bash todo_$mythread.sh
                                    rm -f todo_$mythread.sh

        #                            echo "$it_classifier"
                                # fi
                                cnt=$((cnt +1))
                            fi
                        done
                        it_classifier=$((it_classifier+1))
                    done
                done
            done

        else


            for level in ${levels[*]}
            do

                for solver in ${solvers[*]}
                do

                    for m in ${ms[*]}
                    do


                        for(( runIndex=0;  runIndex < $runIndexes; runIndex++ ))
                        do

                            for lamda in ${weight_regularization[*]}
                            do

                                if [[ $method == Approx* ]]
                                then

                                    for partition in ${partitions[*]}
                                    do


                                            l1=$((${size[$dataset_index]}*9 ))
                                            r1=$((320*$partition))
                                            r2=$((5120*$partition))

                                            if [ $l1 -ge $r1 ] && [ $l1 -le $r2 ];
                                            then

                                                # if (($mythread == $(($cnt%$numthreads))))
                                                # then
                                                    rm -f todo_$mythread.sh
                                                    touch todo_$mythread.sh
#                                                    echo "#!/bin/bash" >> todo_$mythread.sh
                                                    echo "ulimit -t $tlimit" >> todo_$mythread.sh
                                                    echo "ulimit -v $memlimit" >> todo_$mythread.sh
                                                    echo "ulimit -c 0" >> todo_$mythread.sh
#                                                    echo "set -x" >> todo_$mythread.sh


                                                    echo "cd scripts/" >>todo_$mythread.sh
                                                    echo "mkdir ../Benchmarks/tempfiles_$mythread" >>todo_$mythread.sh
                                                    # echo "python init.py --dataset=$dataset --node=$mythread --runIndex=$runIndex  --partition=$partition" >>todo_$mythread.sh
                                                    if [[ $method == Approx ]]
                                                    then
                                                        echo "/usr/bin/time --verbose -o ../Benchmarks/tempfiles_$mythread/time.txt python partitionBasedSat.py --dataset=$dataset  --lamda=$lamda --runIndex=$runIndex --solver=$solver --node=$mythread --partition=$partition --m=$m --level=$level;" >>todo_$mythread.sh
                                                        echo "python ../output/dump_time.py ../Benchmarks/tempfiles_$mythread/time.txt Approx $dataset $level $lamda $solver NAN $partition $runIndex $m" >> todo_$mythread.sh

                                                    else
                                                        echo "/usr/bin/time --verbose -o ../Benchmarks/tempfiles_$mythread/time.txt python partitionBasedLp.py --dataset=$dataset  --lamda=$lamda --runIndex=$runIndex  --node=$mythread --partition=$partition --m=$m --level=$level --solver=$solver --timeout=$tlimit" >>todo_$mythread.sh
                                                        echo "python ../output/dump_time.py ../Benchmarks/tempfiles_$mythread/time.txt Approx_LP $dataset $level $lamda $solver NAN $partition $runIndex $m" >> todo_$mythread.sh

                                                    fi
                                                    echo "rm -r ../Benchmarks/tempfiles_$mythread" >>todo_$mythread.sh
                                                    echo "cd .." >>todo_$mythread.sh

                                                    bash todo_$mythread.sh
                                                    rm -f todo_$mythread.sh

                                                # fi
                                                cnt=$((cnt +1))
                                            fi

                                    done
                                else
                                    # if (($mythread == $(($cnt%$numthreads))))
                                    # then

                                        rm -f todo_$mythread.sh
                                        touch todo_$mythread.sh
#                                        echo "#!/bin/bash" >> todo_$mythread.sh
                                        echo "ulimit -t $tlimit" >> todo_$mythread.sh
                                        echo "ulimit -v $memlimit" >> todo_$mythread.sh
                                        echo "ulimit -c 0" >> todo_$mythread.sh
#                                        echo "set -x" >> todo_$mythread.sh


                                        echo "cd scripts/" >>todo_$mythread.sh
                                        echo "mkdir ../Benchmarks/tempfiles_$mythread" >>todo_$mythread.sh
                                        # echo "python init.py --dataset=$dataset --node=$mythread --runIndex=$runIndex" >>todo_$mythread.sh
                                        if [[ $method == Exact ]]
                                        then
                                            echo "/usr/bin/time --verbose -o ../Benchmarks/tempfiles_$mythread/time.txt python sat.py --dataset=$dataset  --lamda=$lamda --runIndex=$runIndex --solver=$solver --node=$mythread --m=$m --level=$level" >>todo_$mythread.sh
                                            echo "python ../output/dump_time.py ../Benchmarks/tempfiles_$mythread/time.txt Exact $dataset $level $lamda $solver NAN NAN $runIndex $m" >> todo_$mythread.sh

                                        else
                                            echo "/usr/bin/time --verbose -o ../Benchmarks/tempfiles_$mythread/time.txt python lp.py --dataset=$dataset  --lamda=$lamda --runIndex=$runIndex  --node=$mythread --m=$m --level=$level --solver=$solver --timeout=$tlimit" >>todo_$mythread.sh
                                            echo "python ../output/dump_time.py ../Benchmarks/tempfiles_$mythread/time.txt Exact_LP $dataset $level $lamda $solver NAN NAN $runIndex $m" >> todo_$mythread.sh

                                        fi
                                        echo "rm -r ../Benchmarks/tempfiles_$mythread" >>todo_$mythread.sh
                                        echo "cd .." >>todo_$mythread.sh


                                        bash todo_$mythread.sh
                                        rm -f todo_$mythread.sh

                                    # fi
                                    cnt=$((cnt +1))
                                fi
                            done
                        done
                    done
                done
            done
        fi
    done
done

