#!/bin/sh
# script to send multiple training jobs
# USAGE:
# source send_jobs_multiple.sh [PATH_TO_CONFIG] [HOW_MANY_RUNS]
conda activate quantum-tf

# Check if there are arguments
if [ -z "$1" ]
then
    echo "No PATH_TO_CONFIG argument supplied"
# Check if there are 2 arguments
elif [ -z "$2" ]
then
    echo "No HOW_MANY_RUNS argument supplied"
else
    # For loop sometimes have trouble when $2=1, therefore we run it seperately
    if [ $2 -eq 1 ]
    then
        #echo "$2"
        python3 train.py $1 1 &
    # if HOW_MANY_RUNS >1 and <6,  run!
    elif [ $2 -gt 1 ]
    then
        if [ $2 -lt 6 ]
        then
            for i in {1..$2}
            do
                echo "$i"
                python3 train.py $1 $i &
            done
        else
            echo "HOW_MANY_RUNS number is too large (>5), choose a smaller value!"
        fi
    else
        echo "Choose a suitable HOW_MANY_RUNS number! (0<HOW_MANY_RUNS<6)"
    fi 
fi

