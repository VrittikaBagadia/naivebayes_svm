if [ $1 -eq 1 ]
then
	python3 naivebayes.py $2 $3 $4
elif [ $1 -eq 2 ]
then
	python3 svm_all.py $2 $3 $4 $5
fi