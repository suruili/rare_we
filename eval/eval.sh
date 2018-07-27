

model=$1
model_f_base=$(basename $model)
model_type=$2
weight=$3
test_f=$4

for i in 2 4 6
do
nice python -u eval.py $model  $model_type $weight 0 ./eval_data/data-chimeras/dataset.l${i}.${test_f} &> eval_${model_type}_${model_f_base}_${weight}_0_l${i}_${test_f}.log &
done

