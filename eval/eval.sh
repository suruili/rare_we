

model=$1
model_f_base=$(basename $model)
model_type=$2
weight=$3
test_f=$4

if echo $model | grep -q "?" ; then
    echo "joined model type"
    model_1=$(echo $model | cut -d "?" -f 1)
    model_2=$(echo $model | cut -d "?" -f 2)
    model_f_base=$(basename $model_1)$(basename $model_2)
fi




for i in 2 4 6
do
nice python -u eval.py $model  $model_type $weight ./eval_data/data-chimeras/dataset_alacarte.l${i}.${test_f} &> eval_alacarte_${model_type}_${model_f_base}_${weight}_l${i}_${test_f}.log &
done

