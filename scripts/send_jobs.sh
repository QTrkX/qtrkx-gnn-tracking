conda activate quantum
#python train.py configs/test.yaml > logs/test.txt &
python train.py configs/mu10/15Z_embedding/10_hid1_it1_layer1/run1.yaml > logs/mu10/15Z_embedding/10_hid1_it1_layer1/run1/log.txt &
python train.py configs/mu10/15Z_embedding/10_hid1_it1_layer1/run2.yaml > logs/mu10/15Z_embedding/10_hid1_it1_layer1/run2/log.txt &
python train.py configs/mu10/15Z_embedding/10_hid1_it1_layer1/run3.yaml > logs/mu10/15Z_embedding/10_hid1_it1_layer1/run3/log.txt &
