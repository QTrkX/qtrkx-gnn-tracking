import sys
import os
# Turn off warnings and errors due to TF libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import time
import datetime
import csv
from random import shuffle
import tensorflow as tf
# import internal scripts
from tools.tools import *
from test import test
###############################################################################
def batch_train_step(n_step):
    '''combines multiple  graph inputs and executes a step on their mean'''
    with tf.GradientTape() as tape:
        for batch in range(config['batch_size']):
            X, Ri, Ro, y = train_data[
                train_list[n_step*config['batch_size']+batch]
                ]

            label = tf.reshape(tf.convert_to_tensor(y),shape=(y.shape[0],1))
            
            if batch==0:
                # calculate weight for each edge to avoid class imbalance
                weights = tf.convert_to_tensor(true_fake_weights(y))
                # reshape weights
                weights = tf.reshape(tf.convert_to_tensor(weights),
                                     shape=(weights.shape[0],1))
                preds = model([map2angle(X),Ri,Ro])
                labels = label
            else:
                weight = tf.convert_to_tensor(true_fake_weights(y))
                # reshape weights
                weight = tf.reshape(tf.convert_to_tensor(weight),
                                    shape=(weight.shape[0],1))

                weights = tf.concat([weights, weight],axis=0)
                preds = tf.concat([preds, model([map2angle(X),Ri,Ro])],axis=0)
                labels = tf.concat([labels, label],axis=0)

        loss_eval = loss_fn(labels, preds, sample_weight=weights)

    grads = tape.gradient(loss_eval, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    return loss_eval, grads

if __name__ == '__main__':
    # Read config file
    config = load_config(parse_args())
    tools.config = config

    # Set GPU variables
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
    USE_GPU = (config['gpu']  != '-1')

    # Set number of thread to be used
    os.environ['OMP_NUM_THREADS'] = str(config['n_thread'])  # set num workers
    tf.config.threading.set_intra_op_parallelism_threads(config['n_thread'])
    tf.config.threading.set_inter_op_parallelism_threads(config['n_thread'])

    # Load the network
    if config['network'] == 'QGNN':
        from qnetworks.QGNN import GNN
        GNN.config = config
    elif config['network'] == 'CGNN':
        from qnetworks.CGNN import GNN
        GNN.config = config
    else: 
        print('Wrong network specification!')
        sys.exit()
	
    # setup model
    model = GNN()

    # load data
    train_data = get_dataset(config['train_dir'], config['n_train'])
    train_list = [i for i in range(config['n_train'])]

    # execute the model on an example data to test things
    X, Ri, Ro, y = train_data[0]
    model([map2angle(X), Ri, Ro])

    # print model summary
    print(model.summary())

    # Log initial parameters if new run
    if config['run_type'] == 'new_run':    
        if config['log_verbosity']>=2:
            log_parameters(config['log_dir'], model.trainable_variables)
        epoch_start = 0

        # Test the validation and training set
        if config['n_valid']: test(config, model, 'valid')
        if config['n_train']: test(config, model, 'train')
    # Load old parameters if continuing run
    elif config['run_type'] == 'continue':
        # load params 
        model, epoch_start = load_params(model, config['log_dir'])
    else:
        raise ValueError('Run type not defined!')

    # Get loss function and optimizer
    loss_fn = getattr(tf.keras.losses, config['loss_func'])()
    opt = getattr(
        tf.keras.optimizers,
        config['optimizer'])(learning_rate=config['lr_c']
    )

    # Print final message before training
    if epoch_start == 0: 
        print(str(datetime.datetime.now()) + ': Training is starting!')
    else:
        print(
            str(datetime.datetime.now()) 
            + ': Training is continuing from epoch {}!'.format(epoch_start+1)
            )

    # Start training
    for epoch in range(epoch_start, config['n_epoch']):
        shuffle(train_list) # shuffle the order every epoch

        for n_step in range(config['n_train']//config['batch_size']):
            # start timer
            t0 = datetime.datetime.now()  

            # iterate a step
            loss_eval, grads = batch_train_step(n_step)
                        
            # end timer
            dt = datetime.datetime.now() - t0  
            t = dt.seconds + dt.microseconds * 1e-6 # time spent in seconds

            # Print summary
            print(
                str(datetime.datetime.now())
                + ": Epoch: %d, Batch: %d, Loss: %.4f, Elapsed: %dm%ds" \
                %(epoch+1, n_step+1, loss_eval.numpy() ,t / 60, t % 60)
                )
            
            # Start logging 
            
            # Log summary 
            with open(config['log_dir']+'summary.csv', 'a') as f:
                f.write(
                    '%d, %d, %f, %f\n' \
                    %(epoch+1, n_step+1, loss_eval.numpy(), t)
                    )

	       # Log parameters
            if config['log_verbosity']>=2:
                log_parameters(config['log_dir'], model.trainable_variables)

           # Log gradients
            if config['log_verbosity']>=2:
                log_gradients(config['log_dir'], grads)
            
            # Test every TEST_every
            if (n_step+1)%config['TEST_every']==0:
                test(config, model, 'valid')
                test(config, model, 'train')

    print(str(datetime.datetime.now()) + ': Training completed!')

