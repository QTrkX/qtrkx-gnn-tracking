import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import cirq
from qcircuits.QCircuit import QCircuit

###############################################################################
class Rescale01(tf.keras.layers.Layer):
    def __init__(self, name='Rescale01'):
        super(Rescale01, self).__init__(name=name)

    def call(self, X):
        X = tf.divide(
                tf.subtract(
                    X, 
                    tf.reduce_min(X)
                ), 
                tf.subtract(
                    tf.reduce_max(X), 
                    tf.reduce_min(X)
                ),
            lambda: X
        )
        return X
###############################################################################
class EdgeNet(tf.keras.layers.Layer):
    def __init__(self, name='EdgeNet'):
        super(EdgeNet, self).__init__(name=name)

        self.n_layers = GNN.config['EN_qc']['n_layers']
        self.n_qubits = GNN.config['EN_qc']['n_qubits']

        if 'dp_noise' in GNN.config['EN_qc'].keys():
            dp_noise = GNN.config['EN_qc']['dp_noise']
        else:
            # set noise to None if not specified 
            dp_noise = None
                
        # Read the Quantum Circuit with specified configuration
        qc = QCircuit(IEC_id=GNN.config['EN_qc']['IEC_id'],
            PQC_id=GNN.config['EN_qc']['PQC_id'],
            MC_id=GNN.config['EN_qc']['MC_id'],
            n_layers=self.n_layers, 
            input_size=self.n_qubits,
            p=0.01)
        
        self.model_circuit, self.qubits = qc.model_circuit()
        self.measurement_operators = qc.measurement_operators()

        # Prepare symbol list for inputs and parameters of the Quantum Circuits
        self.symbol_names = ['x{}'.format(i) for i in range(qc.n_inputs)]
        for i in range(qc.n_params):
            self.symbol_names.append('theta{}'.format(i)) 

        # Classical input layer of the Node Network
        # takes input data and feeds it to the PQC layer
        self.input_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.n_qubits, 
                activation='relu'),
            Rescale01()
        ])
        
        # Prepare PQC layer
        if (dp_noise!=None):
            # Noisy simulation requires density matrix simulator
            self.exp_layer = tfq.layers.SampledExpectation(
                cirq.DensityMatrixSimulator(noise=cirq.depolarize(dp_noise))
            )
        elif dp_noise==None and GNN.config['EN_qc']['repetitions']!=0:
            # Use default simulator for noiseless execution
            self.exp_layer = tfq.layers.SampledExpectation()
        elif dp_noise==None and GNN.config['EN_qc']['repetitions']==0:
            # Use default simulator for noiseless execution
            self.exp_layer = tfq.layers.Expectation()
        else: 
            raise ValueError('Wrong PQC Specifications!')

         # Classical readout layer
        self.readout_layer = tf.keras.layers.Dense(1, activation='sigmoid')

        # Initialize parameters of the PQC
        self.params = tf.Variable(tf.random.uniform(
            shape=(1,qc.n_params),
            minval=0, maxval=1)*2*np.pi
        ) 

    def call(self,X, Ri, Ro):
        '''forward pass of the edge network. '''

        # Constrcu the B matrix
        bo = tf.matmul(Ro,X,transpose_a=True)
        bi = tf.matmul(Ri,X,transpose_a=True)
        # Shape of B = N_edges x 6 (2x (3 + Hidden Dimension Size))
        # each row consists of two node that are connected in the input graph.
        B  = tf.concat([bo, bi], axis=1) # n_edges x 6, 3-> r,phi,z 

        # Scale the output to be [0,PI]
        # this value is a preference and can be changed 
        # to do: add the scaling as a configuration input
        input_to_circuit = self.input_layer(B) * np.pi

        # Combine input data with parameters in a single circuit_data matrix
        circuit_data = tf.concat(
            [
                input_to_circuit, 
                tf.repeat(self.params,repeats=input_to_circuit.shape[0],axis=0)
            ],
            axis=1
        )        
          
        # Get expectation values for all edges
        if GNN.config['EN_qc']['repetitions']==0:
            exps = self.exp_layer(
                self.model_circuit,
                operators=self.measurement_operators,
                symbol_names=self.symbol_names,
                symbol_values=circuit_data
            )
        else:
            exps = self.exp_layer(
                self.model_circuit,
                operators=self.measurement_operators,
                symbol_names=self.symbol_names,
                symbol_values=circuit_data,
                repetitions=GNN.config['EN_qc']['repetitions']
            )
    
        # Return the output of the final layer
        return self.readout_layer(exps)

class NodeNet(tf.keras.layers.Layer):
    def __init__(self, name='NodeNet'):
        super(NodeNet, self).__init__(name=name)
        
        self.n_layers = GNN.config['NN_qc']['n_layers']
        self.n_qubits = GNN.config['NN_qc']['n_qubits']

        if 'dp_noise' in GNN.config['EN_qc'].keys():
            dp_noise = GNN.config['EN_qc']['dp_noise']
        else:
            # set noise to None if not specified 
            dp_noise = None
        
        # Read the Quantum Circuit with specified configuration
        qc = QCircuit(
            IEC_id=GNN.config['NN_qc']['IEC_id'],
            PQC_id=GNN.config['NN_qc']['PQC_id'],
            MC_id=GNN.config['NN_qc']['MC_id'],
            n_layers=self.n_layers, 
            input_size=self.n_qubits,
            p=0.01
        )
        self.model_circuit, self.qubits = qc.model_circuit()
        self.measurement_operators = qc.measurement_operators()

        # Prepare symbol list for inputs and parameters of the Quantum Circuits
        self.symbol_names = ['x{}'.format(i) for i in range(qc.n_inputs)]
        for i in range(qc.n_params):
            self.symbol_names.append('theta{}'.format(i)) 

        # Classical input layer of the Node Network
        # takes input data and feeds it to the PQC layer
        self.input_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.n_qubits, 
                activation='relu'),
            Rescale01()
        ])

        # Prepare PQC layer
        if (dp_noise!=None):
            # Noisy simulation requires density matrix simulator
            self.exp_layer = tfq.layers.SampledExpectation(
                cirq.DensityMatrixSimulator(noise=cirq.depolarize(dp_noise))
            )
        elif dp_noise==None and  GNN.config['EN_qc']['repetitions']!=0:
            # Use default simulator for noiseless execution
            self.exp_layer = tfq.layers.SampledExpectation()
        elif dp_noise==None and  GNN.config['EN_qc']['repetitions']==0:
            # Use default simulator for noiseless execution
            self.exp_layer = tfq.layers.Expectation()
        else: 
            raise ValueError('Wrong PQC Specifications!')

        # Classical readout layer
        self.readout_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(
                GNN.config['hid_dim'], 
                activation='relu'),
            Rescale01()
        ])

        # Initialize parameters of the PQC
        self.params = tf.Variable(tf.random.uniform(
            shape=(1,qc.n_params),
            minval=0, maxval=1)*2*np.pi
        ) 

    def call(self, X, e, Ri, Ro):
        '''forward pass of the node network. '''

        # The following lines constructs the M matrix
        # M matrix contains weighted averages of input and output nodes
        # the weights are the edge probablities.
        bo  = tf.matmul(Ro, X, transpose_a=True)
        bi  = tf.matmul(Ri, X, transpose_a=True) 
        Rwo = Ro * e[:,0]
        Rwi = Ri * e[:,0]
        mi = tf.matmul(Rwi, bo)
        mo = tf.matmul(Rwo, bi)
        # Shape of M = N_nodes x (3x (3 + Hidden Dimension Size))
        # mi: weighted average of input nodes
        # mo: weighted average of output nodes
        M = tf.concat([mi, mo, X], axis=1)

        # Scale the output to be [0,PI]
        # this value is a preference and can be changed 
        # to do: add the scaling as a configuration input
        input_to_circuit = self.input_layer(M) * np.pi

        # Combine input data with parameters in a single circuit_data matrix
        circuit_data = tf.concat(
            [
                input_to_circuit, 
                tf.repeat(self.params,repeats=input_to_circuit.shape[0],axis=0)
            ],
            axis=1
        )        

        # Get expectation values for all nodes
        if GNN.config['NN_qc']['repetitions']==0:
            exps = self.exp_layer(self.model_circuit,
                operators=self.measurement_operators,
                symbol_names=self.symbol_names,
                symbol_values=circuit_data)
        else:
            exps = self.exp_layer(self.model_circuit,
                operators=self.measurement_operators,
                symbol_names=self.symbol_names,
                symbol_values=circuit_data,
                repetitions=GNN.config['NN_qc']['repetitions'])

        # Return the output of the final layer
        return self.readout_layer(exps)

###############################################################################
class GNN(tf.keras.Model):
    def __init__(self):
        ''' Init function of GNN, inits all GNN blocks. '''
        super(GNN, self).__init__(name='GNN')
        # Define Initial Input Layer
        self.InputNet =  tf.keras.layers.Dense(
            GNN.config['hid_dim'], input_shape=(3,),
            activation='relu',name='InputNet'
        )
        self.EdgeNet  = EdgeNet(name='EdgeNet')
        self.NodeNet  = NodeNet(name='NodeNet')
        self.n_iters  = GNN.config['n_iters']
    
    def call(self, graph_array):
        ''' forward pass of the GNN '''
        # decompose the graph array
        X, Ri, Ro = graph_array
        # execute InputNet to produce hidden dimensions
        H = self.InputNet(X)
        # add new dimensions to original X matrix
        H = tf.concat([H,X],axis=1)
        # recurrent iteration of the network
        for i in range(self.n_iters):
            e = self.EdgeNet(H, Ri, Ro)
            H = self.NodeNet(H, e, Ri, Ro)
            # update H with the output of NodeNet
            H = tf.concat([H,X],axis=1)
        # execute EdgeNet one more time to obtain edge predictions
        e = self.EdgeNet(H, Ri, Ro)
        # return edge prediction array
        return e
