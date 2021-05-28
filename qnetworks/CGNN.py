import tensorflow as tf
import numpy as np
################################################################################################### Define Edge Network
class EdgeNet(tf.keras.layers.Layer):
    def __init__(self, name='EdgeNet', hid_dim=10):
        super(EdgeNet, self).__init__(name=name)
        
        self.layer = tf.keras.Sequential([
            tf.keras.Input(shape=(hid_dim+3)*2,),
            tf.keras.layers.Dense(hid_dim, activation='tanh'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        
    def call(self,X, Ri, Ro):
        bo = tf.matmul(Ro,X,transpose_a=True)
        bi = tf.matmul(Ri,X,transpose_a=True)

        # Shape of B = N_edges x 6 (2x (3 coordinates))
        # each row consists of two node that are possibly connected.
        B  = tf.concat([bo, bi], axis=1) # n_edges x 6, 3-> r,phi,z 

        return self.layer(B)

# Define Node Network
class NodeNet(tf.keras.layers.Layer):
    def __init__(self, name='NodeNet', hid_dim=10):
        super(NodeNet, self).__init__(name=name)

        self.layer = tf.keras.Sequential([
            tf.keras.Input(shape=(hid_dim+3)*3,),
            tf.keras.layers.Dense(hid_dim, activation='tanh'),
            tf.keras.layers.Dense(hid_dim, activation='sigmoid'),
        ])

    def call(self,X, e, Ri, Ro):
        bo  = tf.matmul(Ro, X, transpose_a=True)
        bi  = tf.matmul(Ri, X, transpose_a=True) 
        Rwo = Ro * e[:,0]
        Rwi = Ri * e[:,0]

        # changin the order to test something !!!!!!!!! DONT FORGET TO LOOK BACK!!!
        mi = tf.matmul(Rwi, bo)
        mo = tf.matmul(Rwo, bi)
        # Shape of M = N_nodes x 9 (3x (3 coordinates))
        # each row consists of a node and its 2 possible neigbours
        M = tf.concat([mi, mo, X], axis=1)
    
        return self.layer(M)
##################################################################################################
class GNN(tf.keras.Model):
    def __init__(self):
        # Network definitions here
        super(GNN, self).__init__(name='GNN')
        self.InputNet =  tf.keras.Sequential([
            tf.keras.layers.Dense(GNN.config['hid_dim'], input_shape=(3,), activation='sigmoid')
            ],name='InputNet')    
        self.EdgeNet  = EdgeNet(name='EdgeNet', hid_dim=GNN.config['hid_dim'])
        self.NodeNet  = NodeNet(name='NodeNet', hid_dim=GNN.config['hid_dim'])
        self.n_iters  = GNN.config['n_iters']
    
    def call(self, graph_array):
        X, Ri, Ro = graph_array                   # decompose the graph array
        H = self.InputNet(X)                    # execute InputNet to produce hidden dimensions
        H = tf.concat([H,X],axis=1)             # add new dimensions to original X matrix
        for i in range(self.n_iters):           # recurrent iteration of the network
            e = self.EdgeNet(H, Ri, Ro)         # execute EdgeNet
            H = self.NodeNet(H, e, Ri, Ro)      # execute NodeNet using the output of EdgeNet
            H = tf.concat([H,X],axis=1) # update H with the output of NodeNet
        e = self.EdgeNet(H, Ri, Ro)             # execute EdgeNet one more time to obtain edge predictions
        return e                                # return edge prediction array
