import numpy as np
import cirq
import sympy
def measure_all(qubits, n_measurements):
    return [cirq.Z(qubits[i]) for i in range(n_measurements)]
def measure_last(qubits, n_measurements=1):
    return cirq.Z(qubits[-1])
'''
def state_vector(n_qubits):
    return qml.probs(wires=[0])# need to look in more detail
def probs(n_qubits):
    return qml.probs(wires=[i for i in range(n_qubits)])
def sample(n_qubits):
    return [qml.sample(qml.PauliZ(wires=i)) for i in range(n_qubits)]
'''
###############################################################
###############################################################
#############    INFORMATION ENCODING CIRCUITS    #############
###############################################################
###############################################################
def simple_encoding_y(circuit, qubits, n_qubits=4):
    input_  = sympy.symbols('x:{}'.format(n_qubits))
    for idx, qubit in enumerate(qubits):
        circuit.append(cirq.ry(input_[idx])(qubit))
def yz_arccos(circuit, qubits, n_qubits=4):
        input_  = sympy.symbols('x:{}'.format(n_qubits))
        for idx, qubit in enumerate(qubits):
                circuit.append(cirq.ry(input_[idx])(qubit))
                circuit.append(cirq.rz(input_[idx])(qubit))
def simple_encoding_z(circuit, qubits, n_qubits=4):
    input_  = sympy.symbols('x:{}'.format(n_qubits))
    for idx, qubit in enumerate(qubits):
        circuit.append(cirq.H(qubit))
        circuit.append(cirq.rz(input_[idx])(qubit))
def simple_encoding_x(circuit, qubits, n_qubits=4):
    input_  = sympy.symbols('x:{}'.format(n_qubits))
    for idx, qubit in enumerate(qubits):
        circuit.append(cirq.rx(input_[idx])(qubit))
def ZZFeatureMap(circuit, qubits, n_qubits=4):
    input_  = sympy.symbols('x:{}'.format(n_qubits))
    for idx, qubit in enumerate(qubits):
        circuit.append(cirq.H(qubit))
    for _ in range(2):
        for idx, qubit in enumerate(qubits):
            U1 = cirq.ZPowGate(exponent=2*input_[idx])
            circuit.append(U1(qubit))
        for idx in range(n_qubits-1):
            for idy in range(idx+1, n_qubits):
                circuit.append(cirq.CNOT(qubits[idx], qubits[idy]))
                v = 2*(np.pi - input_[idx])*(np.pi - input_[idy])
                U1 = cirq.ZPowGate(exponent=v)
                circuit.append(U1(qubits[idy]))
                circuit.append(cirq.CNOT(qubits[idx], qubits[idy]))
###############################################################
###############################################################
#############    PARAMETRIZED QUANTUM CIRCUITS    #############
###############################################################
###############################################################
def qc10_pqc(circuit, qubits, n_layers=1, n_qubits=4):
    params  = sympy.symbols('theta:{}'.format(n_qubits*(1+n_layers)))
    for i, qubit in enumerate(qubits):
        #symbol = sympy.Symbol('theta_{}'.format(i+1))
        circuit.append(cirq.ry(params[i])(qubit))
    for layer in range(n_layers):
        for i in range(n_qubits):
            circuit.append(cirq.CZ(qubits[(n_qubits-2-i)%n_qubits], qubits[(n_qubits-1-i)%n_qubits]))
        for i, qubit in enumerate(qubits):
            #symbol = sympy.Symbol('theta_{}'.format(i+1+n_qubits*(layer+1)))
            circuit.append(cirq.ry(params[i+n_qubits*(layer+1)])(qubit))
###############################################################################
def generic_pqc(circuit, qubits, n_layers=1, n_qubits=4):
    params  = sympy.symbols('theta:{}'.format(6*n_qubits*n_layers))
    for layer in range(n_layers):
        NN_entangler(circuit, qubits)
        for i, qubit in enumerate(qubits):
            n_gate = i+(2*layer)*n_qubits
            n_param = n_gate*3
            U3(params[n_param+0], params[n_param+1], params[n_param+2], circuit, qubit)
        NN2_entangler(circuit, qubits)
        for i, qubit in enumerate(qubits):
            n_gate = i+(2*layer+1)*n_qubits
            n_param = n_gate*3
            U3(params[n_param+0], params[n_param+1], params[n_param+2], circuit, qubit)

###############################################################################
def TTN(circuit, qubits, n_layers=None, n_qubits=4):
    # n_qubits must be a multiple of 4
    assert (n_qubits%4)==0
    n_layers = int(np.log2(n_qubits))
    param_count = 0
    n_params = int(2**(np.log2(n_qubits)+1)-2 +1) # +1 is for the final gate
    params  = sympy.symbols('theta:{}'.format(n_params))
    for layer in range(n_layers):
        n_gates = n_qubits//(2**(layer+1))
        for idx in range(n_gates):
            qubit0 = idx * (n_qubits//(2**(n_layers-layer-1))) + 2**layer - 1
            qubit1 = idx * (n_qubits//(2**(n_layers-layer-1))) + 2**(layer+1) - 1
            two_qubit_ry(params[param_count], params[param_count+1], circuit, qubits[qubit0], qubits[qubit1])
            param_count += 2
    circuit.append(cirq.ry(params[param_count])(qubits[n_qubits-1]))
###############################################################################
def MPS(circuit, qubits, n_layers=None, n_qubits=4):
    # n_qubits must be a multiple of 4
    assert (n_qubits%4)==0
    n_layers = int(n_qubits-1)
    param_count = 0
    n_params = 2*(n_layers) +1 # +1 is for the final gate
    params = sympy.symbols('theta:{}'.format(n_params))
    for layer in range(n_layers):
        two_qubit_ry(params[param_count], params[param_count+1], circuit, qubits[layer], qubits[layer+1])
        param_count += 2
    circuit.append(cirq.ry(params[param_count])(qubits[n_qubits-1]))
###############################################################################
def qc10_pqc_local(circuit, qubits, n_layers=1, n_qubits=4):
    params  = sympy.symbols('theta:{}'.format(n_qubits*(1+n_layers)))
    for i, qubit in enumerate(qubits):
        #symbol = sympy.Symbol('theta_{}'.format(i+1))
        circuit.append(cirq.ry(params[i])(qubit))
    for layer in range(n_layers):
        for i in range(n_qubits):
            circuit.append(cirq.CZ(qubits[(n_qubits-2-i)%n_qubits], qubits[(n_qubits-1-i)%n_qubits]))
        if layer!=(n_layers-1):
            for i, qubit in enumerate(qubits):
                circuit.append(cirq.ry(params[i+n_qubits*(layer+1)])(qubit))
        else:
            circuit.append(cirq.ry(params[n_qubits*(layer+1)])(qubit))
###############################################################################
def qc19_pqc(circuit, qubits, n_layers=1, n_qubits=4):
    params  = sympy.symbols('theta:{}'.format(3*n_qubits*n_layers))
    param_count = 0
    for layer in range(n_layers):
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.rx(params[param_count])(qubit))
            param_count += 1
            circuit.append(cirq.rz(params[param_count])(qubit))
            param_count += 1
        for i in range(n_qubits):
            # controlled Rx operations are applied
            # implementation is explained here: https://stackoverflow.com/questions/61852590/how-do-i-implement-a-controlled-rx-in-cirq-tensorflow-quantum
            circuit.append(cirq.CNOT(qubits[(n_qubits-1-i)%n_qubits], qubits[(n_qubits-i)%n_qubits])**(params[param_count]/np.pi))
            param_count += 1
###############################################################################
def qc10P_pqc(circuit, qubits, n_layers=1, n_qubits=4):
    n_params = 2*n_qubits*(1+n_layers)
    params  = sympy.symbols('theta:{}'.format(n_params))
    for i, qubit in enumerate(qubits):
        #symbol = sympy.Symbol('theta_{}'.format(i+1))
        circuit.append(cirq.ry(params[2*i])(qubit))
        circuit.append(cirq.rx(params[2*i+1])(qubit))
    for layer in range(n_layers):
        for i in range(n_qubits):
            circuit.append(cirq.CZ(qubits[(n_qubits-2-i)%n_qubits], qubits[(n_qubits-1-i)%n_qubits]))
        for i, qubit in enumerate(qubits):
            #symbol = sympy.Symbol('theta_{}'.format(i+1+n_qubits*(layer+1)))
            circuit.append(cirq.ry(params[2*i+2*n_qubits*(layer+1)])(qubit))
            circuit.append(cirq.rx(params[2*i+1+2*n_qubits*(layer+1)])(qubit))
###############################################################################
def qc6_pqc(circuit, qubits, n_layers=1, n_qubits=4):
    params = sympy.symbols('theta:{}'.format((n_qubits**2 + 3*n_qubits)*n_layers))
    param_count = 0
    for layer in range(n_layers):
        # First Rx-Rz layer
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.rx(params[param_count])(qubit))
            param_count += 1
            circuit.append(cirq.rz(params[param_count])(qubit))
            param_count += 1
        # Apply set of Rx gate (the size of this set is n_qubits)
        for idx in range(n_qubits):
            # Apply Controlled Rx gates (n_qubits-1 times)
            for idy in range(n_qubits-1):
                circuit.append(cirq.CNOT(
                    qubits[(n_qubits-1-idx)%n_qubits],     # Control qubit
                    qubits[(n_qubits-2-idx-idy)%n_qubits]  # Target qubit
                    )**(params[param_count]/np.pi))
                param_count += 1
        # Final Rx-Rz layer
        for i, qubit in enumerate(qubits):
            circuit.append(cirq.rx(params[param_count])(qubit))
            param_count += 1
            circuit.append(cirq.rz(params[param_count])(qubit))
            param_count += 1
###############################################################################
############################## Helper Functions ###############################
###############################################################################
def U3(param0, param1, param2, circuit, qubit):
    circuit.append(cirq.rz(param2)(qubit))
    circuit.append(cirq.rx(np.pi/2)(qubit))
    circuit.append(cirq.rz(param0)(qubit))
    circuit.append(cirq.rx(-np.pi/2)(qubit))
    circuit.append(cirq.rz(param1)(qubit))
###############################################################################
def NN_entangler(circuit, qubits):
        n_qubits = len(qubits)
        for i in range(n_qubits):
                circuit.append(cirq.CNOT(qubits[i%n_qubits], qubits[(i+1)%n_qubits]))
###############################################################################
def NN2_entangler(circuit, qubits):
        n_qubits = len(qubits)
        for i in range(n_qubits):
                circuit.append(cirq.CNOT(qubits[i%n_qubits], qubits[(i+2)%n_qubits]))
###############################################################################
def two_qubit_ry(param0, param1, circuit, qubit0, qubit1):
        circuit.append(cirq.ry(param0)(qubit0))
        circuit.append(cirq.ry(param1)(qubit1))
        circuit.append(cirq.CZ(qubit0, qubit1))
###############################################################################
'''
def simple_encoding_y(n_qubits=4):
    input_  = sympy.symbols('x:{}'.format(n_qubits))
    qubits = cirq.GridQubit.rect(input_.shape[0], 1)
    circuit = cirq.Circuit()
    for idx, qubit in enumerate(qubits):
        circuit.append(cirq.ry(input_[idx])(qubit))
    return circuit

def simple_encoding_z(input_):
    qubits = cirq.GridQubit.rect(input_.shape[0], 1)
    circuit = cirq.Circuit()
    for idx, qubit in enumerate(qubits):
        circuit.append(cirq.H(qubit))
        circuit.append(cirq.rz(input_[idx])(qubit))
    return circuit

def ZZ_feature_map(input_, n_iter=2):
    n_qubits = input_.shape[0]
    qubits = cirq.GridQubit.rect(input_.shape[0], 1)
    circuit = cirq.Circuit()
    for idx, qubit in enumerate(qubits):
        circuit.append(cirq.H(qubit))

    for _ in range(n_iter):
        for idx, qubit in enumerate(qubits):
            circuit.append(cirq.rz(input_[idx])(qubit))
        for idx in range(n_qubits-1):
            for idy in range(idx+1,n_qubits):
                circuit.append(cirq.CX(qubits[i], qubits[j]))
                circuit.append(cirq.rz((np.pi-inputs_[idx])*(np.pi-inputs_[idy]))(qubits[idy]))
                circuit.append(cirq.CX(qubits[i], qubits[j]))

    return circuit

def pqc_10(n_layers=1, n_qubits=4):
    qubits  = cirq.GridQubit.rect(n_qubits, 1)
    circuit = cirq.Circuit()
    input_  = sympy.symbols('x:{}'.format(n_qubits))
    params  = sympy.symbols('theta:{}'.format(n_qubits*(1+n_layers)))
    for idx, qubit in enumerate(qubits):
        circuit.append(cirq.ry(input_[idx])(qubit))
    for i, qubit in enumerate(qubits):
        #symbol = sympy.Symbol('theta_{}'.format(i+1))
        circuit.append(cirq.ry(params[i])(qubit))
    for layer in range(n_layers):
        for i in range(n_qubits):
            circuit.append(cirq.CZ(qubits[(n_qubits-2-i)%n_qubits], qubits[(n_qubits-1-i)%n_qubits]))
        for i, qubit in enumerate(qubits):
            #symbol = sympy.Symbol('theta_{}'.format(i+1+n_qubits*(layer+1)))
            circuit.append(cirq.ry(params[i+n_qubits*(layer+1)])(qubit))
    return circuit, [cirq.Z(qubits[i]) for i in range(n_qubits)]


def pqc(n_layers=1, n_qubits=4):
    qubits  = cirq.GridQubit.rect(n_qubits, 1)
    circuit = cirq.Circuit()
    input_  = sympy.symbols('x:{}'.format(n_qubits))
    params  = sympy.symbols('theta:{}'.format(n_qubits*(1+n_layers)))
    for idx, qubit in enumerate(qubits):
        circuit.append(cirq.ry(input_[idx])(qubit))
    for i, qubit in enumerate(qubits):
        #symbol = sympy.Symbol('theta_{}'.format(i+1))
        circuit.append(cirq.ry(params[i])(qubit))
    for layer in range(n_layers):
        for i in range(n_qubits):
            circuit.append(cirq.CZ(qubits[(n_qubits-2-i)%n_qubits], qubits[(n_qubits-1-i)%n_qubits]))
        for i, qubit in enumerate(qubits):
            #symbol = sympy.Symbol('theta_{}'.format(i+1+n_qubits*(layer+1)))
            circuit.append(cirq.ry(params[i+n_qubits*(layer+1)])(qubit))
    return circuit, qubits
'''
