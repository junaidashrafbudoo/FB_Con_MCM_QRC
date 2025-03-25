import numpy as np
import math
from qiskit.quantum_info import SparsePauliOp
def generate_binary_combinations(bit_length: int) -> list[str]:
    '''
    Creates a list of sorted binary strings, given a bit-length.

       generate_binary_combinations(3)
    >>['000', '001', '010', '011', '100', '101', '110', '111']


    :param bit_length: How many bits should be in the list
    :return: Lit of sorted binary strings
    '''
    combinations = []
    for i in range(2 ** bit_length):
        binary_str = bin(i)[2:].zfill(bit_length)
        combinations.append(binary_str)
    return combinations


def create_dictionary(bit_length: int) -> dict[str:int]:
    '''
    Creates base dictionary for measuring counts given an amount of qubits.

      create_dictionary(3)
    >>{'000': 0, '001': 0, '010': 0, '011': 0, '100': 0, '101': 0, '110': 0, '111': 0}

    :param bit_length: How many bits should be in the list
    :return: Sorted dictionary with binary strings and 0 integers.
    '''
    combinations = generate_binary_combinations(bit_length)
    sorted_combinations = sorted(combinations)  # Sort the combinations
    dictionary = {combo: 0 for combo in sorted_combinations}
    return dictionary


def get_seperate_measurements(dictionary: dict[str:int]) -> list[dict]:
    '''
    Given a dictionary of multiple measurements results (which are separated by a whitespace) creates
    a list of dictionaries containing the separate/isolated results.

      get_seperate_measurements({'00 00 11': 239, '10 00 11': 244, '01 00 11': 284, '11 00 11': 257})
    >>[{'00': 0, '01': 0, '10': 0, '11': 1024}, {'00': 1024, '01': 0, '10': 0, '11': 0}, {'00': 239, '01': 284, '10': 244, '11': 257}]


    :param dictionary: Dictionary of multiple classical registers according to qiskit 1.0
    :return: list of separate dictionaries
    '''
    bit_lenght = len(list(dictionary.keys())[0].split()[0])
    # Initialize dictionaries to store aggregated values
    part_dicts = [create_dictionary(bit_lenght) for _ in range(len(list(dictionary.keys())[0].split()))]

    # Iterate over the original dictionary
    for key, value in dictionary.items():
        parts = key.split()  # Split the key into parts

        # Aggregate values based on the parts
        for i, part in enumerate(parts):
            part_dicts[i][part] = part_dicts[i].get(part, 0) + value

    return part_dicts[::-1]


def get_expectation_values(meas: dict[str:int]) -> list[float]:
    '''
    Given a dictionaries of measurement counts, it returns the n expectation values
    according to the observables: I x ... x Z_n x I x ... x I, where n is position of the nth qubit.

    get_expectation_values({'00': 1367367, '01': 98270, '10': 7963973, '11': 570390})
    >>[-0.7068726, 0.8662679999999999]


    :param meas: dictionary of measurement counts
    :return: list of expectation values
    '''
    list_of_expV = []
    n_shots = sum(list(meas.values()))
    l_bitstring = len(list(meas.keys())[0])
    for i in range(l_bitstring):
        B_i_0 = 0
        B_i_1 = 0
        for key, value in meas.items():
            if key[i] == '0':
                B_i_0 += value
            elif key[i] == '1':
                B_i_1 += value
        list_of_expV.append((1/n_shots)*(B_i_0-B_i_1))
    return list_of_expV

def get_state_from_number(n: int, num_qubits: int) -> np.array(int):
    state = np.zeros(2**num_qubits)
    state[n] = 1
    return state


"""from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator, EstimatorOptions
from qiskit_aer import AerSimulator"""

"""options = EstimatorOptions(default_shots=1)

aer_sim = AerSimulator(seed_simulator=1)"""


def get_chopped_possonian_sample(gamma, t_max):
    denominator = 1 - math.exp(-gamma * t_max)
    if denominator == 0:
        denominator = 1 - math.exp(-(gamma+0.1) * t_max)
    c = 1 / denominator
    u = np.random.uniform(0, 1)
    return -1 * (math.log(1 - (u / c)) / gamma)


def get_chopped_possonian_samples(num_samples, gamma, t_max):
    samples = []
    c = 1 / (1 - math.exp(-gamma * t_max))
    while len(samples) < num_samples:
        u = np.random.uniform(0, 1)
        t = -1 * (math.log(1 - (u / c)) / gamma)
        # This will actually never occur, only useful for cases when t_max approaches 0 or infinity
        if t <= t_max:
            samples.append(t)
    return samples


def get_observables_m2(Pauli: str, N: int):
    string_list = []
    for n in range(N):
        for m in range(N):
            string = ["I"] * N
            if n != m:
                string[n] = Pauli
                string[m] = Pauli
            string_list.append(''.join(string))
    observables = SparsePauliOp.from_list([(pauli, 1 / N ** 2) for pauli in string_list])
    return observables


def get_observables_m(Pauli: str, N: int):
    string_list = []
    for n in range(N):
        string = ["I"] * N
        string[n] = Pauli
        string_list.append(''.join(string))
    observables = SparsePauliOp.from_list([(pauli, 1/N) for pauli in string_list])
    return observables


"""def run_circuit_on_backend(backend, observables, circuit):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
    isa_circuit = pm.run(circuit)
    isa_observables = [operator.apply_layout(isa_circuit.layout) for operator in observables]

    estimator = Estimator(mode=aer_sim)
    job = estimator.run([(isa_circuit, isa_observables)])
    pub_result = job.result()[0]
    return pub_result


def run_circuit_on_backend_single_operator(backend, observable, circuit):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(circuit)
    isa_observable = Operator(observable).apply_layout(isa_circuit.layout)

    estimator = Estimator(mode=aer_sim)
    job = estimator.run([(isa_circuit, isa_observable)])
    pub_result = job.result()[0]
    return pub_result


def run_circuit_on_backend_single_shot(backend, observables, circuit):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
    isa_circuit = pm.run(circuit)
    isa_observables = [operator.apply_layout(isa_circuit.layout) for operator in observables]

    estimator = Estimator(mode=aer_sim, options=options)
    job = estimator.run([(isa_circuit, isa_observables)])
    pub_result = job.result()[0]
    return pub_result"""


def ket_plus_state(N):
    # Define the plus state for a single qubit
    plus = np.array([1, 1]) / np.sqrt(2)

    # Initialize the state
    state = plus.copy()

    # Iteratively compute the tensor product
    for _ in range(N - 1):
        state = np.kron(state, plus)

    return state


def ket_minus_state(N):
    # Define the plus state for a single qubit
    minus = np.array([1, -1]) / np.sqrt(2)

    # Initialize the state
    state = minus.copy()

    # Iteratively compute the tensor product
    for _ in range(N - 1):
        state = np.kron(state, minus)

    return state


def ket_0_state(N):
    # Define the |0⟩ state
    ket_0 = np.array([1, 0])

    # Initialize the state
    state = ket_0.copy()

    # Iteratively compute the tensor product
    for _ in range(N - 1):
        state = np.kron(state, ket_0)

    return state


def ket_1_state(N):
    # Define the |1⟩ state
    ket_1 = np.array([0, 1])

    # Initialize the state
    state = ket_1.copy()

    # Iteratively compute the tensor product
    for _ in range(N - 1):
        state = np.kron(state, ket_1)

    return state

def get_sublist_after_last_r(data):
    try:
        # Find the last occurrence of 'r'
        last_r_index = len(data) - 1 - data[::-1].index('r')
        # Return the sublist from the last 'r' to the end
        return data[last_r_index + 1:]
    except ValueError:
        # If 'r' is not found in the list, return the original list
        return data

def get_X_Z_Hamiltonian(N, O, D):
    """
    :param N: Number of spins
    :param O: Longitudinal field strength
    :param D: transverse field strenght
    :return: Numpy matrix of the Hamiltonian
    """
    # Initialize lists
    pauli_strings = []
    coefficients = []

    # Longitudinal field terms
    for i in range(N):
        pauli_label = ['I'] * N
        pauli_label[i] = 'X'
        pauli_string = ''.join(pauli_label)
        pauli_strings.append(pauli_string)
        coefficients.append(-O)

    # Transverse field terms
    for i in range(N):
        pauli_label = ['I'] * N
        pauli_label[i] = 'Z'
        pauli_string = ''.join(pauli_label)
        pauli_strings.append(pauli_string)
        coefficients.append(-D)

    # Convert coefficients to numpy array
    coefficients = np.array(coefficients, dtype=complex)

    # Create the Hamiltonian
    ising_hamiltonian = SparsePauliOp(pauli_strings, coefficients)

    return ising_hamiltonian.to_matrix()

def get_ZZ_X_Hamiltonian(N, J, h):
    """
    :param N: Number of spins
    :param J: Coupling strength
    :param h: External field
    :return: Numpy matrix of the Hamiltonian
    """
    # Initialize lists
    pauli_strings = []
    coefficients = []

    # Interaction terms
    for i in range(N):
        pauli_label = ['I'] * N
        pauli_label[i] = 'Z'
        pauli_label[(i + 1) % N] = 'Z'
        pauli_string = ''.join(pauli_label)
        pauli_strings.append(pauli_string)
        coefficients.append(-J)

    # Magnetic field terms
    for i in range(N):
        pauli_label = ['I'] * N
        pauli_label[i] = 'X'
        pauli_string = ''.join(pauli_label)
        pauli_strings.append(pauli_string)
        coefficients.append(-J * h)

    # Convert coefficients to numpy array
    coefficients = np.array(coefficients, dtype=complex)

    # Create the Hamiltonian
    ising_hamiltonian = SparsePauliOp(pauli_strings, coefficients)

    return ising_hamiltonian.to_matrix()

def get_XX_Z_Hamiltonian(N, J, h):
    """
    :param N: Number of spins
    :param J: Coupling strength
    :param h: External field
    :return: Numpy matrix of the Hamiltonian
    """
    # Initialize lists
    pauli_strings = []
    coefficients = []

    # Interaction terms
    for i in range(N):
        pauli_label = ['I'] * N
        pauli_label[i] = 'X'
        pauli_label[(i + 1) % N] = 'X'
        pauli_string = ''.join(pauli_label)
        pauli_strings.append(pauli_string)
        coefficients.append(-J)

    # Magnetic field terms
    for i in range(N):
        pauli_label = ['I'] * N
        pauli_label[i] = 'Z'
        pauli_string = ''.join(pauli_label)
        pauli_strings.append(pauli_string)
        coefficients.append(-J * h)

    # Convert coefficients to numpy array
    coefficients = np.array(coefficients, dtype=complex)

    # Create the Hamiltonian
    ising_hamiltonian = SparsePauliOp(pauli_strings, coefficients)

    return ising_hamiltonian.to_matrix()

def get_ZZ_Hamiltonian(N, J, h):
    """
        :param N: Number of spins
        :param J: Coupling strength
        :param h: External field
        :return: Numpy matrix of the Hamiltonian
        """
    # Initialize lists
    pauli_strings = []
    coefficients = []

    # Interaction terms
    for i in range(N-1):
        pauli_label = ['I'] * N
        pauli_label[i] = 'Z'
        pauli_label[i+1] = 'Z'
        pauli_string = ''.join(pauli_label)
        pauli_strings.append(pauli_string)
        coefficients.append(-J)

    # Convert coefficients to numpy array
    coefficients = np.array(coefficients, dtype=complex)

    # Create the Hamiltonian
    ising_hamiltonian = SparsePauliOp(pauli_strings, coefficients)

    return ising_hamiltonian.to_matrix()


def get_X_Hamiltonian(N, J, h):
    """
    :param N: Number of spins
    :param J: Coupling strength
    :param h: External field
    :return: Numpy matrix of the Hamiltonian
    """
    # Initialize lists
    pauli_strings = []
    coefficients = []
    # Magnetic field terms
    for i in range(N):
        pauli_label = ['I'] * N
        pauli_label[i] = 'X'
        pauli_string = ''.join(pauli_label)
        pauli_strings.append(pauli_string)
        coefficients.append(-J * h)

    # Convert coefficients to numpy array
    coefficients = np.array(coefficients, dtype=complex)

    # Create the Hamiltonian
    ising_hamiltonian = SparsePauliOp(pauli_strings, coefficients)

    return ising_hamiltonian.to_matrix()





