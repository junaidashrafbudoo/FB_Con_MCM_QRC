import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

def ket_0_state(N):
    # Define the |0⟩ state
    ket_0 = np.array([1, 0])

    # Initialize the state
    state = ket_0.copy()

    # Iteratively compute the tensor product
    for _ in range(N - 1):
        state = np.kron(state, ket_0)

    return state

Z = np.array([[1, 0], [0, -1]])

def get_z_operator(site, N):
    I = np.eye(2)
    Z_list = [I] * N  # Identity on all qubits
    Z_list[site] = Z   # Place Z on the desired qubit
    return np.kron(*Z_list)

class QuantumIsingChain:
    def __init__(self, N, J ,hx, hz):
        self.N = N
        self.J = J
        self.hx = hx
        self.hz = hz

    def get_Ising_Hamiltonian(self):
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
        for i in range(self.N-1):
            pauli_label = ['I'] * self.N
            pauli_label[i] = 'Z'
            pauli_label[(i + 1)] = 'Z'
            pauli_string = ''.join(pauli_label)
            pauli_strings.append(pauli_string)
            coefficients.append(-self.J)

        # Magnetic field terms
        for i in range(self.N):
            pauli_label = ['I'] * self.N
            pauli_label[i] = 'X'
            pauli_string = ''.join(pauli_label)
            pauli_strings.append(pauli_string)
            coefficients.append(self.hx)

        # Magnetic field terms
        for i in range(self.N):
            pauli_label = ['I'] * self.N
            pauli_label[i] = 'Z'
            pauli_string = ''.join(pauli_label)
            pauli_strings.append(pauli_string)
            coefficients.append(self.hz)

        # Convert coefficients to numpy array
        coefficients = np.array(coefficients, dtype=complex)

        # Create the Hamiltonian
        ising_hamiltonian = SparsePauliOp(pauli_strings, coefficients)

        return ising_hamiltonian.to_matrix()

    def get_seq_expectation_value(self, spin_site, timesteps):
        H = self.get_Ising_Hamiltonian()

        start_state = ket_0_state(self.N)

        tau = 0

        Z_operator = SparsePauliOp.from_sparse_list([("Z", [spin_site], 1.0)], num_qubits=self.N).to_matrix()
        sequence = []
        for _ in range(timesteps):
            tau += 0.05
            state = expm(-1j * H * tau) @ start_state
            sequence.append(np.real(np.vdot(state, Z_operator @ state)))

        return sequence


q1dim = QuantumIsingChain(N=5, J=1, hx=1.05, hz=-0.5)
seq = q1dim.get_seq_expectation_value(spin_site=2, timesteps=3000)

# Prepare x and y data
x_data = []
y_data = []
for i in range(len(seq) - 10):
    x_data.append(seq[i])
    y_data.append(seq[i + 10])

# Plot as a scatter plot
plt.plot(x_data, y_data)

plt.show()
