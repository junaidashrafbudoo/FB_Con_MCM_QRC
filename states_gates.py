import numpy as np
from qiskit.quantum_info import SparsePauliOp

def RX(theta):
    """Returns the 2x2 RX rotation matrix."""
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]])

def RY(theta):
    """Returns the 2x2 RY rotation matrix."""
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]])
def RZ(theta):
    """Returns the 2x2 RZ rotation matrix."""
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]])

def CNOT():
    return np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]])

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


def rho_0_state(N):
    return np.outer(ket_0_state(N), ket_0_state(N).conj())

def get_local_Z_observables(N):
    pauli_terms = []
    for i in range(N):
        pauli_temp = ['I'] * N
        pauli_temp[i] = 'Z'
        pauli_terms.append(SparsePauliOp(''.join(pauli_temp)).to_matrix())
    pauli_terms.reverse()
    return pauli_terms
