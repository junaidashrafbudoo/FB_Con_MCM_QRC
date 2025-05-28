from qiskit.quantum_info import Operator, partial_trace, DensityMatrix
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import states_gates
import utils
import mackey_glass
from states_gates import RZ, RX, CNOT, rho_0_state
import functools as ft
import multiprocessing
from scipy.stats import unitary_group
from itertools import product
from ising_chain import QuantumIsingChain
import pickle
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.quantum_info import random_unitary
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.qasm3 import dump

''' Insert your IBM credentials here '''

#backend = service.backend(name="your_QPU")

backend = AerSimulator(seed_simulator=1)

def run_circuits_on_backend(circuits):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    pubs = []

    """observables = [my_obs_list] * len(circuits)
    for qc, obs in zip(circuits, observables):
        isa_circuit = pm.run(qc)
        isa_obs = obs.apply_layout(isa_circuit.layout)
        pubs.append((isa_circuit, isa_obs))"""

    for qc in circuits:
        isa_circuit = pm.run(qc)
        pubs.append(isa_circuit)

    sampler = Sampler(backend, options={"default_shots": int(10_000)})
    sampler.options.execution.rep_delay = 0.001
    #sampler.options.twirling.enable_gates = True
    #sampler.options.twirling.enable_measure = True
    job = sampler.run(pubs)

    return job

def input_block(a_in: float, s_k: float) -> np.array(complex):
    """
    """
    qc = QuantumCircuit(2)
    qc.rx(theta=a_in*s_k, qubit=0)
    qc.rx(theta=a_in * s_k, qubit=1)
    qc.cx(control_qubit=0, target_qubit=1)
    qc.rz(phi=a_in * s_k, qubit=1)
    qc.cx(control_qubit=0, target_qubit=1)
    return qc


def evaluate(mode: str, pred: np.array(float), trajectory: np.array(float), lw: int, ltr: int, lts: int, tf: int=0) -> float:
    """
    We evaluate the models performance given some measure, shich is specified in mode.
    :param mode: Evaluation mode
    :param pred: Prediction vector
    :param trajectory: The entire time seriens
    :param lw: Length of washout
    :param ltr: Length of training
    :param lts: Length of testing
    :param tf: The timestep we want to predict
    :return: The performance
    """
    y_true = []
    for i in range(lts):
        label_idx = lw + ltr + i + tf
        y_true.append(trajectory[label_idx])
    y_true = np.array(y_true)

    if mode == "short_term_mem_cap":
        R_squared = np.cov(y_true, pred)[0, 1] ** 2 / (np.var(y_true) * np.var(pred))
        return R_squared
    elif mode == "NMSE":
        numerator = np.linalg.norm(y_true - pred) ** 2  # Squared Euclidean norm
        denominator = np.linalg.norm(y_true) ** 2  # Squared Euclidean norm of y_true
        return numerator / denominator


lw = 5
ltr = 25
lts = 20

def create_circuit(N, U_res, trajectory, a_in, a_fb):
    w_trajectory = trajectory[:lw]
    tr_trajectory = trajectory[lw:lw + ltr]
    ts_trajectory = trajectory[lw + ltr:lw + ltr + lts]

    qreg = QuantumRegister(N, 'q')

    cregs = [ClassicalRegister(N, f'c{i}') for i in range(lw+ltr+lts)]

    dynamic_circuit = QuantumCircuit(qreg, *cregs)

    assert len(w_trajectory) == lw, "The washout trajectory needs to be the same length as in lw specified."
    assert len(tr_trajectory) == ltr, "The training trajectory needs to be the same length as in ltr specified."
    assert len(ts_trajectory) == lts, "The testing trajectory needs to be the same length as in lts specified."

    for index, s_k in enumerate(trajectory[:lw + ltr + lts]):
        if index == 0:
            # Input block
            dynamic_circuit.append(input_block(a_in=a_in, s_k=s_k), [0, 1])
            # Random bitstring 10 for initialisation
            # Start with the first fb block
            dynamic_circuit.rx(theta=a_fb * 1, qubit=0)
            dynamic_circuit.rx(theta=a_fb * 1, qubit=1)
            dynamic_circuit.cx(control_qubit=0, target_qubit=1)
            dynamic_circuit.rz(phi=a_fb * 1, qubit=1)
            dynamic_circuit.cx(control_qubit=0, target_qubit=1)
            #Second fb block
            dynamic_circuit.rx(theta=a_fb * -1, qubit=0)
            dynamic_circuit.rx(theta=a_fb * -1, qubit=1)
            dynamic_circuit.cx(control_qubit=0, target_qubit=1)
            dynamic_circuit.rz(phi=a_fb * -1, qubit=1)
            dynamic_circuit.cx(control_qubit=0, target_qubit=1)
            dynamic_circuit.append(UnitaryGate(U_res), [0, 1])
            dynamic_circuit.measure(range(2), cregs[index])
        else:
            with dynamic_circuit.switch(cregs[index-1]) as case:
                with case(0b00):
                    # The reset strategy

                    # The input block
                    dynamic_circuit.append(input_block(a_in=a_in, s_k=s_k), [0, 1])
                    # Start with the first fb block
                    dynamic_circuit.rx(theta=a_fb * -1, qubit=0)
                    dynamic_circuit.rx(theta=a_fb * -1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)
                    dynamic_circuit.rz(phi=a_fb * -1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)
                    # Second fb block
                    dynamic_circuit.rx(theta=a_fb * -1, qubit=0)
                    dynamic_circuit.rx(theta=a_fb * -1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)
                    dynamic_circuit.rz(phi=a_fb * -1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)
                with case(0b01):
                    # The reset strategy
                    dynamic_circuit.x(qubit=0)
                    # The input block
                    dynamic_circuit.append(input_block(a_in=a_in, s_k=s_k), [0, 1])
                    # Start with the first fb block
                    dynamic_circuit.rx(theta=a_fb * -1, qubit=0)
                    dynamic_circuit.rx(theta=a_fb * -1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)
                    dynamic_circuit.rz(phi=a_fb * -1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)
                    # Second fb block
                    dynamic_circuit.rx(theta=a_fb * 1, qubit=0)
                    dynamic_circuit.rx(theta=a_fb * 1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)
                    dynamic_circuit.rz(phi=a_fb * 1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)
                with case(0b10):
                    # The reset strategy
                    dynamic_circuit.x(qubit=1)
                    # The input block
                    dynamic_circuit.append(input_block(a_in=a_in, s_k=s_k), [0, 1])
                    # Start with the first fb block
                    dynamic_circuit.rx(theta=a_fb * 1, qubit=0)
                    dynamic_circuit.rx(theta=a_fb * 1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)
                    dynamic_circuit.rz(phi=a_fb * 1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)
                    # Second fb block
                    dynamic_circuit.rx(theta=a_fb * -1, qubit=0)
                    dynamic_circuit.rx(theta=a_fb * -1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)
                    dynamic_circuit.rz(phi=a_fb * -1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)
                with case(0b11):
                    # The reset strategy
                    dynamic_circuit.x(qubit=range(2))
                    # The input block
                    dynamic_circuit.append(input_block(a_in=a_in, s_k=s_k), [0, 1])
                    # Start with the first fb block
                    dynamic_circuit.rx(theta=a_fb * 1, qubit=0)
                    dynamic_circuit.rx(theta=a_fb * 1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)
                    dynamic_circuit.rz(phi=a_fb * 1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)
                    # Second fb block
                    dynamic_circuit.rx(theta=a_fb * 1, qubit=0)
                    dynamic_circuit.rx(theta=a_fb * 1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)
                    dynamic_circuit.rz(phi=a_fb * 1, qubit=1)
                    dynamic_circuit.cx(control_qubit=0, target_qubit=1)

            dynamic_circuit.append(UnitaryGate(U_res), [0, 1])
            dynamic_circuit.measure(range(2), cregs[index])
    return dynamic_circuit

N=2
tf = 0
np.random.seed(10)
a=unitary_group.rvs(2 ** 2)
#q1dim = QuantumIsingChain(N=5, J=1, hx=1.05, hz=-0.5)
#seq = trajectory = q1dim.get_seq_expectation_value(spin_site=2, timesteps=3000)
l = mackey_glass.MackeyGlassSequence(alpha=0.2, beta=10, gamma=0.1, td=17)
seq = trajectory = l.get_mackey_glass_sequence(N=10000)
qc = create_circuit(N=2, U_res=a, trajectory=seq[2000:], a_in=1.0, a_fb=2.0)
res = run_circuits_on_backend([qc])


expectation_values_training = []
for i in range(ltr):
    index = lw+i
    reg = getattr(res.result()[0].data, f"c{index}")
    dict = reg.get_counts()
    expectation_values_training.append(utils.get_expectation_values(dict))

# Creating X_tr for linear regression
assert len(expectation_values_training) == ltr
X_tr = np.array([z_vector + [1] for z_vector in expectation_values_training])
assert X_tr.shape == (ltr, N + 1)

#Creating y_true
y_true = []
for i in range(ltr):
    label_idx = lw + i + tf
    y_true.append(trajectory[label_idx])
assert len(y_true) == ltr

# Doing linear regression with pseudoinverse
w_opt = np.linalg.inv(X_tr.T @ X_tr) @ X_tr.T @ y_true
assert w_opt.shape == (N + 1,)


expectation_values_testing = []
for i in range(lts):
    index = lw+ltr+i
    reg = getattr(res.result()[0].data, f"c{index}")
    dict = reg.get_counts()
    expectation_values_testing.append(utils.get_expectation_values(dict))

X_ts = np.array([z_vector + [1] for z_vector in expectation_values_testing])
pred = X_ts @ w_opt

res_ev = evaluate(mode="NMSE", pred=pred, trajectory=trajectory, lw=lw, ltr=ltr, lts=lts, tf=tf)
print(res_ev)
