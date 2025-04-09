# Difference to 1.2: I do the entire training and testing in one circuit. In 1.2 Training and testing is seperated

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator, partial_trace, DensityMatrix
#from qiskit.circuit.library import RealAmplitudes, EfficientSU2
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.linear_model import Ridge
#from qiskit_ibm_runtime import SamplerV2
#from qiskit_aer import AerSimulator
#from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import utils
from tqdm import tqdm
import mackey_glass
import scipy
import narma_sequence
from states_gates import RZ, RX, CNOT, RY, rho_0_state, get_local_Z_observables
import functools as ft
import multiprocessing
from scipy.stats import unitary_group
import random
from itertools import product
from ising_chain import QuantumIsingChain
import pickle

n_shots = 5000
#l = []

#np.random.seed(7910)
def get_U_res(N: int) -> np.array(complex):
    """
    For the commented code: The approximate Haar random unitary as numpy array
    Here we approximate the Haar random unitary. We start with a block on ry(pi/4) gates on all qubis. Then
    we layers of blocks consisting of random single qubit rotations RX RY RZ with random angles.
    This is followed by nearest neighbour CNOTs. We repeat that block for layers times.

    For the uncommented code: We just return the Haar random unitary given via unitary_group.rvs(2**N)
    from scipy.

    :param N: Number of qubits
    :param layers: how many layers of the random unitary one qubit interactions
    :return: The U-res unitary
    """
    return unitary_group.rvs(2 ** N)


def input_block(a_in: float, s_k: float) -> np.array(complex):
    """
    Computes the input block given via CNOT @ I RZ @ CNOT @ RX RX.
    :param a_in: a_in input parameter weight
    :param s_k: Current timestep of the trajectory to feed in the input block
    :return: The unitary of the input block
    """
    return CNOT() @ np.kron(np.eye(2), RZ(a_in * s_k)) @ CNOT() @ np.kron(RX(a_in * s_k), RX(a_in * s_k))


def feeback_block(a_fb: float, z_i: int) -> np.array(complex):
    """
    Computes the input block given via CNOT @ I RZ @ CNOT @ RX RX.
    :param a_fb: a_fb feedback parameter weight
    :param z_i: feedback value
    :return: The unitary of the feedback block
    """
    # Change  measurement outputs from 0, 1, to -1 , 1
    if z_i == 0:
        z_i = -1
    return CNOT() @ np.kron(np.eye(2), RZ(a_fb * z_i)) @ CNOT() @ np.kron(RX(a_fb * z_i), RX(a_fb * z_i))


def measure_single_shot(rho: np.array(complex), N: int, projector: np.array(complex)) -> list[int]:
    """
    This function computes the measurement outcome of one single shot given a projector.
    :param rho: State to measure.
    :param N: System size
    :param projector: Measurement projector
    :return: The measurement outcome as list of binary integers
    """
    z_list = []
    # We go over each qubit seperately
    for k in range(N):
        # We trace out the according subsystem
        qargs_to_trace_out = [i for i in range(N) if i != k]
        reduced_density_matrix = partial_trace(state=DensityMatrix(rho), qargs=qargs_to_trace_out)
        # The probability of measuring +1 is given via P(+1) = tr(rho*P_+1), where P_+1 is the projector for the eigenstate +1
        prob_plus1 = np.trace(np.array(reduced_density_matrix) @ projector).real
        # Draw a random sample from this obtained probability distribution. This simulates one shot.
        outcome = int(np.random.choice([0, 1], p=[prob_plus1, 1 - prob_plus1]))
        z_list.append(outcome)
    return z_list


def get_U_QRC(N: int, a_in: float, a_fb: float, s_k: float, z: np.array(float), U_res: np.array(complex)) -> np.array(
    complex):
    """
    This function returns the entire unitary of the quantum reservoir computing scheme, consisting of all the entire input and
    feedback blocks and the Haar random unitary.

    I devide the computation into steps, where each step corresponds to one "column" of the unitary circuit (if you divide the circuit into columns).
    :param N: System size / number of qubits
    :param a_in: a_in input parameter weight
    :param a_fb: a_fb feedback parameter weight
    :param s_k: Current timestep of the trajectory to feed in the input block
    :param z: Feedback vector we want to feed into the current timestep
    :param U_res: Haar random unitary
    :return: The unitary
    """
    assert len(z) == N, "The len of the z_vector vector needs to be the same size as the system"

    if N == 2:
        # First step
        first_step_U = input_block(a_in=a_in, s_k=s_k)

        # Second step
        second_step_U = feeback_block(a_fb, z_i=z[0])

        # Third step
        third_step_U = feeback_block(a_fb, z_i=z[1])

        return U_res @ third_step_U @ second_step_U @ first_step_U

    elif N == 4:
        # First step
        first_step_matrices = [input_block(a_in=a_in, s_k=s_k), feeback_block(a_fb, z_i=z[0])]
        first_step_U = ft.reduce(np.kron, first_step_matrices)

        # Second step
        second_step_matrices = [np.eye(2 ** 2), feeback_block(a_fb, z_i=z[1])]
        second_step_U = ft.reduce(np.kron, second_step_matrices)

        #Third step
        third_step_matrices = [np.eye(2 ** 2), feeback_block(a_fb, z_i=z[2])]
        third_step_U = ft.reduce(np.kron, third_step_matrices)

        # Fourth step
        fourth_step_matrices = [np.eye(2 ** 2), feeback_block(a_fb, z_i=z[3])]
        fourth_step_U = ft.reduce(np.kron, fourth_step_matrices)

        return U_res @ fourth_step_U @ third_step_U @ second_step_U @ first_step_U
    elif N == 6:
        # First step
        first_step_matrices = [input_block(a_in=a_in, s_k=s_k), feeback_block(a_fb, z_i=z[0]),
                               feeback_block(a_fb, z_i=z[1])]
        first_step_U = ft.reduce(np.kron, first_step_matrices)

        # Second step
        second_step_matrices = [np.eye(2 ** 3), feeback_block(a_fb, z_i=z[2]), np.eye(2)]
        second_step_U = ft.reduce(np.kron, second_step_matrices)

        # Third step
        third_step_matrices = [np.eye(2 ** 2), feeback_block(a_fb, z_i=z[3]), feeback_block(a_fb, z_i=z[4])]
        third_step_U = ft.reduce(np.kron, third_step_matrices)

        # Fourth step
        fourth_step_matrices = [np.eye(2 ** 3), feeback_block(a_fb, z_i=z[5]), np.eye(2)]
        fourth_step_U = ft.reduce(np.kron, fourth_step_matrices)

        return U_res @ fourth_step_U @ third_step_U @ second_step_U @ first_step_U

    elif N == 8:
        # First step
        first_step_matrices = [input_block(a_in=a_in, s_k=s_k), feeback_block(a_fb, z_i=z[0]),
                               feeback_block(a_fb, z_i=z[1]), feeback_block(a_fb, z_i=z[2])]
        first_step_U = ft.reduce(np.kron, first_step_matrices)

        # Second step
        second_step_matrices = [np.eye(2 ** 3), feeback_block(a_fb, z_i=z[3]), feeback_block(a_fb, z_i=z[4]), np.eye(2)]
        second_step_U = ft.reduce(np.kron, second_step_matrices)

        # Third step
        third_step_matrices = [np.eye(2 ** 2), feeback_block(a_fb, z_i=z[5]), feeback_block(a_fb, z_i=z[6]),
                               feeback_block(a_fb, z_i=z[7])]
        third_step_U = ft.reduce(np.kron, third_step_matrices)

        return U_res @ third_step_U @ second_step_U @ first_step_U

    else:
        print("Not implemented")


def create_list_of_dicts(N: int, lenght: int) -> list[dict]:
    """
    This simple helper function creates a list of dictionaries. These dictionaries are filled with all possible
    measurement outcomes of a given system size N as keys. The value for each key is initialized with 0.
    :param N: System size / number of qubits
    :param lenght: Number of dictionaries you want. Usually that is lw + ltr or just ltr.
    :return: List of dictionaries
    """
    # Generate all k-bit combinations as strings, e.g. "00", "01", ...
    keys = [''.join(bits) for bits in product('01', repeat=N)]
    # Build each dictionary with all keys mapping to 0
    # and repeat it n times in a list
    return [dict.fromkeys(keys, 0) for _ in range(lenght)]


def do_routine(N: int, lw: int, ltr: int, lts:int, trajectory: np.array(float), a_in: float, a_fb: float, U_res: np.array(complex), tf: int=0) -> (np.array(float), list):
    """
    This routine does the entire training and washout routine. It returns the optimized weight matrix
    and the list of all measurement outcomes so far.

    :param N: System size / number of qubits
    :param lw: Length of washout routine
    :param ltr: Length of training routine
    :param trajectory: The trajectory/time series
    :param a_in: a_in input parameter weight
    :param a_fb: a_fb feedback parameter weight
    :param U_res: Haar random unitary
    :param tf: The timestep we want to predict
    :return: The optimized weight matrix and the z_list. (I kept the z_list for debugging and testing purposes. But you
    were totally right! We can just take one variable and always overwrite this variable with your latest measurement
    value. It works the same. When we do further bigger tests I can just replace z_list wit z_last_value: int.)
    """
    z_list = [np.random.choice([0, 1], size=N)]
    start_state = rho_0_state(N)
    projector_0 = np.outer(utils.ket_0_state(1), utils.ket_0_state(1).conj())

    w_trajectory = trajectory[:lw]
    tr_trajectory = trajectory[lw:lw + ltr]
    ts_trajectory = trajectory[lw + ltr:lw + ltr + lts]

    cl_regs = create_list_of_dicts(N=N, lenght=lw+ltr+lts)

    assert len(w_trajectory) == lw, "The washout trajectory needs to be the same length as in lw specified."
    assert len(tr_trajectory) == ltr, "The training trajectory needs to be the same length as in ltr specified."
    assert len(ts_trajectory) == lts, "The testing trajectory needs to be the same length as in lts specified."


    # For each shot we do the entre routine
    for _ in range(n_shots):
        # Ensure that each shot starts with a fresh random sample
        z_list.append(np.random.choice([0, 1], size=N))
        # Doing the washout routine
        for i, s_k in enumerate(w_trajectory):
            U = get_U_QRC(N=N, a_in=a_in, a_fb=a_fb, s_k=s_k, z=z_list[-1], U_res=U_res)
            evolved_state = U @ start_state @ U.conj().T
            measurement_result = measure_single_shot(rho=evolved_state, N=N, projector=projector_0)
            z_list.append(measurement_result)
            cl_regs[i]["".join(map(str, measurement_result))] += 1

        # The actual training
        for j, s_k in enumerate(tr_trajectory):
            U = get_U_QRC(N=N, a_in=a_in, a_fb=a_fb, s_k=s_k, z=z_list[-1], U_res=U_res)
            evolved_state = U @ start_state @ U.conj().T
            measurement_result = measure_single_shot(rho=evolved_state, N=N, projector=projector_0)
            z_list.append(measurement_result)
            cl_regs[lw + j]["".join(map(str, measurement_result))] += 1

        # The testing phase
        for k, s_k in enumerate(ts_trajectory):
            U = get_U_QRC(N=N, a_in=a_in, a_fb=a_fb, s_k=s_k, z=z_list[-1], U_res=U_res)
            evolved_state = U @ start_state @ U.conj().T
            measurement_result = measure_single_shot(rho=evolved_state, N=N, projector=projector_0)
            z_list.append(measurement_result)
            cl_regs[lw+ltr+k]["".join(map(str, measurement_result))] += 1


    # Creating expectation values
    assert len(cl_regs[lw: lw + ltr]) == ltr
    expectation_values_training = []
    for dic in cl_regs[lw: lw + ltr]:
        expectation_values_training.append(utils.get_expectation_values(dic))

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
    assert len(cl_regs[lw + ltr:]) == lts
    for dic in cl_regs[lw + ltr:]:
        expectation_values_testing.append(utils.get_expectation_values(dic))
    assert len(expectation_values_testing) == lts
    X_ts = np.array([z_vector + [1] for z_vector in expectation_values_testing])
    pred = X_ts @ w_opt

    return pred



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


# Define at the top level (not inside a function)
def compute_single_iteration(args) -> float:
    """
    Just a wrapper function used for multiprocessing
    :param args:  tf, N, lw, ltr, lts, trajectory, a_in, a_fb, layers
    :return: The result of our evaluation
    """
    # The arguments defined in this order
    tf, N, lw, ltr, lts, trajectory, a_in, a_fb, U_res = args
    pred = do_routine(N=N, lw=lw, ltr=ltr, lts=lts, trajectory=trajectory, a_in=a_in, a_fb=a_fb, U_res=U_res, tf=tf)
    res = evaluate(mode="NMSE", pred=pred, trajectory=trajectory, lw=lw, ltr=ltr, lts=lts, tf=tf)
    return res

def compute_single_iteration_C_S(args) -> (float, list):
    """
    Just a wrapper function used for multiprocessing for the C_Sigma task
    :param args:  tf_list, N, lw, ltr, lts, trajectory, a_in, a_fb, layers
    :return: The result of our evaluation
    """
    # The arguments defined in this order
    tf_list, N, lw, ltr, lts, trajectory, a_in, a_fb, U_res = args
    list_of_r_values = []
    for d_value in tf_list:
        pred = do_routine(N=N, lw=lw, ltr=ltr, lts=lts, trajectory=trajectory, a_in=a_in, a_fb=a_fb, U_res=U_res, tf=d_value)
        res = evaluate(mode="short_term_mem_cap", pred=pred, trajectory=trajectory, lw=lw, ltr=ltr, lts=lts, tf=d_value)
        list_of_r_values.append(res)

    return sum(list_of_r_values), list_of_r_values


if __name__ == '__main__':
    # Here in the main function some routines are already specified.

    lw = 25
    ltr = 100
    lts = 100
    tfi = 0


    # Testing NMSE
    N = 2
    #seq = cosine_trajectory = [np.cos((np.pi/25) * k) for k in range(1, 20_000)]
    delay_values = range(10)
    q1dim = QuantumIsingChain(N=5, J=1, hx=1.05, hz=-0.5)
    seq = q1dim.get_seq_expectation_value(spin_site=2, timesteps=3000)
    num_tasks = 128  # Total number of tasks
    #args_list = [(tfi, 2, lw, ltr, lts, seq[10_000:], 2.0, 2.0, 10) for _ in range(num_tasks)]
    #l = mackey_glass.MackeyGlassSequence(alpha=0.2, beta=10, gamma=0.1, td=17)
    #seq = l.get_mackey_glass_sequence(N=10000)
    a_fb_list = [2.2]
    U_res_list = [get_U_res(N) for _ in range(num_tasks)]
    all_results = []  # We'll store one dictionary per (a_fb, d_value) combination.

    for a_fb in a_fb_list:
        for d_value in delay_values:
            current_d_value = d_value  # Ensure each worker gets the correct `d_value`
            args_list = [(current_d_value, 2, lw, ltr, lts, seq[2000:], 1.0, a_fb, U_res_list[i]) for i in range(num_tasks)]
            with multiprocessing.Pool(processes=128) as pool:
                results = []
                for result in pool.imap_unordered(compute_single_iteration, args_list):
                    results.append(result)
            res_mean = np.mean(results)
            res_std = np.std(results)
            print(f"a_fb = {a_fb:.3f}, d_value = {d_value}, STM mean = {res_mean}, std = {res_std}")

            # Store everything in a dictionary for easy serialization
            entry = {
                "a_fb": a_fb,
                "d_value": d_value,
                "subresults": results,  # all 128 values
                "mean_result": res_mean,
                "std_result": res_std
            }
            all_results.append(entry)

        # Now save the entire list to a pickle file
    with open("Experiments_Results/final_experiments/EXP_175_data.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print("Experiments_Results/final_experiments/EXP_175_data.pkl.")

    """
    # Testing R_squard
    N = 2
    num_tasks = 128
    trajectory = np.random.uniform(0, 1, size=10_000)
    a_fb_list = [1.3]
    delay_values = [0, -1, -2, -3]
    U_res_list = [get_U_res(N) for _ in range(num_tasks)]

    all_results = []  # We'll store one dictionary per (a_fb, d_value) combination.

    for a_fb in a_fb_list:
        for d_value in delay_values:
            current_d_value = d_value  # Ensure each worker gets the correct `d_value`
            args_list = [(current_d_value, N, lw, ltr, lts, trajectory, 1.0, a_fb, U_res_list[i]) for i in range(num_tasks)]
            with multiprocessing.Pool(processes=128) as pool:
                results = []
                for result in pool.imap_unordered(compute_single_iteration, args_list):
                    results.append(result)
            res_mean = np.mean(results)
            res_std = np.std(results)

            print(f"a_fb = {a_fb:.3f}, d_value = {d_value}, STM mean = {res_mean}, std = {res_std}")

            # Store everything in a dictionary for easy serialization
            entry = {
                "a_fb": a_fb,
                "d_value": d_value,
                "subresults": results,  # all 128 values
                "mean_result": res_mean,
                "std_result": res_std
            }
            all_results.append(entry)

    # Now save the entire list to a pickle file
    with open("Experiments_Results/final_experiments/EXP_200_data.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print("Experiments_Results/final_experiments/EXP_200_data.pkl.")"""

    '''
    # Testing C_Sigma
    N = 2
    num_tasks = 128
    trajectory = np.random.uniform(0, 1, size=10_000)
    delay_values = [0, -1, -2, -3]
    a_fb_list = np.arange(0.5, 2.3, 0.1)
    U_res_list = [get_U_res(N) for _ in range(num_tasks)]

    all_results = []  # We'll store one dictionary per (a_fb, c_sigma) combination.

    for a_fb in a_fb_list:
        args_list = [(delay_values, N, lw, ltr, lts, trajectory, 1.0, a_fb, U_res_list[i]) for i in range(num_tasks)]
        with multiprocessing.Pool(processes=128) as pool:
            results_cs = []
            results_r_values = []
            for c_s, r_values in pool.imap_unordered(compute_single_iteration_C_S, args_list):
                results_cs.append(c_s)
                results_r_values.append(r_values)
            res_mean = np.mean(results_cs)
            res_std = np.std(results_cs)

        print(f"a_fb = {a_fb:.3f}, C_Sigma mean = {res_mean}, std = {res_std}")

        # Store everything in a dictionary for easy serialization
        entry = {
            "a_fb": a_fb,
            "C_Sigma": res_mean,
            "subresults_c_sigma": results_cs,  # all 128 values
            "subresults_r_values": results_r_values,
            "mean_result": res_mean,
            "std_result": res_std
        }
        all_results.append(entry)

        # Now save the entire list to a pickle file
    with open("Experiments_Results/final_experiments/EXP_220_data.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print("Experiments_Results/final_experiments/EXP_220_data.pkl.")'''



