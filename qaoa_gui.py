import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

def get_qubo_matrix():
    print("Enter a QUBO matrix as a Python-style nested list (e.g., [[-1, 2], [0, -1]])")
    user_input = input("Enter 0 to use the default QUBO: ")
    
    if user_input.strip() == '0':
        print("Using default 3-qubit QUBO.")
        return np.array([
            [-1,  2,  0],
            [ 2, -1,  2],
            [ 0,  2, -1]
        ])
    else:
        try:
            import ast
            matrix = np.array(ast.literal_eval(user_input))
            if len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]:
                return matrix
            else:
                print("Invalid shape. Using default 3-qubit QUBO.")
        except Exception as e:
            print(f"Error parsing input ({e}). Using default 3-qubit QUBO.")
            
        return np.array([[-1, 2, 0], [2, -1, 2], [0, 2, -1]])

def build_qaoa_assets(qubo):
    from qiskit.circuit.library import QAOAAnsatz
    n = qubo.shape[0]
    
    # Track pauli strings for the Cost Hamiltonian
    pauli_list = []
    
    for i in range(n):
        weight = qubo[i, i]
        if weight != 0:
            p = ["I"] * n
            p[n - 1 - i] = "Z"
            pauli_list.append(("".join(p), weight))
        
    for i in range(n):
        for j in range(i + 1, n):
            weight = qubo[i, j] + qubo[j, i]
            if weight != 0:
                p = ["I"] * n
                p[n - 1 - i] = "Z"
                p[n - 1 - j] = "Z"
                pauli_list.append(("".join(p), weight))
                
    cost_op = SparsePauliOp.from_list(pauli_list) if pauli_list else SparsePauliOp("I"*n)
    
    qaoa = QAOAAnsatz(cost_op, reps=1)
    
    # Decompose to collapse the nested blocks so it's measurable and drawable
    qc = qaoa.decompose(reps=3)
    
    meas_qc = qc.copy()
    meas_qc.measure_all()
    
    return qc, meas_qc, cost_op, n

def main():
    qubo = get_qubo_matrix()
    
    print("\n--- Input QUBO Matrix ---")
    print(qubo)
    
    n_bits = qubo.shape[0]
    print("\n--- Classical QUBO Costs for each bitstring ---")
    for i in range(2**n_bits):
        b_str = f"{i:0{n_bits}b}"
        # Qiskit lists qubit 0 as the rightmost bit, so we reverse it to map x[0] to Qubit 0
        x = np.array([int(bit) for bit in reversed(b_str)])
        cost = x @ qubo @ x.T
        print(f"[{b_str}]: Cost = {cost}")
        
    qc_unmeasured, meas_qc, cost_op, n_qubits = build_qaoa_assets(qubo)
    
    print("\n--- 1-Layer QAOA Circuit (with measurements) ---")
    print(meas_qc.draw("text"))
    
    # Identity dynamically generated parameter order from QAOAAnsatz
    params = qc_unmeasured.parameters
    b_idx = 0 if ('β' in params[0].name or 'beta' in params[0].name) else 1
    g_idx = 1 - b_idx
    
    # 1. Compute the 3D Landscape for the base plot
    print("\nPre-computing QAOA Energy landscape...")
    estimator = StatevectorEstimator()
    sampl_points = 20
    b_range = np.linspace(0, np.pi, sampl_points)
    g_range = np.linspace(0, 2*np.pi, sampl_points)
    B, G = np.meshgrid(b_range, g_range)
    
    b_flat = B.flatten()
    g_flat = G.flatten()
    
    param_grid = np.zeros((len(b_flat), 2))
    param_grid[:, b_idx] = b_flat
    param_grid[:, g_idx] = g_flat
    
    job_est = estimator.run([(qc_unmeasured, cost_op, param_grid)])
    evs = job_est.result()[0].data.evs
    E_surf = evs.reshape(sampl_points, sampl_points)
    print("Done!")

    # 2. Setup the split layout interactive figure
    fig = plt.figure(figsize=(14, 6))
    plt.subplots_adjust(bottom=0.25)
    
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_bar = fig.add_subplot(122)
    
    # Draw the static 3D surface
    surf = ax_3d.plot_surface(B, G, E_surf, cmap='plasma', alpha=0.8, edgecolor='none')
    fig.colorbar(surf, ax=ax_3d, label="Expected Energy $\\langle H_C \\rangle$", shrink=0.5)
    ax_3d.set_title("QAOA Cost Landscape")
    ax_3d.set_xlabel("$\\beta$ (Mixer)")
    ax_3d.set_ylabel("$\\gamma$ (Cost)")
    ax_3d.set_zlabel("Energy")
    
    # We will hold a reference to the active point scatter plot
    active_point = [None]
    
    sampler = StatevectorSampler()
    init_beta = np.pi / 4
    init_gamma = 1.0

    def update_plot(b, g):
        # Pack parameter arrays safely
        param_list = [0, 0]
        param_list[b_idx] = b
        param_list[g_idx] = g
        
        # Sampling for the bar chart
        job_samp = sampler.run([(meas_qc, param_list)], shots=1000)
        counts = job_samp.result()[0].data.meas.get_counts()
        
        # Estimate single exact point energy for the 3D scatter dot
        job_point = estimator.run([(qc_unmeasured, cost_op, [param_list])])
        exact_energy = job_point.result()[0].data.evs[0]

        # Update Bar Chart
        ax_bar.clear()
        bitstrings = [f"{i:0{n_qubits}b}" for i in range(2**n_qubits)]
        frequencies = [counts.get(b_s, 0) for b_s in bitstrings]
        
        ax_bar.bar(bitstrings, frequencies, color='teal', alpha=0.8, edgecolor='black')
        ax_bar.set_title(f"QAOA Measured Counts (1000 shots)\n$\\beta={b:.2f}$, $\\gamma={g:.2f}$ (Energy = {exact_energy:.2f})")
        ax_bar.set_xlabel("Bitstring")
        ax_bar.set_ylabel("Counts")
        ax_bar.set_ylim(0, 1000)
        ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
        ax_bar.tick_params(axis='x', rotation=45)
        
        # Update 3D Dot
        if active_point[0] is not None:
            active_point[0].remove()
        
        # Draw a big red dot natively pinned to the evaluated energy surface coordinates
        active_point[0] = ax_3d.scatter(b, g, exact_energy + 0.1, color='red', s=100, zorder=10)
        
        fig.canvas.draw_idle()

    # Sliders
    ax_beta = plt.axes([0.15, 0.10, 0.3, 0.03])
    ax_gamma = plt.axes([0.55, 0.10, 0.3, 0.03])
    
    slider_beta = Slider(ax_beta, 'Beta (β)', 0.0, np.pi, valinit=init_beta, valfmt='%0.2f')
    slider_gamma = Slider(ax_gamma, 'Gamma (γ)', 0.0, 2*np.pi, valinit=init_gamma, valfmt='%0.2f')
    
    # Button
    ax_button = plt.axes([0.4, 0.02, 0.2, 0.06])
    btn_sample = Button(ax_button, 'Sample & Update Dot', hovercolor='0.975')
    
    def trigger(event):
        update_plot(slider_beta.val, slider_gamma.val)
        
    btn_sample.on_clicked(trigger)
    
    update_plot(init_beta, init_gamma)
    
    print("\nInteractive QAOA GUI... Adjust sliders and click Sample!")
    plt.show()

if __name__ == "__main__":
    main()
