import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

# 1. Define the QUBO and corresponding Cost Operator
# We use the default 3-qubit QUBO to align with qaoa_gui.py
qubo = np.array([
    [-1,  2,  0],
    [ 2, -1,  2],
    [ 0,  2, -1]
])
n = qubo.shape[0]

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

# 2. Build QAOA Circuit from Qiskit Native Standard
qaoa = QAOAAnsatz(cost_op, reps=1)
qc = qaoa.decompose(reps=3)

# Safely extract QAOA parameters dynamically mapping to precisely Beta and Gamma
params = qc.parameters
b_idx = 0 if ('β' in params[0].name or 'beta' in params[0].name.lower()) else 1
g_idx = 1 - b_idx

# 3. Define the parameter variations on a 2D Grid
points = 20
b_range = np.linspace(0, np.pi, points)
g_range = np.linspace(0, 2*np.pi, points)
B, G = np.meshgrid(b_range, g_range)

b_flat = B.flatten()
g_flat = G.flatten()

param_values = np.zeros((len(b_flat), 2))
param_values[:, b_idx] = b_flat
param_values[:, g_idx] = g_flat

# 4. Use Estimator to get Expectation Values (Cost Landscape)
estimator = StatevectorEstimator()
job = estimator.run([(qc, cost_op, param_values)])
evs_flat = job.result()[0].data.evs
E_surf = evs_flat.reshape(points, points)

# 5. Finite-difference parameter shift for Gradients
shift = np.pi / 4

param_plus_b = param_values.copy()
param_plus_b[:, b_idx] += shift
param_minus_b = param_values.copy()
param_minus_b[:, b_idx] -= shift

param_plus_g = param_values.copy()
param_plus_g[:, g_idx] += shift
param_minus_g = param_values.copy()
param_minus_g[:, g_idx] -= shift

# Run estimator for all shifted parameter sets
job_shift = estimator.run([
    (qc, cost_op, param_plus_b),
    (qc, cost_op, param_minus_b),
    (qc, cost_op, param_plus_g),
    (qc, cost_op, param_minus_g)
])
results = job_shift.result()

# Calculate macroscopic partial gradients mapped across the arrays
grad_b_flat = 0.5 * (results[0].data.evs - results[1].data.evs)
grad_g_flat = 0.5 * (results[2].data.evs - results[3].data.evs)

# 6. Calculate Full Covariance Matrix 
# Stacking Beta, Gamma, Expectation, Grad(Beta), Grad(Gamma)
data_matrix = np.vstack((b_flat, g_flat, evs_flat, grad_b_flat, grad_g_flat))
covariance_matrix = np.cov(data_matrix)

print("\n--- QAOA Circuit ---")
print(qc.draw("text"))

print("\n5x5 Covariance Matrix (Beta, Gamma, Cost, Grad_Beta, Grad_Gamma):")
print(np.round(covariance_matrix, 4))
print(f"\nVariance of Grad(Beta): {covariance_matrix[3, 3]:.4f}")
print(f"Variance of Grad(Gamma): {covariance_matrix[4, 4]:.4f}")

# 7. Plotting the statical 3D Surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(B, G, E_surf, cmap='viridis', alpha=0.9, edgecolor='none')
fig.colorbar(surf, ax=ax, label="Expected Cost $\\langle H_C \\rangle$")

ax.set_title("3D Landscape of QAOA Cost")
ax.set_xlabel("$\\beta$ (Mixer) radians")
ax.set_ylabel("$\\gamma$ (Cost) radians")
ax.set_zlabel("Cost $\\langle H_C \\rangle$")

plt.savefig("plot_3d.png", dpi=300)
print("\nPlot saved successfully to 'plot_3d.png'")

# Display the plot window interactively
plt.show()
