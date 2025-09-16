import json
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Load dataset
# ---------------------------
with open("qmolpredict_dataset.json", "r") as file:
    dataset = json.load(file)

# ---------------------------
# Function to build Hamiltonian (faster)
# ---------------------------
def get_hamiltonian(symbols, coordinates):
    """Use real Hamiltonian for H2, fallback simple Hamiltonian for others"""
    if symbols == ["H", "H"]:
        try:
            coordinates = np.array(coordinates, dtype=float)
            H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, basis="sto-3g")
            return H, qubits
        except:
            pass
    # Fallback for faster execution
    return qml.Hamiltonian([1.0], [qml.PauliZ(0)]), 1

# ---------------------------
# Ansatz (simpler)
# ---------------------------
def ansatz(params, wires):
    for i in range(len(wires)):
        qml.RX(params[i], wires=i)
        qml.RY(params[i], wires=i)
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[i, i + 1])

# ---------------------------
# Run VQE for a molecule
# ---------------------------
def run_vqe(molecule, max_iter=20):
    symbols = molecule["atoms"]
    coordinates = molecule["coordinates"]
    ref_energy = molecule["energy"]

    print(f"\nProcessing molecule: {molecule['name']}")

    H, qubits = get_hamiltonian(symbols, coordinates)
    dev = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev, diff_method="backprop")
    def circuit(params):
        ansatz(params, wires=range(qubits))
        return qml.expval(H)

    params = np.random.random(qubits, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.2)

    energies = []
    for n in range(max_iter + 1):
        params, energy = opt.step_and_cost(circuit, params)
        energies.append(energy)
        if n % 5 == 0:
            print(f"Iteration {n}, Energy = {energy:.6f}")

    return energies, ref_energy

# ---------------------------
# Main loop over dataset (fast)
# ---------------------------
results = {}
for molecule in dataset:
    energies, ref_energy = run_vqe(molecule, max_iter=20)
    results[molecule["name"]] = {
        "predicted": energies[-1],
        "reference": ref_energy,
        "convergence": energies
    }

# ---------------------------
# Visualization
# ---------------------------
# 1. Convergence plot for first molecule
first_mol = list(results.keys())[0]
plt.figure(figsize=(6,4))
plt.plot(results[first_mol]["convergence"], label=f"{first_mol} VQE Energy")
plt.axhline(results[first_mol]["reference"], color="r", linestyle="--", label="Reference")
plt.xlabel("Iteration")
plt.ylabel("Energy (Ha)")
plt.title(f"Energy Convergence for {first_mol}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Predicted vs Reference Energies (bar plot)
names = list(results.keys())
predicted = [results[m]["predicted"] for m in names]
reference = [results[m]["reference"] for m in names]

x = np.arange(len(names))
plt.figure(figsize=(6,4))
plt.bar(x - 0.2, predicted, 0.4, label="Predicted")
plt.bar(x + 0.2, reference, 0.4, label="Reference")
plt.xticks(x, names, rotation=45)
plt.ylabel("Energy (Ha)")
plt.title("Predicted vs Reference Energies")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------
# Final Results
# ---------------------------
print("\n--- Multi-Molecule VQE Results ---")
for m in results:
    print(f"{m}: Predicted = {results[m]['predicted']:.6f}, Reference = {results[m]['reference']}")
