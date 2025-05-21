import matplotlib.pyplot as plt
import numpy as np

def dgcq_workflow(Vs, Vcm, Vref, max_iter=100):
    Vacc = 0
    Counter = 0
    history = []

    for i in range(max_iter):
        Vacc = (Vacc + Vs) / 2
        if Vacc > Vcm:
            Vota = 2 * Vacc - Vref
            Vacc = Vota
            Counter += 1
        else:
            Vota = 2 * Vacc
            Vacc = Vota
        history.append({'Vacc': Vacc, 'Vota': Vota, 'Counter': Counter})

    quantized = round(Vacc, 2)
    return quantized, history

Vs = 0.35144
Vcm = 0.25

Vrefs = np.linspace(0.1, 0.5, 5)  # 支援多組 Vref
# Vrefs = [0.2, 0.5, 0.8]  # 支援多組 Vref
# Vrefs = [1.0]
max_iters = np.arange(1, 1001, 10)
counter_matrix = np.zeros((len(Vrefs), len(max_iters)))

for i, Vref in enumerate(Vrefs):
    for j, max_iter in enumerate(max_iters):
        _, hist = dgcq_workflow(Vs, Vcm, Vref, max_iter)
        counter_matrix[i, j] = hist[-1]['Counter'] / max_iter * Vref

plt.figure(figsize=(8, 6))
for i, Vref in enumerate(Vrefs):
    plt.plot(
        max_iters,
        counter_matrix[i],
        marker='o',
        label=f'Vref={Vref:.2f}',
    )

plt.axhline(y=Vs, color='red', linestyle='--', label=f'Vs={Vs}')  # 加上紅色橫線

plt.xlabel('max_iter')
# plt.xscale('log')
plt.ylabel('Counter / max_iter')
plt.title('Counter/max_iter vs max_iter')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('counter_vs_max_iter.png', dpi=300)
plt.show()

Vrefs = [1.0]
max_iters = np.arange(1, 3001, 20)
num_samples = 1000  # 隨機 Vs 數量
bit_precision = np.zeros((len(max_iters),))

for j, max_iter in enumerate(max_iters):
    _, hist = dgcq_workflow(Vs, Vcm, Vrefs[0], max_iter)
    ratio = hist[-1]['Counter'] / max_iter
    error = abs(ratio - Vs)
    if error == 0:
        bit = 16  # 假設最大16 bit
    else:
        bit = -np.log2(error)
    bit_precision[j] = bit

plt.figure(figsize=(8, 6))
plt.plot(max_iters, bit_precision, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Bit precision (bits)')
plt.title('Bit precision of Coarse result vs Iterations')
plt.grid(True)
plt.tight_layout()
plt.savefig('bit_precision_vs_iterations.png', dpi=300)
plt.show()



# import matplotlib.pyplot as plt
# import numpy as np

# def dgcq_workflow(Vs, Vcm, Vref, max_iter=100):
#     Vacc = 0
#     Counter = 0
#     history = []

#     for i in range(max_iter):
#         Vacc = (Vacc + Vs) / 2
#         if Vacc > Vcm:
#             Vota = 2 * Vacc - Vref
#             Vacc = Vota
#             Counter += 1
#         else:
#             Vota = 2 * Vacc
#             Vacc = Vota
#         history.append({'Vacc': Vacc, 'Vota': Vota, 'Counter': Counter})

#     quantized = round(Vacc, 2)
#     return quantized, history

# Vs = 0.35144
# Vref = 1.0  # 固定 Vref
# Vcms = [0.15, 0.25]  # 多組 Vcm
# max_iters = np.arange(1, 1001, 10)
# counter_matrix = np.zeros((len(Vcms), len(max_iters)))

# for i, Vcm in enumerate(Vcms):
#     for j, max_iter in enumerate(max_iters):
#         _, hist = dgcq_workflow(Vs, Vcm, Vref, max_iter)
#         counter_matrix[i, j] = hist[-1]['Counter'] / max_iter * Vref

# plt.figure(figsize=(8, 6))
# for i, Vcm in enumerate(Vcms):
#     plt.plot(
#         max_iters,
#         counter_matrix[i],
#         marker='o',
#         label=f'Vcm={Vcm:.2f}',
#     )

# plt.axhline(y=Vs, color='red', linestyle='--', label=f'Vs={Vs}')  # 加上紅色橫線

# plt.xlabel('max_iter')
# plt.ylabel('Counter / max_iter')
# plt.title(f'Counter/max_iter vs max_iter (Vref={Vref})')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('counter_vs_max_iter_vcm.png', dpi=300)
# plt.show()