import matplotlib.pyplot as plt
import numpy as np
import random

def bits_to_int(bits):
    value = int("".join(map(str, bits)), 2)
    if bits[0] == 1:
        value -= (1 << len(bits))
    return value

def unsigned_bits_to_int(bits):
    if bits != []:
        value = int("".join(map(str, bits)), 2)
    else:
        value = 0
    return value

def int_to_bits(val, width):
    val = int(val)
    if val < 0:
        val = (1 << width) + val
    return [int(x) for x in f"{val:0{width}b}"]

def sign_extend(bits, target_len):
    sign_bit = bits[0]
    return [sign_bit] * (target_len - len(bits)) + bits

def tolist(inputt):
    return [inputt] if not isinstance(inputt, list) else inputt

def visualize_multiplication_aligned(A_bits, B_bits, decrease_rate=0.0, Analogtype=""):
    A_val = bits_to_int(A_bits)

    partials = []
    max_len = 0
    analog_val = 0
    for i in range(8):
        analogpart = 0
        b = B_bits[7 - i]
        if i < 4:
            part = int_to_bits(A_val if b == 1 else 0, 8)
            if b == 1 and decrease_rate > 0:
                if(Analogtype == "Lightning"):
                    analogpart = unsigned_bits_to_int(tolist(part[2:]))
                    part[2:] = [0 for _ in range(6)]
                elif(Analogtype == "BigTriangle"):
                    analogpart = unsigned_bits_to_int(tolist(part[i+1:]))
                    part[i+1:] = [0 for _ in range(7 - i)]
                elif(Analogtype == "SmallTriangle"):
                    analogpart = unsigned_bits_to_int(tolist(part[i+3:]))
                    part[i+3:] = [0 for _ in range(5 - i)]
        else:
            part = int_to_bits(A_val if b == 1 else 0, 8)
            if b == 1 and decrease_rate > 0:
                if(Analogtype == "Lightning"):
                    analogpart = unsigned_bits_to_int(tolist(part[6:]))
                    part[6:] = [0 for _ in range(2)]
                elif(Analogtype == "BigTriangle"):
                    analogpart = unsigned_bits_to_int(tolist(part[i+1:]))
                    part[i+1:] = [0 for _ in range(7 - i)]
                elif(Analogtype == "SmallTriangle"):
                    analogpart = unsigned_bits_to_int(tolist(part[i+3:]))
                    part[i+3:] = [0 for _ in range(5 - i)]
        analog_shifted = analogpart * (2 ** i)
        shifted = part + [0] * i
        partials.append(shifted)
        analog_val += analog_shifted
        max_len = max(max_len, len(shifted))

    partials_extended = [sign_extend(p, max_len) for p in partials]

    total = [0] * max_len
    carry = 0
    for i in reversed(range(max_len)):
        s = sum(p[i] for p in partials_extended) + carry
        total[i] = s % 2
        carry = s // 2
    digital_val = bits_to_int(total)
    analog_val = analog_val * (1 - decrease_rate) if Analogtype != "" else analog_val
    final_val = digital_val + analog_val
    total = int_to_bits(final_val, 15)
    return total, final_val

def simulate_and_collect(decrease_rate, iteration=10000, Analogtype=""):
    decimal_errors = []
    mult_result_sizes = []
    bit_error_list = [0 for _ in range(15)]

    for _ in range(iteration):
        A = [int(random.random() > 0.5) for _ in range(8)]
        B = [int(random.random() > 0.5) for _ in range(8)]
        B[0] = 0  # 保證 B 為正

        bit_result, mult_result = visualize_multiplication_aligned(A, B, decrease_rate=0.0, Analogtype="")
        noisy_bit_result, noisy_mult_result = visualize_multiplication_aligned(A, B, decrease_rate=decrease_rate, Analogtype=Analogtype)

        for i in range(len(bit_result)):
            if bit_result[i] != noisy_bit_result[i]:
                bit_error_list[i] += 1

        if mult_result != 0:
            error_rate = abs(mult_result - noisy_mult_result) / (2 ** 15) * 100
        elif noisy_mult_result != 0:
            error_rate = abs(mult_result - noisy_mult_result) / (2 ** 15) * 100
        else:
            error_rate = 0

        decimal_errors.append(error_rate)
        mult_result_sizes.append(abs(mult_result))

    bit_error_rate = [bit_error_count * 100 / iteration for bit_error_count in bit_error_list]
    return mult_result_sizes, decimal_errors, bit_error_rate

def plot_all_results(results_dict, num_bins=16, Analogtype=""):
    max_value = 2 ** 14
    bin_edges = np.linspace(0, max_value, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(16, 6))
    for rate, (sizes, errors, _) in results_dict.items():
        bin_error_data = [[] for _ in range(num_bins)]
        for size, error in zip(sizes, errors):
            for i in range(num_bins):
                if bin_edges[i] <= size < bin_edges[i + 1]:
                    bin_error_data[i].append(error)
                    break
        bin_avg_error = [np.mean(e) if e else 0 for e in bin_error_data]
        # bin_stdev_error = [np.std(e) if e else 0 for e in bin_error_data]
        # print(f"Decrease Rate: {rate:.2f}, Bin Avg Error: {bin_avg_error}, Bin Std Dev Error: {bin_stdev_error}")
        # plt.errorbar(bin_centers, bin_avg_error, yerr=bin_stdev_error, fmt='o', label=f"{rate:.2f}", capsize=5)
        plt.plot(bin_centers, bin_avg_error, label=f"{rate:.2f}", marker='o')
    
    plt.xlabel("Mult Result Absolute Value")
    plt.ylabel("Average Decimal Error Rate")
    plt.title(f"<{Analogtype}> Decimal Error Rate vs Different Decrease Rates")
    plt.legend(title="Decrease Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{Analogtype}DecimalError.png")
    plt.show()

    # 畫 bit error rate 比較圖
    plt.figure(figsize=(10, 5))
    for rate, (_, _, bit_errors) in results_dict.items():
        plt.plot(list(reversed(range(15))), bit_errors, label=f"{rate:.2f}", marker='s')
    plt.xlabel("Bit Index (0=LSB)")
    plt.ylabel("Bit Error Rate (%)")
    plt.title(f"<{Analogtype}> Bit Error Distribution")
    plt.gca().invert_xaxis()
    plt.legend(title="Decrease Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{Analogtype}BitError.png")
    plt.show()

# 主程式：跑多組 decrease_rate
results_by_rate = {}
for rate in np.arange(0.0, 0.201, 0.04):
    print(f"Simulating decrease_rate = {rate:.2f} ...")
    sizes, errors, bit_errors = simulate_and_collect(rate, Analogtype="Lightning")
    results_by_rate[rate] = (sizes, errors, bit_errors)

plot_all_results(results_by_rate, Analogtype="Lightning")

results_by_rate = {}
for rate in np.arange(0.0, 0.201, 0.04):
    print(f"Simulating decrease_rate = {rate:.2f} ...")
    sizes, errors, bit_errors = simulate_and_collect(rate, Analogtype="BigTriangle")
    results_by_rate[rate] = (sizes, errors, bit_errors)

plot_all_results(results_by_rate, Analogtype="BigTriangle")

results_by_rate = {}
for rate in np.arange(0.0, 0.201, 0.04):
    print(f"Simulating decrease_rate = {rate:.2f} ...")
    sizes, errors, bit_errors = simulate_and_collect(rate, Analogtype="SmallTriangle")
    results_by_rate[rate] = (sizes, errors, bit_errors)

plot_all_results(results_by_rate, Analogtype="SmallTriangle")
