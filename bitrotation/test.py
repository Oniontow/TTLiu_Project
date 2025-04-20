import matplotlib.pyplot as plt
import numpy as np
import random

def bits_to_int(bits):
    value = int("".join(map(str, bits)), 2)
    if bits[0] == 1:
        value -= (1 << len(bits))
    return value

def int_to_bits(val, width):
    if val < 0:
        val = (1 << width) + val
    return [int(x) for x in f"{val:0{width}b}"]

def sign_extend(bits, target_len):
    sign_bit = bits[0]
    return [sign_bit] * (target_len - len(bits)) + bits

def visualize_multiplication_aligned(A_bits, B_bits, decrease_rate=0.0):
    A_val = bits_to_int(A_bits)

    partials = []
    max_len = 0
    for i in range(8):
        b = B_bits[7 - i]
        if i < 4:
            part = int_to_bits(A_val if b == 1 else 0, 8)
            if b == 1 and decrease_rate > 0:
                for j in range(2, 8):
                    part[j] = int(random.random() > decrease_rate) * part[j]
        else:
            part = int_to_bits(A_val if b == 1 else 0, 8)
            if b == 1 and decrease_rate > 0:
                for j in range(6, 8):
                    part[j] = int(random.random() > decrease_rate) * part[j]
        shifted = part + [0] * i
        partials.append(shifted)
        max_len = max(max_len, len(shifted))

    partials_extended = [sign_extend(p, max_len) for p in partials]

    total = [0] * max_len
    carry = 0
    for i in reversed(range(max_len)):
        s = sum(p[i] for p in partials_extended) + carry
        total[i] = s % 2
        carry = s // 2

    final_val = bits_to_int(total)
    return total, final_val

def simulate_and_collect(decrease_rate, iteration=10000):
    decimal_errors = []
    mult_result_sizes = []
    bit_error_list = [0 for _ in range(15)]

    for _ in range(iteration):
        A = [int(random.random() > 0.5) for _ in range(8)]
        B = [int(random.random() > 0.5) for _ in range(8)]
        B[0] = 0  # 保證 B 為正

        bit_result, mult_result = visualize_multiplication_aligned(A, B, decrease_rate=0.0)
        noisy_bit_result, noisy_mult_result = visualize_multiplication_aligned(A, B, decrease_rate=decrease_rate)

        for i in range(len(bit_result)):
            if bit_result[i] != noisy_bit_result[i]:
                bit_error_list[i] += 1

        if mult_result != 0:
            error_rate = abs(mult_result - noisy_mult_result) * 100 / abs(mult_result)
        elif noisy_mult_result != 0:
            error_rate = abs(mult_result - noisy_mult_result) * 100 / abs(noisy_mult_result)
        else:
            error_rate = 0

        decimal_errors.append(error_rate)
        mult_result_sizes.append(abs(mult_result))

    bit_error_rate = [bit_error_count * 100 / iteration for bit_error_count in bit_error_list]
    return mult_result_sizes, decimal_errors, bit_error_rate

def plot_all_results(results_dict, num_bins=32):
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
        plt.plot(bin_centers, bin_avg_error, label=f"{rate:.1f}", marker='o')

    plt.xlabel("Mult Result Absolute Value")
    plt.ylabel("Average Decimal Error Rate (%)")
    plt.title("Decimal Error Rate vs Mult Result for Different Decrease Rates")
    plt.legend(title="Decrease Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("all_decimal_error_rates.png")
    plt.show()

    # 畫 bit error rate 比較圖
    plt.figure(figsize=(10, 5))
    for rate, (_, _, bit_errors) in results_dict.items():
        plt.plot(range(15), bit_errors, label=f"{rate:.1f}", marker='s')
    plt.xlabel("Bit Index (0=MSB)")
    plt.ylabel("Bit Error Rate (%)")
    plt.title("Bit Error Rate Distribution Across Bit Positions")
    plt.gca().invert_xaxis()
    plt.legend(title="Decrease Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("all_bit_error_rates.png")
    plt.show()

# 主程式：跑多組 decrease_rate
results_by_rate = {}
for rate in np.arange(0.0, 1.01, 0.2):
    print(f"Simulating decrease_rate = {rate:.1f} ...")
    sizes, errors, bit_errors = simulate_and_collect(rate)
    results_by_rate[rate] = (sizes, errors, bit_errors)

plot_all_results(results_by_rate)
