import matplotlib.pyplot as plt
import numpy as np

def binary_to_decimal(binary_str):
    """Convert binary string to decimal value"""
    decimal = 0
    for i, bit in enumerate(binary_str):
        if bit == '1':
            decimal += 2 ** (-(i+1))
    return decimal

def decimal_to_binary(decimal_val, num_bits):
    """Convert decimal to binary string with specified number of bits"""
    binary_str = ""
    for i in range(num_bits):
        decimal_val *= 2
        if decimal_val >= 1:
            binary_str += "1"
            decimal_val -= 1
        else:
            binary_str += "0"
    return binary_str


def dgcq_workflow_3bit_with_loss(Vs, Vcm, Vref, dgcq_iterations=8, dgcq_loss=0.0):
    """
    DGCQ workflow with exactly 8 iterations for coarse quantization
    Includes DGCQ loss model
    Returns the residual voltage for ADC processing
    """
    Vacc = 0
    Counter = 0
    history = []
    
    for i in range(dgcq_iterations):
        # Vacc = (Vacc + Vs) / 2
        Vacc = (Vacc + Vs) / 2
        
        # Apply DGCQ loss (affects Vacc accumulation)
        Vacc = Vacc * (1 - dgcq_loss) + np.random.normal(-dgcq_loss * 0.1, dgcq_loss * 0.1)
        
        if Vacc > Vcm:
            # Y branch: Vota = 2*Vacc - Vref, Counter += 1
            Vota = 2 * Vacc - Vref
            Vacc = Vota
            Counter += 1
        else:
            # N branch: Vota = 2*Vacc
            Vota = 2 * Vacc
            Vacc = Vota
            
        history.append({'iteration': i+1, 'Vacc': Vacc, 'Vota': Vota, 'Counter': Counter})
    
    # DGCQ coarse quantization (first 8 bits)
    dgcq_output = Counter / 8  # Normalize to 0-1 range
    
    # Residual voltage for ADC processing
    residual = Vacc  # This will be quantized by ADC for remaining 8 bits
    
    return dgcq_output, residual, Counter, history

def adc_quantization_8bit_with_loss(residual_voltage, adc_bits=8, adc_loss=0.0):
    """
    Simulate 8-bit ADC quantization of residual voltage with loss model
    """
    # Apply ADC loss (affects input voltage to ADC)
    residual_with_loss = residual_voltage * (1 - adc_loss) + np.random.normal(-adc_loss * 0.1, adc_loss * 0.1)
    
    # Normalize residual to 0-1 range if needed
    if residual_with_loss < 0:
        residual_with_loss = 0
    elif residual_with_loss > 1:
        residual_with_loss = 1
    
    # Quantize to 8 bits
    quantization_levels = 2 ** adc_bits
    quantized_level = round(residual_with_loss * (quantization_levels - 1))
    adc_output = quantized_level / (quantization_levels - 1)
    
    return adc_output, quantized_level

def complete_dgcq_adc_system_with_loss(Vs, Vcm, Vref, dgcq_iterations=8, adc_bits=8, dgcq_loss=0.0, adc_loss=0.0):
    """
    Complete DGCQ + ADC quantization system with loss models
    DGCQ handles first 8 bits, ADC handles remaining 8 bits
    """
    # DGCQ processing (coarse quantization) with loss
    dgcq_output, residual, counter, dgcq_history = dgcq_workflow_3bit_with_loss(Vs, Vcm, Vref, dgcq_iterations, dgcq_loss)
    
    # ADC processing (fine quantization of residual) with loss
    adc_output, adc_level = adc_quantization_8bit_with_loss(residual, adc_bits, adc_loss)
    
    # Combine DGCQ and ADC outputs
    # DGCQ provides coarse bits, ADC provides fine bits
    total_output = dgcq_output + adc_output / 8
    
    return {
        'total_output': total_output,
        'dgcq_output': dgcq_output,
        'adc_output': adc_output,
        'residual': residual,
        'counter': counter,
        'dgcq_history': dgcq_history,
        'adc_level': adc_level
    }

def calculate_error_bit_position(error):
    """Calculate at which bit position the error first appears"""
    if error == 0:
        return 16  # No error
    
    # Find the position of the first significant bit in the error
    bit_position = 1
    threshold = 0.5
    
    while threshold > error and bit_position <= 16:
        threshold /= 2
        bit_position += 1
    
    return bit_position - 1

def experiment4_quantization_method_comparison(input_voltages, Vcm, Vref, adc_loss=0.005, dgcq_loss=0.005):
    """
    Experiment 4: Compare three quantization methods:
    1. Pure ADC (11-bit ADC quantization)
    2. Pure DGCQ (2^11 = 2048 iterations)
    3. Hybrid DGCQ+ADC (8 iterations + 8-bit ADC)
    """
    print(f"\n=== Experiment 4: Quantization Method Comparison ===")
    print(f"ADC Loss: {adc_loss}, DGCQ Loss: {dgcq_loss}")
    
    # Method 1: Pure ADC (11-bit)
    print("Method 1: Pure ADC (11-bit) quantization...")
    pure_adc_results = []
    pure_adc_errors = []
    
    for Vs in input_voltages:
        adc_output, _ = adc_quantization_8bit_with_loss(Vs, adc_bits=11, adc_loss=adc_loss)
        error = abs(Vs - adc_output)
        pure_adc_results.append(adc_output)
        pure_adc_errors.append(error)
    
    pure_adc_bit_positions = [calculate_error_bit_position(error) for error in pure_adc_errors]
    
    # Method 2: Pure DGCQ (2048 iterations for 11-bit precision)
    print("Method 2: Pure DGCQ (2048 iterations) quantization...")
    pure_dgcq_results = []
    pure_dgcq_errors = []
    
    for Vs in input_voltages:
        # Use 2048 iterations to achieve 11-bit precision
        dgcq_output, residual, counter, _ = dgcq_workflow_3bit_with_loss(
            Vs, Vcm, Vref, dgcq_iterations=2048, dgcq_loss=dgcq_loss
        )
        # For pure DGCQ, the output is the normalized counter value
        final_output = counter / 2048  # Normalize to 0-1 range
        error = abs(Vs - final_output)
        pure_dgcq_results.append(final_output)
        pure_dgcq_errors.append(error)
    
    pure_dgcq_bit_positions = [calculate_error_bit_position(error) for error in pure_dgcq_errors]
    
    # Method 3: Hybrid DGCQ+ADC (8 iterations + 8-bit ADC) - current method
    print("Method 3: Hybrid DGCQ+ADC (8 iterations + 8-bit ADC)...")
    hybrid_results = []
    hybrid_errors = []
    
    for Vs in input_voltages:
        result = complete_dgcq_adc_system_with_loss(
            Vs, Vcm, Vref, dgcq_iterations=8, adc_bits=8, 
            dgcq_loss=dgcq_loss, adc_loss=adc_loss
        )
        error = abs(Vs - result['total_output'])
        hybrid_results.append(result['total_output'])
        hybrid_errors.append(error)
    
    hybrid_bit_positions = [calculate_error_bit_position(error) for error in hybrid_errors]
    
    # Calculate statistics
    methods_stats = {
        'Pure ADC (11-bit)': {
            'bit_positions': pure_adc_bit_positions,
            'errors': pure_adc_errors,
            'outputs': pure_adc_results,
            'avg_bit_precision': np.mean(pure_adc_bit_positions),
            'std_bit_precision': np.std(pure_adc_bit_positions),
            'avg_error': np.mean(pure_adc_errors),
            'max_error': np.max(pure_adc_errors)
        },
        'Pure DGCQ (2048 iter)': {
            'bit_positions': pure_dgcq_bit_positions,
            'errors': pure_dgcq_errors,
            'outputs': pure_dgcq_results,
            'avg_bit_precision': np.mean(pure_dgcq_bit_positions),
            'std_bit_precision': np.std(pure_dgcq_bit_positions),
            'avg_error': np.mean(pure_dgcq_errors),
            'max_error': np.max(pure_dgcq_errors)
        },
        'Hybrid DGCQ+ADC': {
            'bit_positions': hybrid_bit_positions,
            'errors': hybrid_errors,
            'outputs': hybrid_results,
            'avg_bit_precision': np.mean(hybrid_bit_positions),
            'std_bit_precision': np.std(hybrid_bit_positions),
            'avg_error': np.mean(hybrid_errors),
            'max_error': np.max(hybrid_errors)
        }
    }
    
    # Print results
    print("\nResults Summary:")
    print("-" * 80)
    for method, stats in methods_stats.items():
        print(f"{method}:")
        print(f"  Average Bit Precision: {stats['avg_bit_precision']:.3f} ± {stats['std_bit_precision']:.3f}")
        print(f"  Average Error: {stats['avg_error']:.6f}")
        print(f"  Maximum Error: {stats['max_error']:.6f}")
        print()
    
    # Create 4 main comparison plots only
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Bit Precision Comparison (Bar Chart)
    plt.subplot(2, 2, 1)
    methods = list(methods_stats.keys())
    avg_precisions = [methods_stats[m]['avg_bit_precision'] for m in methods]
    std_precisions = [methods_stats[m]['std_bit_precision'] for m in methods]
    
    bars = plt.bar(range(len(methods)), avg_precisions, yerr=std_precisions, 
                   capsize=5, alpha=0.7, color=['blue', 'red', 'green'])
    plt.xlabel('Quantization Method')
    plt.ylabel('Average Bit Precision')
    plt.title('Bit Precision Comparison')
    plt.xticks(range(len(methods)), [m.replace(' ', '\n') for m in methods], rotation=0)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (avg, std) in enumerate(zip(avg_precisions, std_precisions)):
        plt.text(i, avg + std + 0.1, f'{avg:.2f}', ha='center', va='bottom')
    
    # Plot 2: Error Distribution Comparison
    plt.subplot(2, 2, 2)
    colors = ['blue', 'red', 'green']
    for i, (method, stats) in enumerate(methods_stats.items()):
        plt.hist(stats['errors'], bins=30, alpha=0.6, label=method, 
                density=True, histtype='step', linewidth=2, color=colors[i])
    plt.xlabel('Quantization Error')
    plt.ylabel('Density')
    plt.title('Error Distribution Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Input vs Output for each method
    plt.subplot(2, 2, 3)
    for i, (method, stats) in enumerate(methods_stats.items()):
        plt.scatter(input_voltages[::10], stats['outputs'][::10], 
                   alpha=0.5, s=10, color=colors[i], label=method)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Ideal')
    plt.xlabel('Input Voltage')
    plt.ylabel('Output Voltage')
    plt.title('Input vs Output Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Method efficiency (bits per computational unit)
    plt.subplot(2, 2, 4)
    # Assume computational cost: ADC=1 unit per bit, DGCQ=1 unit per iteration
    computational_costs = [11, 2048, 8+8]  # ADC bits, DGCQ iterations, hybrid
    efficiency = [avg_precisions[i]/computational_costs[i] for i in range(len(methods))]
    
    bars = plt.bar(range(len(methods)), efficiency, alpha=0.7, 
                   color=['blue', 'red', 'green'])
    plt.xlabel('Method')
    plt.ylabel('Bits per Computational Unit')
    plt.title('Computational Efficiency')
    plt.xticks(range(len(methods)), [m.replace(' ', '\n') for m in methods])
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, val in enumerate(efficiency):
        plt.text(i, val + 0.001, f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('experiment4_quantization_method_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return methods_stats

# Main program with flat structure (unchanged)
if __name__ == "__main__":
    # Parameter settings
    Vcm = 0.5
    Vref = 1.0
    num_samples = 100  # Reduced for faster loss analysis
    dgcq_iterations = 8
    adc_bits = 8
    
    test_voltages = np.random.uniform(0, 1, num_samples)
    
    # NEW EXPERIMENT 4: Quantization Method Comparison
    print("\n" + "="*60)
    print("EXPERIMENT 4: QUANTIZATION METHOD COMPARISON")
    print("="*60)
    
    # Use same loss values from previous experiments
    comparison_adc_loss = 0.005  # From experiment 1
    comparison_dgcq_loss = 0.002  # From experiment 2
    
    methods_comparison = experiment4_quantization_method_comparison(
        test_voltages, Vcm, Vref, 
        adc_loss=comparison_adc_loss, 
        dgcq_loss=comparison_dgcq_loss
    )
    
    print("\nQuantization method comparison complete!")
    print("Generated plots:")
    print("  - experiment4_quantization_method_comparison.png")
    print("  - experiment4_summary_comparison.png")
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL ANALYSIS SUMMARY")
    print("="*60)
    
    print("Best performing method:")
    best_method = max(methods_comparison.keys(), 
                     key=lambda x: methods_comparison[x]['avg_bit_precision'])
    print(f"  {best_method}: {methods_comparison[best_method]['avg_bit_precision']:.3f} bits")
    
    print("\nMost efficient method (bits per computational unit):")
    computational_costs = {'Pure ADC (11-bit)': 11, 'Pure DGCQ (2048 iter)': 2048, 'Hybrid DGCQ+ADC': 16}
    efficiency_ranking = {method: methods_comparison[method]['avg_bit_precision']/computational_costs[method] 
                         for method in methods_comparison.keys()}
    most_efficient = max(efficiency_ranking.keys(), key=lambda x: efficiency_ranking[x])
    print(f"  {most_efficient}: {efficiency_ranking[most_efficient]:.4f} bits/unit")
    
    print("\nAll experiments completed successfully!")