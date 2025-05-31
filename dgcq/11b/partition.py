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

def generate_11bit_binary_voltages(num_samples=100):
    """Generate random 11-bit binary fractional voltages (0.xxxxxxxxxxx)"""
    voltages = []
    binary_strings = []
    
    for _ in range(num_samples):
        # Generate random 11-bit binary string
        binary_str = ''.join(np.random.choice(['0', '1']) for _ in range(11))
        voltage = binary_to_decimal(binary_str)
        voltages.append(voltage)
        binary_strings.append(binary_str)
    
    return np.array(voltages), binary_strings

def dgcq_workflow_3bit(Vs, Vcm, Vref, dgcq_iterations=8):
    """
    DGCQ workflow with exactly 8 iterations for coarse quantization
    Returns the residual voltage for ADC processing
    """
    Vacc = 0
    Counter = 0
    history = []
    
    for i in range(dgcq_iterations):
        # Vacc = (Vacc + Vs) / 2
        Vacc = (Vacc + Vs) / 2
        
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

def adc_quantization_8bit(residual_voltage, adc_bits=8):
    """
    Simulate 8-bit ADC quantization of residual voltage
    """
    # Normalize residual to 0-1 range if needed
    if residual_voltage < 0:
        residual_voltage = 0
    elif residual_voltage > 1:
        residual_voltage = 1
    
    # Quantize to 8 bits
    quantization_levels = 2 ** adc_bits
    quantized_level = round(residual_voltage * (quantization_levels - 1))
    adc_output = quantized_level / (quantization_levels - 1)
    
    return adc_output, quantized_level

def complete_dgcq_adc_system(Vs, Vcm, Vref, dgcq_iterations=8, adc_bits=8):
    """
    Complete DGCQ + ADC quantization system
    DGCQ handles first 8 bits, ADC handles remaining 8 bits
    """
    # DGCQ processing (coarse quantization)
    dgcq_output, residual, counter, dgcq_history = dgcq_workflow_3bit(Vs, Vcm, Vref, dgcq_iterations)
    
    # ADC processing (fine quantization of residual)
    adc_output, adc_level = adc_quantization_8bit(residual, adc_bits)
    
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

def decimal_to_binary_fractional(decimal_val, num_bits=16):
    """Convert decimal to binary fractional string"""
    if decimal_val == 0:
        return "0." + "0" * num_bits
    
    integer_part = int(decimal_val)
    fractional_part = decimal_val - integer_part
    
    binary_str = str(integer_part) + "."
    
    for _ in range(num_bits):
        fractional_part *= 2
        if fractional_part >= 1:
            binary_str += "1"
            fractional_part -= 1
        else:
            binary_str += "0"
    
    return binary_str

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

def plot_quantization_analysis_binary(input_voltages, results, quantization_errors):
    """Plot quantization analysis results with binary representation - each plot saved separately"""
    
    # Set font properties globally for all plots
    plt.rcParams.update({
        'font.size': 18,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16
    })
    
    total_outputs = [r['total_output'] for r in results]
    dgcq_outputs = [r['dgcq_output'] for r in results]
    adc_outputs = [r['adc_output'] for r in results]
    
    # Calculate error bit positions
    error_bit_positions = [calculate_error_bit_position(error) for error in quantization_errors]
    
    # Plot 1: Input vs Total Output (with binary labels)
    plt.figure(figsize=(12, 10))
    plt.scatter(input_voltages, total_outputs, alpha=0.7, s=20)
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)', linewidth=2)
    plt.xlabel('Input Voltage', fontsize=20, fontweight='bold')
    plt.ylabel('Total Output Voltage', fontsize=20, fontweight='bold')
    plt.title('Input vs Total Output', fontsize=22, fontweight='bold')
    plt.legend(fontsize=18)
    plt.grid(True, alpha=0.7)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig('plot1_input_vs_output.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Error Bit Position Distribution
    plt.figure(figsize=(12, 10))
    plt.hist(error_bit_positions, bins=range(1, 17), alpha=0.7, edgecolor='black', linewidth=2)
    plt.xlabel('Error First Appears at Bit Position', fontsize=20, fontweight='bold')
    plt.ylabel('Frequency', fontsize=20, fontweight='bold')
    plt.title('Error Bit Position Distribution', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7)
    plt.xticks(range(1, 16), fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot2_error_bit_position_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Quantization Error vs Bit Position
    plt.figure(figsize=(12, 10))
    plt.scatter(error_bit_positions, quantization_errors, alpha=0.7, s=20, color='red')
    plt.xlabel('Error Bit Position', fontsize=20, fontweight='bold')
    plt.ylabel('Quantization Error (Decimal)', fontsize=20, fontweight='bold')
    plt.title('Error Value vs Bit Position', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7)
    plt.yscale('log')
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig('plot3_error_vs_bit_position.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4: DGCQ Output in Binary
    plt.figure(figsize=(12, 10))
    dgcq_binary_values = [int(output * 256) for output in dgcq_outputs]  # Convert to 8-bit integer
    plt.hist(dgcq_binary_values, bins=30, alpha=0.7, edgecolor='black', color='green', linewidth=2)
    plt.xlabel('DGCQ Output (8-bit Integer)', fontsize=20, fontweight='bold')
    plt.ylabel('Frequency', fontsize=20, fontweight='bold')
    plt.title('DGCQ Output Distribution (Binary)', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig('plot4_dgcq_output_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 5: ADC Output in Binary
    plt.figure(figsize=(12, 10))
    adc_binary_values = [int(output * 255) for output in adc_outputs]  # Convert to 8-bit integer
    plt.hist(adc_binary_values, bins=30, alpha=0.7, edgecolor='black', color='orange', linewidth=2)
    plt.xlabel('ADC Output (8-bit Integer)', fontsize=20, fontweight='bold')
    plt.ylabel('Frequency', fontsize=20, fontweight='bold')
    plt.title('ADC Output Distribution (Binary)', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig('plot5_adc_output_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 6: Error vs Input with Binary Scale
    plt.figure(figsize=(12, 10))
    plt.scatter(input_voltages, error_bit_positions, alpha=0.7, s=20, color='purple')
    plt.xlabel('Input Voltage', fontsize=20, fontweight='bold')
    plt.ylabel('Error Bit Position', fontsize=20, fontweight='bold')
    plt.title('Error Bit Position vs Input', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7)
    plt.ylim(0, 16)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig('plot6_error_bit_position_vs_input.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 7: Binary Error Magnitude
    plt.figure(figsize=(12, 10))
    binary_error_magnitudes = [-np.log2(error) if error > 0 else 16 for error in quantization_errors]
    plt.hist(binary_error_magnitudes, bins=30, alpha=0.7, edgecolor='black', color='cyan', linewidth=2)
    plt.xlabel('Error Magnitude (-log2(error))', fontsize=20, fontweight='bold')
    plt.ylabel('Frequency', fontsize=20, fontweight='bold')
    plt.title('Binary Error Magnitude Distribution', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7)
    plt.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig('plot7_binary_error_magnitude.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 8: System Performance Summary (Modified to 16 bits and percentage)
    plt.figure(figsize=(14, 10))
    total_samples = len(error_bit_positions)
    
    # Calculate accuracy percentage for each bit position (1 to 16)
    bit_accuracy_percentage = []
    bit_positions = range(1, 17)
    
    for i in bit_positions:
        accurate_count = sum(1 for pos in error_bit_positions if pos >= i)
        percentage = (accurate_count / total_samples) * 100
        bit_accuracy_percentage.append(percentage)
    
    plt.plot(bit_positions, bit_accuracy_percentage, 'bo-', linewidth=3, markersize=8)
    plt.xlabel('Bit Position', fontsize=20, fontweight='bold')
    plt.ylabel('Accuracy Percentage (%)', fontsize=20, fontweight='bold')
    plt.title('System Accuracy by Bit Position (Percentage)', fontsize=22, fontweight='bold')
    plt.grid(True, alpha=0.7)
    plt.xticks(range(1, 17), fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('plot8_system_accuracy_percentage.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Combined overview plot (optional)
    plt.figure(figsize=(24, 16))
    
    # Recreate all subplots in one figure for overview
    plt.subplot(2, 4, 1)
    plt.scatter(input_voltages, total_outputs, alpha=0.7, s=20)
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)', linewidth=2)
    plt.xlabel('Input Voltage', fontsize=18, fontweight='bold')
    plt.ylabel('Total Output Voltage', fontsize=18, fontweight='bold')
    plt.title('Input vs Total Output', fontsize=20, fontweight='bold')
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.7)
    plt.tick_params(labelsize=16)
    
    plt.subplot(2, 4, 2)
    plt.hist(error_bit_positions, bins=range(1, 17), alpha=0.7, edgecolor='black', linewidth=2)
    plt.xlabel('Error First Appears at Bit Position', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')
    plt.title('Error Bit Position Distribution', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.7)
    plt.xticks(range(1, 16), fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    
    plt.subplot(2, 4, 3)
    plt.scatter(error_bit_positions, quantization_errors, alpha=0.7, s=20, color='red')
    plt.xlabel('Error Bit Position', fontsize=18, fontweight='bold')
    plt.ylabel('Quantization Error (Decimal)', fontsize=18, fontweight='bold')
    plt.title('Error Value vs Bit Position', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.7)
    plt.yscale('log')
    plt.tick_params(labelsize=16)
    
    plt.subplot(2, 4, 4)
    plt.hist(dgcq_binary_values, bins=30, alpha=0.7, edgecolor='black', color='green', linewidth=2)
    plt.xlabel('DGCQ Output (8-bit Integer)', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')
    plt.title('DGCQ Output Distribution', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.7)
    plt.tick_params(labelsize=16)
    
    plt.subplot(2, 4, 5)
    plt.hist(adc_binary_values, bins=30, alpha=0.7, edgecolor='black', color='orange', linewidth=2)
    plt.xlabel('ADC Output (8-bit Integer)', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')
    plt.title('ADC Output Distribution', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.7)
    plt.tick_params(labelsize=16)
    
    plt.subplot(2, 4, 6)
    plt.scatter(input_voltages, error_bit_positions, alpha=0.7, s=20, color='purple')
    plt.xlabel('Input Voltage', fontsize=18, fontweight='bold')
    plt.ylabel('Error Bit Position', fontsize=18, fontweight='bold')
    plt.title('Error Bit Position vs Input', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.7)
    plt.ylim(0, 16)
    plt.tick_params(labelsize=16)
    
    plt.subplot(2, 4, 7)
    plt.hist(binary_error_magnitudes, bins=30, alpha=0.7, edgecolor='black', color='cyan', linewidth=2)
    plt.xlabel('Error Magnitude (-log2(error))', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=18, fontweight='bold')
    plt.title('Binary Error Magnitude', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.7)
    plt.tick_params(labelsize=16)
    
    plt.subplot(2, 4, 8)
    plt.plot(bit_positions, bit_accuracy_percentage, 'bo-', linewidth=3, markersize=6)
    plt.xlabel('Bit Position', fontsize=18, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=18, fontweight='bold')
    plt.title('System Accuracy by Bit Position', fontsize=20, fontweight='bold')
    plt.grid(True, alpha=0.7)
    plt.xticks(range(1, 17, 2), fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('plot_overview_all_combined.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main program with flat structure (unchanged)
if __name__ == "__main__":
    # Parameter settings
    Vcm = 0.5
    Vref = 1.0
    num_samples = 10000
    dgcq_iterations = 8
    adc_bits = 8
    
    print("=== 11-bit Binary Voltage Quantization Analysis ===")
    print(f"DGCQ: {dgcq_iterations} iterations (coarse quantization)")
    print(f"ADC: {adc_bits} bits (fine quantization)")
    print(f"Vcm = {Vcm}, Vref = {Vref}")
    print()
    
    # Step 1: Generate 11-bit binary voltages
    print("Step 1: Generating random 11-bit binary voltages...")
    input_voltages, binary_strings = generate_11bit_binary_voltages(num_samples)
    print(f"Generated {len(input_voltages)} voltage samples")
    
    # Step 2: Process each voltage through DGCQ + ADC system
    print("\nStep 2: Processing voltages through DGCQ + ADC system...")
    results = []
    quantization_errors = []
    
    for i, (Vs, binary_str) in enumerate(zip(input_voltages, binary_strings)):
        # Process through DGCQ + ADC system
        result = complete_dgcq_adc_system(Vs, Vcm, Vref, dgcq_iterations, adc_bits)
        
        # Calculate quantization error
        error = abs(Vs - result['total_output'])
        quantization_errors.append(error)
        results.append(result)
        
        # Print first few examples
        if i < 5:
            print(f"Sample {i+1}:")
            print(f"  Input binary: 0.{binary_str}")
            print(f"  Input voltage: {Vs:.6f}")
            print(f"  DGCQ output: {result['dgcq_output']:.6f} (Counter: {result['counter']})")
            print(f"  ADC output: {result['adc_output']:.6f} (Level: {result['adc_level']})")
            print(f"  Total output: {result['total_output']:.6f}")
            print(f"  Error: {error:.6f}")
            print()
    
    # Step 3: Calculate error bit positions
    print("Step 3: Calculating error bit positions...")
    error_bit_positions = []
    for error in quantization_errors:
        bit_pos = calculate_error_bit_position(error)
        error_bit_positions.append(bit_pos)
    
    # Step 4: Generate statistical summary
    print("\nStep 4: Statistical Analysis")
    print("=" * 50)
    print(f"Total samples: {len(input_voltages)}")
    print(f"Average quantization error: {np.mean(quantization_errors):.8f}")
    print(f"Maximum quantization error: {np.max(quantization_errors):.8f}")
    print(f"Minimum quantization error: {np.min(quantization_errors):.8f}")
    print(f"RMS error: {np.sqrt(np.mean(np.array(quantization_errors)**2)):.8f}")
    print(f"Error standard deviation: {np.std(quantization_errors):.8f}")
    print(f"Average error bit position: {np.mean(error_bit_positions):.2f}")
    print(f"Best accuracy (highest bit position): {max(error_bit_positions)}")
    print(f"Worst accuracy (lowest bit position): {min(error_bit_positions)}")
    
    # Accuracy statistics by bit position (extended to 16 bits)
    print("\nAccuracy by bit position (percentage):")
    total_samples = len(error_bit_positions)
    for bit_pos in range(1, 17):
        accurate_count = sum(1 for pos in error_bit_positions if pos >= bit_pos)
        percentage = (accurate_count / total_samples) * 100
        print(f"  {bit_pos:2d} bits: {accurate_count:3d} samples ({percentage:5.1f}%)")
    
    # Step 5: Binary representation examples
    print("\nStep 5: Binary Representation Examples")
    print("=" * 50)
    for i in range(min(5, len(input_voltages))):
        input_binary = decimal_to_binary_fractional(input_voltages[i], 11)
        output_binary = decimal_to_binary_fractional(results[i]['total_output'], 11)
        error_pos = error_bit_positions[i]
        
        print(f"Sample {i+1}:")
        print(f"  Input:  {input_binary}")
        print(f"  Output: {output_binary}")
        print(f"  Error first appears at bit position: {error_pos}")
        print(f"  Error magnitude: {quantization_errors[i]:.8f}")
        print()
    
    # Step 6: Theoretical comparison
    print("Step 6: Theoretical Analysis")
    print("=" * 50)
    theoretical_error = 1 / (2 ** 11) / 2
    theoretical_bit_position = calculate_error_bit_position(theoretical_error)
    print(f"Theoretical 11-bit system:")
    print(f"  Quantization error: {theoretical_error:.8f}")
    print(f"  Expected error bit position: {theoretical_bit_position}")
    
    # Step 7: Plot results
    print("\nStep 7: Generating plots...")
    plot_quantization_analysis_binary(input_voltages, results, quantization_errors)
    print("Individual analysis plots saved:")
    print("  - plot1_input_vs_output.png")
    print("  - plot2_error_bit_position_distribution.png")
    print("  - plot3_error_vs_bit_position.png")
    print("  - plot4_dgcq_output_distribution.png")
    print("  - plot5_adc_output_distribution.png")
    print("  - plot6_error_bit_position_vs_input.png")
    print("  - plot7_binary_error_magnitude.png")
    print("  - plot8_system_accuracy_percentage.png")
    print("  - plot_overview_all_combined.png")
    
    # Step 8: Test specific binary values
    print("\nStep 8: Testing specific binary values...")
    test_cases = [
        "10000000000",  # 0.5
        "01000000000",  # 0.25
        "11000000000",  # 0.75
        "00100000000",  # 0.125
        "10100000000",  # 0.625
    ]
    
    print("Specific binary value tests:")
    for binary_str in test_cases:
        input_voltage = binary_to_decimal(binary_str)
        result = complete_dgcq_adc_system(input_voltage, Vcm, Vref, dgcq_iterations, adc_bits)
        error = abs(input_voltage - result['total_output'])
        error_bit_pos = calculate_error_bit_position(error)
        
        input_binary_full = "0." + binary_str
        output_binary = decimal_to_binary_fractional(result['total_output'], 11)
        
        print(f"Input Binary:  {input_binary_full}")
        print(f"Output Binary: {output_binary}")
        print(f"DGCQ Counter: {result['counter']} (Binary: {format(result['counter'], '08b')})")
        print(f"ADC Level: {result['adc_level']} (Binary: {format(result['adc_level'], '08b')})")
        print(f"Error: {error:.8f}")
        print(f"Error bit position: {error_bit_pos}")
        print("-" * 60)
    
    print("\nAnalysis complete!")