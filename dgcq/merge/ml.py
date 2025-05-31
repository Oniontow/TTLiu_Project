import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

# ===========================================
# Model Architecture
# ===========================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        # Handle quantized tensors differently
        if out.is_quantized:
            out = torch.ops.quantized.add(out, identity, scale=out.q_scale(), zero_point=out.q_zero_point())
        else:
            out = out + identity
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

# ===========================================
# Training and Evaluation Functions
# ===========================================
def train(model, device, train_loader, optimizer, criterion, epoch):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += outputs.argmax(dim=1).eq(labels).sum().item()
    
    print(f"Epoch {epoch} - Train Loss: {total_loss:.4f}, Accuracy: {correct / len(train_loader.dataset):.4f}")

def evaluate(model, device, loader, label="Validation"):
    """Evaluate model on the given data loader"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
    accuracy = correct / len(loader.dataset)
    print(f"{label} Accuracy: {accuracy:.4f}")
    return accuracy  # 返回準確度值

def plot_accuracy_comparison(accuracies, model_names):
    """繪製不同模型準確度的長條圖比較（正規化到Fine-Grained=1）"""
    plt.figure(figsize=(12, 7))
    bars = plt.bar(model_names, accuracies, color=['royalblue', 'tomato', 'forestgreen'])
    
    # 在每個長條上方顯示準確度數值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=12)
    
    plt.title('Comparison of Coarse/Fine-Grained Results (Normalized to Fine-Grained=1)', fontsize=16)
    plt.ylabel('Normalized Accuracy', fontsize=14)
    plt.ylim(0, 1.1)  # 調整y軸範圍
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=10, fontsize=12)
    plt.tight_layout()
    plt.savefig('model_accuracy_comparison_normalized.png')
    plt.show()

# ===========================================
# Quantization Functions
# ===========================================
def simple_static_quantization(model, data_loader, num_samples=100):
    """Apply static quantization to the model"""
    float_model = copy.deepcopy(model).cpu().eval()
    
    # 修改量化配置
    qconfig = torch.ao.quantization.QConfig(
        activation=torch.ao.quantization.observer.MinMaxObserver.with_args(
            qscheme=torch.per_tensor_affine,  # 改為非對稱量化
            quant_min=0,                      # 對應 quint8 的下限
            quant_max=255,                    # 對應 quint8 的上限
            dtype=torch.quint8                # 啟動值使用 quint8
        ),
        weight=torch.ao.quantization.observer.MinMaxObserver.with_args(
            qscheme=torch.per_tensor_symmetric,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8  # 權重使用 qint8
        )
    )
    float_model.qconfig = qconfig
    
    # 其餘程式碼保持不變
    prepared_model = torch.quantization.prepare(float_model, inplace=False)
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            prepared_model(inputs)
            if i >= num_samples:
                break

    quantized_model = torch.quantization.convert(prepared_model, inplace=False)
    
    # 收集量化權重
    quantized_weights = {}
    for name, module in quantized_model.named_modules():
        if isinstance(module, torch.nn.quantized.Conv2d):
            quantized_weights[name] = module.weight().int_repr().cpu().numpy()

    return float_model, quantized_model, quantized_weights

# ===========================================
# Noise Simulation Functions
# ===========================================
def add_mac_noise_hook_to_quantized_model(model, noise_mean=10, noise_std=5):
    """
    Add forward hooks to simulate MAC errors in quantized model.
    Adds negative Gaussian noise to convolution outputs.
    """
    def mac_noise_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if output.is_quantized:
                output_fp = output.dequantize()
                scale = output.q_scale()
                # Generate noise in int8 scale, then multiply by scale
                noise = -torch.abs(torch.randn_like(output_fp) * noise_std + noise_mean) * scale
                noisy_fp = output_fp + noise
                noisy_q = torch.quantize_per_tensor(
                    noisy_fp, scale=scale, zero_point=output.q_zero_point(), dtype=output.dtype
                )
                return noisy_q
            else:
                # For float tensors, add noise directly
                noise = -torch.abs(torch.randn_like(output) * noise_std + noise_mean)
                return output + noise
        return output
    
    def improved_mac_noise_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if output.is_quantized:
                # 1. 反量化到浮點數域
                output_fp = output.dequantize()
                scale = output.q_scale()
                zp = output.q_zero_point()
                
                # 2. 模擬MAC運算結果的位寬擴展 (從8bit到16/32bit)
                # 使用縮放因子模擬更高的位元精度
                precision_factor = 256.0  # 模擬8bit到16bit的擴展
                high_precision_out = output_fp * precision_factor
                
                # 3. 在高精度的MAC結果上加入雜訊
                # 注意：這裡的雜訊大小應該與MAC結果的範圍相匹配
                noise = -torch.abs(torch.randn_like(high_precision_out) * noise_std + noise_mean)
                noisy_high_precision = high_precision_out + noise
                
                # 4. 將結果縮放回原始範圍並重新量化
                noisy_fp = noisy_high_precision / precision_factor
                noisy_q = torch.quantize_per_tensor(
                    noisy_fp, scale=scale, zero_point=zp, dtype=output.dtype
                )
                return noisy_q
            else:
                # 對於浮點張量的處理
                precision_factor = 256.0
                high_precision_out = output * precision_factor
                noise = -torch.abs(torch.randn_like(high_precision_out) * noise_std + noise_mean)
                return (high_precision_out + noise) / precision_factor
        return output

    for module in model.modules():
        if isinstance(module, torch.nn.quantized.Conv2d) or \
           isinstance(module, torch.nn.intrinsic.quantized.ConvReLU2d):
            module.register_forward_hook(improved_mac_noise_hook)
    return model

# ===========================================
# Data Preparation
# ===========================================
def prepare_data(batch_size=128):
    """Prepare CIFAR-10 datasets and dataloaders"""
    transform = transforms.Compose([transforms.ToTensor()])

    train_val_dataset = datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transform)

    # Split training/validation (90%/10%)
    train_len = int(0.9 * len(train_val_dataset))
    val_len = len(train_val_dataset) - train_len
    train_dataset, val_dataset = random_split(train_val_dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# ===========================================
# Main Program
# ===========================================
def main():
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(batch_size=128)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model, optimizer and loss function
    model = ResNet18().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model (3 epochs for demonstration)
    print("=== Training Model ===")
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, criterion, epoch)
        evaluate(model, device, val_loader, label="Validation")
    
    # Quantize the model
    print("\n=== Quantizing Model ===")
    float_model, quantized_model, quantized_weights = simple_static_quantization(
        model, val_loader, num_samples=100
    )

        accuracies = []
    model_names = ['Fine-Grained', 'Coarse-Grained', 'MIX']
    
    # 不再 append Quantized Model (No Noise)
    quantized_model = quantized_model.cpu()
    print("\n=== Evaluating Quantized Model ===")
    evaluate(quantized_model, torch.device("cpu"), test_loader, label="Quantized Model (No Noise)")

    print("\n=== Adding MAC Noise to Quantized Model ===")
    quantized_model_bigtriangle_noise = copy.deepcopy(quantized_model)
    quantized_model_bigtriangle_noise = add_mac_noise_hook_to_quantized_model(
        quantized_model_bigtriangle_noise, noise_mean=26.5, noise_std=1
    )
    print("\n=== Evaluating Ideal Model ===")
    acc = evaluate(quantized_model_bigtriangle_noise, torch.device("cpu"), test_loader, label="Fine-Grained")
    accuracies.append(acc)
    
    print("\n=== Adding MAC Noise to Fine-Grained Model ===")
    quantized_model_lightning_noise = copy.deepcopy(quantized_model)
    quantized_model_lightning_noise = add_mac_noise_hook_to_quantized_model(
        quantized_model_lightning_noise, noise_mean=34.1, noise_std=1
    )
    print("\n=== Evaluating Fine-Grained Model ===")
    acc = evaluate(quantized_model_lightning_noise, torch.device("cpu"), test_loader, label="Coarse-Grained")
    accuracies.append(acc)
    
    print("\n=== Adding MAC Noise to Quantized Model ===")
    quantized_model_smalltriangle_noise = copy.deepcopy(quantized_model)
    quantized_model_smalltriangle_noise = add_mac_noise_hook_to_quantized_model(
        quantized_model_smalltriangle_noise, noise_mean=24.42, noise_std=1
    )
    print("\n=== Evaluating Coarse-Grained Model ===")
    acc = evaluate(quantized_model_smalltriangle_noise, torch.device("cpu"), test_loader, label="MIX")
    accuracies.append(acc)

    # 正規化 accuracies 到 Fine-Grained = 1
    fine_grained_acc = accuracies[0]  # Fine-Grained 是第一個
    normalized_accuracies = [acc / fine_grained_acc for acc in accuracies]
    
    print(f"\n=== Original Accuracies ===")
    for name, acc in zip(model_names, accuracies):
        print(f"{name}: {acc:.4f}")
    
    print(f"\n=== Normalized Accuracies (Fine-Grained = 1.0) ===")
    for name, norm_acc in zip(model_names, normalized_accuracies):
        print(f"{name}: {norm_acc:.4f}")

    plot_accuracy_comparison(normalized_accuracies, model_names)
    
if __name__ == "__main__":
    main()