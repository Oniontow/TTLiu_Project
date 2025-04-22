import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

# ----------------------------
# 自定義 ResNet Block
# ----------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # self.quant = torch.ao.quantization.QuantStub()  # 移除
        # self.dequant = torch.ao.quantization.DeQuantStub()  # 移除
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

        # 判斷是否為量化張量
        if out.is_quantized:
            out = torch.ops.quantized.add(out, identity, scale=out.q_scale(), zero_point=out.q_zero_point())
        else:
            out = out + identity
        out = self.relu(out)
        return out
# ----------------------------
# 自定義 ResNet-18 架構
# ----------------------------
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                              padding=1, bias=False)
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

def train(model, device, train_loader, optimizer, criterion, epoch):
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
    return accuracy
    
def add_noise_to_quantized_model(quantized_model, noise_mean=-40, noise_std=10):
    """
    對量化模型的捲積層權重添加高斯噪聲
    
    參數:
        quantized_model: 已量化的模型
        noise_mean: 高斯噪聲的平均值
        noise_std: 高斯噪聲的標準差
    
    返回:
        添加噪聲後的量化模型
    """
    print(f"\n--- Adding Gaussian noise (mean={noise_mean}, std={noise_std}) to quantized model ---")
    
    # 確保噪聲足夠大，能夠在量化後仍然產生變化
    noise_std = max(noise_std, 20)  # 增加噪聲標準差確保可見性
    
    # 遍歷模型中的量化捲積層
    for name, module in quantized_model.named_modules():
        if isinstance(module, torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d) or \
           isinstance(module, torch.nn.quantized.modules.conv.Conv2d):
            
            if hasattr(module, '_packed_params'):
                # 獲取原始量化權重、比例和零點
                weight_int, bias = module._packed_params.unpack()
                scale = module.scale
                zero_point = module.zero_point
                
                # 記錄原始int8權重用於後續對比
                orig_weight_int = weight_int.int_repr().clone()
                
                # 將int8權重轉換為浮點數 (解量化)
                weight_float = ((weight_int.int_repr().float() - zero_point) * scale)
                
                # 加入平均值為noise_mean、標準差為noise_std的高斯噪聲
                # 將噪聲放大，確保量化後仍然看得到變化
                noise = torch.randn_like(weight_float) * noise_std + noise_mean
                # 這裡做一個特別處理，確保噪聲夠大
                noise = noise * (scale)  # 放大噪聲
                noisy_weight_float = weight_float + noise
                
                # 直接修改INT8表示，而不是通過量化
                # 這確保我們能夠看到變化
                noisy_int_repr = torch.round(noisy_weight_float / scale + zero_point).clamp(-128, 127).to(torch.int8)
                
                # 創建新的量化張量
                noisy_weight_q = torch._make_per_tensor_quantized_tensor(
                    noisy_int_repr, scale, zero_point)
                
                # 創建新的打包參數
                module._packed_params = torch.ops.quantized.conv2d_prepack(
                    noisy_weight_q,
                    bias,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups
                )
                
                # 檢查權重是否真的被修改了
                changed_weights = (orig_weight_int != noisy_int_repr).sum().item()
                total_weights = orig_weight_int.numel()
                print(f"Added noise to layer: {name} - Changed {changed_weights}/{total_weights} weights ({changed_weights/total_weights*100:.2f}%)")
    
    return quantized_model

def compare_weight_distributions(original_model, noisy_model, save_path='weight_distribution_comparison.png'):
    """
    分析並比較添加噪聲前後的權重分佈
    """
    print("\n--- Comparing Weight Distributions Before and After Noise ---")
    
    # 儲存統計數據
    layer_changes = {}
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    plot_idx = 1
    
    # 計算層數來準備子圖
    conv_layers = 0
    for name, module in original_model.named_modules():
        if (isinstance(module, torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d) or 
            isinstance(module, torch.nn.quantized.modules.conv.Conv2d)) and hasattr(module, '_packed_params'):
            conv_layers += 1
    
    # 比較每層的權重
    all_diffs = []
    
    for (orig_name, orig_module), (noisy_name, noisy_module) in zip(
            original_model.named_modules(), noisy_model.named_modules()):
        
        if (isinstance(orig_module, torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d) or 
            isinstance(orig_module, torch.nn.quantized.modules.conv.Conv2d)) and hasattr(orig_module, '_packed_params'):
            
            # 提取權重
            orig_weights = orig_module._packed_params.unpack()[0].int_repr().detach().cpu().numpy()
            noisy_weights = noisy_module._packed_params.unpack()[0].int_repr().detach().cpu().numpy()
            
            # 計算統計數據
            orig_mean = np.mean(orig_weights)
            noisy_mean = np.mean(noisy_weights)
            mean_diff = abs(orig_mean - noisy_mean)
            
            orig_std = np.std(orig_weights)
            noisy_std = np.std(noisy_weights)
            std_diff = abs(orig_std - noisy_std)
            
            # 計算權重變化百分比
            abs_diff = np.abs(orig_weights - noisy_weights)
            mean_abs_diff = np.mean(abs_diff)
            max_abs_diff = np.max(abs_diff)
            all_diffs.extend(abs_diff.flatten())
            
            # 儲存統計數據
            layer_changes[orig_name] = {
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'mean_abs_diff': mean_abs_diff,
                'max_abs_diff': max_abs_diff,
                'changed_percentage': np.mean(abs_diff > 0) * 100  # 變化權重的百分比
            }
            
            # 繪製權重分佈
            plt.subplot(conv_layers, 1, plot_idx)
            
            plt.hist(orig_weights.flatten(), bins=50, alpha=0.5, label='Original')
            plt.hist(noisy_weights.flatten(), bins=50, alpha=0.5, label='Noisy')
            plt.title(f"{orig_name}")
            plt.legend()
            
            plot_idx += 1
            
            print(f"Layer: {orig_name}")
            print(f"  Mean difference: {mean_diff:.2f}")
            print(f"  Std difference: {std_diff:.2f}")
            print(f"  Avg absolute diff: {mean_abs_diff:.2f}")
            print(f"  Max absolute diff: {max_abs_diff}")
            print(f"  % of weights changed: {layer_changes[orig_name]['changed_percentage']:.2f}%")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Weight comparison saved to '{save_path}'")
    
    # 繪製整體權重變化直方圖
    plt.figure(figsize=(10, 6))
    plt.hist(all_diffs, bins=50)
    plt.title("Distribution of Weight Changes Across All Layers")
    plt.xlabel("Absolute Weight Change")
    plt.ylabel("Frequency")
    plt.savefig('overall_weight_changes.png')
    
    return layer_changes

def estimate_noise_impact(original_model, noisy_model):
    """
    估計噪聲對模型性能的理論影響
    """
    print("\n--- Estimating Theoretical Impact of Noise ---")
    
    total_weights = 0
    changed_weights = 0
    
    for (orig_name, orig_module), (noisy_name, noisy_module) in zip(
            original_model.named_modules(), noisy_model.named_modules()):
        
        if (isinstance(orig_module, torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d) or 
            isinstance(orig_module, torch.nn.quantized.modules.conv.Conv2d)) and hasattr(orig_module, '_packed_params'):
            
            # 提取權重
            orig_weights = orig_module._packed_params.unpack()[0].int_repr().detach().cpu().numpy()
            noisy_weights = noisy_module._packed_params.unpack()[0].int_repr().detach().cpu().numpy()
            
            # 計算變化量
            weight_diff = np.abs(orig_weights - noisy_weights)
            
            # 更新計數
            layer_size = orig_weights.size
            total_weights += layer_size
            changed_weights += np.sum(weight_diff > 0)
    
    # 計算變化的比例
    change_ratio = changed_weights / total_weights
    
    # 估計準確率影響（基於經驗法則，可根據實際情況調整）
    estimated_acc_drop = change_ratio * 100
    
    print(f"Total weights in model: {total_weights}")
    print(f"Changed weights due to noise: {changed_weights} ({change_ratio:.4f})")
    print(f"Estimated accuracy drop: ~{estimated_acc_drop:.2f}%")
    
    severity = "低" if estimated_acc_drop < 5 else "中" if estimated_acc_drop < 15 else "高"
    print(f"噪聲嚴重性評估: {severity}")
    
    return {
        'total_weights': total_weights,
        'changed_weights': changed_weights,
        'change_ratio': change_ratio,
        'estimated_acc_drop': estimated_acc_drop,
        'severity': severity
    }
    
def simple_static_quantization(model, data_loader, num_samples=100):
    import copy
    import torch.quantization

    float_model = copy.deepcopy(model).cpu().eval()
    float_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    prepared_model = torch.quantization.prepare(float_model, inplace=False)

    # 用部分驗證資料做校準
    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            prepared_model(inputs)
            if i >= num_samples:
                break

    quantized_model = torch.quantization.convert(prepared_model, inplace=False)

    # 收集所有量化層的權重
    quantized_weights = {}
    for name, module in quantized_model.named_modules():
        if isinstance(module, torch.nn.quantized.Conv2d):
            quantized_weights[name] = module.weight().int_repr().cpu().numpy()

    return float_model, quantized_model, quantized_weights

def analyze_noise_impact(original_model, noisy_model, data_loader, num_classes=10):
    original_model.eval()
    noisy_model.eval()
    agreement = 0
    total = 0
    confidence_change = []
    activation_diffs = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            # 確保輸入在 CPU 並為 float
            inputs = inputs.cpu().float()
            outputs_orig = original_model(inputs)
            outputs_noisy = noisy_model(inputs)
            probs_orig = F.softmax(outputs_orig, dim=1)
            probs_noisy = F.softmax(outputs_noisy, dim=1)
            pred_orig = probs_orig.argmax(dim=1)
            pred_noisy = probs_noisy.argmax(dim=1)
            agreement += (pred_orig == pred_noisy).sum().item()
            total += inputs.size(0)
            confidence_change.extend((probs_orig.max(dim=1).values - probs_noisy.max(dim=1).values).cpu().numpy())

    agreement_pct = 100 * agreement / total if total > 0 else 0
    avg_conf_change = float(np.mean(np.abs(confidence_change))) if confidence_change else 0.0

    activation_diffs = [{"layer": "N/A", "diff": 0.0}]

    return {
        "agreement": agreement_pct,
        "confidence_change": avg_conf_change,
        "activation_diffs": activation_diffs
    }

def add_simple_noise_to_quantized_model(quantized_model, noise_mean=-40, noise_std=10):
    """
    更簡單的向量化模型添加噪聲的方法，適合在Kaggle環境中運行
    """
    print(f"\n--- Adding Gaussian noise (mean={noise_mean}, std={noise_std}) to quantized model ---")
    
    # 創建深拷貝以避免修改原始模型
    noisy_model = copy.deepcopy(quantized_model)
    
    # 用於記錄統計信息
    layer_stats = {}
    
    # 遍歷模型中的量化層
    for name, module in noisy_model.named_modules():
        if isinstance(module, torch.nn.quantized.modules.conv.Conv2d) or \
           isinstance(module, torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d):
            
            if hasattr(module, '_packed_params'):
                # 解包權重和偏置
                weight, bias = module._packed_params.unpack()
                scale = module.scale
                zero_point = module.zero_point
                
                # 獲取int8權重
                int_weight = weight.int_repr()
                
                # 將int8權重轉換為浮點（解量化）
                float_weight = (int_weight.float() - zero_point) * scale
                
                # 加入噪聲
                # 我們將噪聲值乘以scale來確保噪聲在同一尺度上
                noise_scale_factor = scale 
                noise = torch.randn_like(float_weight) * noise_std * noise_scale_factor + noise_mean * noise_scale_factor
                noisy_float_weight = float_weight + noise
                
                # 重新量化到int8
                noisy_int_weight = torch.round(noisy_float_weight / scale + zero_point)
                noisy_int_weight = torch.clamp(noisy_int_weight, -128, 127).to(torch.int8)
                
                # 創建新的量化張量
                noisy_weight = torch._make_per_tensor_quantized_tensor(noisy_int_weight, scale, zero_point)
                
                # 使用新權重重新打包參數
                module._packed_params = torch.ops.quantized.conv2d_prepack(
                    noisy_weight,
                    bias,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups
                )
                
                # 記錄變化統計信息
                changed = (int_weight != noisy_int_weight).sum().item()
                total = int_weight.numel()
                
                layer_stats[name] = {
                    'total_weights': total,
                    'changed_weights': changed,
                    'change_percentage': (changed / total) * 100,
                    'original_mean': int_weight.float().mean().item(),
                    'noisy_mean': noisy_int_weight.float().mean().item(),
                    'original_std': int_weight.float().std().item(),
                    'noisy_std': noisy_int_weight.float().std().item()
                }
                
                # print(f"Layer {name}: Changed {changed}/{total} weights ({layer_stats[name]['change_percentage']:.2f}%)")
    
    return noisy_model, layer_stats

def add_mac_noise_hook_to_quantized_model(model, noise_mean=10, noise_std=5):
    """
    在量化模型的每個卷積層的forward輸出加上高斯噪聲（模擬MAC誤差）
    噪聲分布：mean=10, std=5（以int8尺度），且為-x軸單邊（全為負值）
    """
    def mac_noise_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if output.is_quantized:
                output_fp = output.dequantize()
                scale = output.q_scale()
                # 以 int8 尺度產生 noise，再乘以 scale
                noise = -torch.abs(torch.randn_like(output_fp) * noise_std + noise_mean) * scale
                noisy_fp = output_fp + noise
                noisy_q = torch.quantize_per_tensor(
                    noisy_fp, scale=scale, zero_point=output.q_zero_point(), dtype=output.dtype
                )
                return noisy_q
            else:
                # 若是 float tensor，直接加 noise
                noise = -torch.abs(torch.randn_like(output) * noise_std + noise_mean)
                return output + noise
        return output

    for module in model.modules():
        if isinstance(module, torch.nn.quantized.Conv2d) or \
           isinstance(module, torch.nn.intrinsic.quantized.ConvReLU2d):
            module.register_forward_hook(mac_noise_hook)
    return model


# ----------------------------
# 資料處理
# ----------------------------
# transform = transforms.Compose([
#     transforms.Lambda(custom_transform)
# ])
transform = transforms.Compose([
    transforms.ToTensor()
])

train_val_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False,
                                download=True, transform=transform)

# 分出 validation set（訓練資料的 10%）
train_len = int(0.9 * len(train_val_dataset))
val_len = len(train_val_dataset) - train_len
train_dataset, val_dataset = random_split(train_val_dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ----------------------------
# 主程式邏輯
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 訓練模型（此處僅示範3個epoch）
for epoch in range(1, 4):
    train(model, device, train_loader, optimizer, criterion, epoch)
    evaluate(model, device, val_loader, label="Validation")

# 模型量化與統計分析
import matplotlib.pyplot as plt
import torch.quantization

# 1. 準備量化感知訓練（QAT）
print("\n--- Preparing for Quantization Aware Training (QAT) ---")
# 將模型移至CPU並設為訓練模式
model = model.cpu().train()

# 設定QAT配置
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

# 為模型準備QAT
model_prepared = torch.quantization.prepare_qat(model)

# 2. 模擬訓練過程中的量化效果
print("--- Simulating quantization effects ---")
# 切換到評估模式
model_prepared.eval()

with torch.no_grad():
    for i, (data, _) in enumerate(val_loader):
        model_prepared(data)
        if i >= 5:  # 僅運行幾個批次
            break

# 3. 轉換為量化模型
print("--- Converting to quantized model ---")
model_prepared.cpu()
quantized_model = torch.quantization.convert(model_prepared)

# 主程式部分

import seaborn as sns
import copy

# 使用靜態量化方法量化模型
float_model, quantized_model, quantized_weights = simple_static_quantization(
    model, 
    val_loader,
    num_samples=100
)
quantized_model = quantized_model.cpu()
print("--- Quantized Model Accuracy ---")
evaluate(quantized_model, torch.device("cpu"), test_loader, label="Quantized Model (No Noise)")

# noisy_model, noise_stats = add_simple_noise_to_quantized_model(
#     quantized_model,
#     noise_mean=-6,
#     noise_std=1
# )
# noisy_model = noisy_model.cpu()
print("--- Converting to quantized model ---")

quantized_model_with_mac_noise = copy.deepcopy(quantized_model)
quantized_model_with_mac_noise = add_mac_noise_hook_to_quantized_model(
    quantized_model_with_mac_noise, noise_mean=5, noise_std=1
)

print("\n--- Quantized Model Accuracy (With Noise) ---")
evaluate(quantized_model_with_mac_noise, torch.device("cpu"), test_loader, label="Quantized Model (With Noise)")