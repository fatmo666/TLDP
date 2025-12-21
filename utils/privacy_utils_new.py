import torch
import numpy as np
from numpy.array_api import int16
import random
from main import device
from torch.utils.data import SubsetRandomSampler


def add_laplace_noise(tensor, scale):
    noise = torch.distributions.laplace.Laplace(0, scale).sample(tensor.shape).to(tensor.device)
    return tensor + noise

def add_gaussian_noise(tensor, mean=0.0, std=1.0):
    noise = torch.distributions.Normal(mean, std).sample(tensor.shape).to(tensor.device)
    return tensor + noise.to(tensor.device)

def mvg_mechanism(X, epsilon, delta, theta):
    def compute_H(r, n):
        return sum(1 / (i ** r) for i in range(1, n + 1))

    # 记录原始形状
    original_shape = X.shape

    # 三维转二维
    reshaped_X = X.view(-1, X.size(1))

    m = reshaped_X.shape[0]
    n = reshaped_X.shape[1]

    r = min(m, n)
    gamma = np.sqrt(m * n)
    s_2_f = 2 * np.sqrt(m * n)
    zeta = 2 * np.sqrt((-m * n * np.log(delta))) - 2 * np.log(delta) + m * n

    Hr = compute_H(1, r)
    Hr_half = compute_H(0.5, r)

    alpha = (Hr + Hr_half) * gamma ** 2 + 2 * Hr * gamma * s_2_f
    beta = 2 * (m * n)**0.25 * Hr * s_2_f * zeta

    P = (-beta + np.sqrt(beta ** 2 + 8 * alpha * epsilon))**4 / (16 * alpha**4 * n)

    variances = np.zeros(m)
    for i in range(m):
        p_i = theta[i] * P
        sigma_i = 1 / (np.sqrt(p_i) + 1e-30)
        variances[i] = sigma_i

    u, s, vh = np.linalg.svd(reshaped_X.cpu().numpy())
    Lambda_Sigma = np.diag(variances)

    N = np.random.randn(m, n)
    Z = np.dot(u, np.dot(np.sqrt(Lambda_Sigma), N))

    noise_X = reshaped_X + torch.tensor(Z).to(device)

    # 二维转三维
    # final_X = noise_X.view(X.size(0), X.size(1), -1)
    final_X = noise_X.view(*original_shape)
    return final_X.float()

def mvg_mechanism_2d(X, epsilon, delta, theta):
    def compute_H(r, n):
        return sum(1 / (i ** r) for i in range(1, n + 1))

    # X = X.unsqueeze(0)
    reshaped_X = X.T

    m = reshaped_X.shape[0]
    n = 1

    r = min(m, n)
    gamma = np.sqrt(m * n)
    s_2_f = 2 * np.sqrt(m * n)
    zeta = 2 * np.sqrt((-m * n * np.log(delta))) - 2 * np.log(delta) + m * n

    Hr = compute_H(1, r)
    Hr_half = compute_H(0.5, r)

    alpha = (Hr + Hr_half) * gamma ** 2 + 2 * Hr * gamma * s_2_f
    beta = 2 * (m * n)**0.25 * Hr * s_2_f * zeta

    P = (-beta + np.sqrt(beta ** 2 + 8 * alpha * epsilon))**4 / (16 * alpha**4 * n)

    variances = np.zeros(m)
    for i in range(m):
        p_i = theta[i] * P
        sigma_i = 1 / (np.sqrt(p_i) + 1e-30)
        variances[i] = sigma_i

    # 必须二维
    u, s, vh = np.linalg.svd(reshaped_X.cpu().numpy())
    Lambda_Sigma = np.diag(variances)

    N = np.random.randn(m, n)
    Z = np.dot(u, np.dot(np.sqrt(Lambda_Sigma), N))

    noise_X = reshaped_X + torch.tensor(Z).to(device)

    # 二维转三维
    final_X = noise_X.T
    return final_X.float().flatten()

def perturb_tensor_krr(tensor, epsilon, d=10):
    device = tensor.device

    precision = 1.0 / d

    perturbed_tensor = torch.round(tensor.clone() / precision) * precision

    replace_values = torch.tensor(np.arange(0.0, 1.0, precision), device=device, dtype=tensor.dtype)

    # 定义一个小的容差
    epsilon_tolerance = 1e-6
    # 使用容差处理，将接近1.0的值变为0.9
    perturbed_tensor[(perturbed_tensor >= 1.0 - epsilon_tolerance) & (perturbed_tensor <= 1.0 + epsilon_tolerance)] = 0.9

    p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)

    # 生成随机数张量
    rnd = torch.rand(tensor.shape, device=device)

    # 生成掩码
    mask = rnd > p

    for i in range(len(replace_values)):
        # value_mask = perturbed_tensor == replace_values[i]
        value_mask = (perturbed_tensor >= replace_values[i] - epsilon_tolerance) & (perturbed_tensor <= replace_values[i] + epsilon_tolerance)
        combined_mask = value_mask & mask
        if value_mask.any():
            available_values = replace_values[replace_values != replace_values[i]]
            replace_indices = torch.randint(0, len(available_values), (combined_mask.sum().item(),), device=device)
            perturbed_tensor[combined_mask] = available_values[replace_indices]

    return perturbed_tensor


def perturb_tensor_krr_new(tensor, epsilon, d=20):
    device = tensor.device

    epsilon = epsilon / torch.numel(tensor)

    # 计算精度
    precision = 2.0 / d  # 将精度范围调整为 -1 到 1 的跨度

    # 四舍五入并近似张量值
    perturbed_tensor = torch.round(tensor.clone() / precision) * precision

    # 生成替换值，范围为 -1.0 到 1.0
    replace_values = torch.tensor(np.arange(-1.0, 1.0 + precision, precision), device=device, dtype=tensor.dtype)

    # 定义一个小的容差
    epsilon_tolerance = 1e-6

    # 使用容差处理，将接近1.0的值变为0.9
    perturbed_tensor[(perturbed_tensor >= 1.0 - epsilon_tolerance) & (perturbed_tensor <= 1.0 + epsilon_tolerance)] = 0.9

    # 将接近 -1.0 的值调整为 -0.9
    perturbed_tensor[(perturbed_tensor <= -1.0 + epsilon_tolerance) & (perturbed_tensor >= -1.0 - epsilon_tolerance)] = -0.9

    # 计算扰动概率
    p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)

    # 生成随机数张量
    rnd = torch.rand(tensor.shape, device=device)

    # 生成掩码
    mask = rnd > p
    # print("krr p: ", p)
    # print("mask: ", mask)

    # # 创建随机替换索引
    # replace_indices = torch.randint(0, len(replace_values) - 1, perturbed_tensor.shape, device=device)
    #
    # # 确保替换值与原值相同
    # current_indices = torch.searchsorted(replace_values, perturbed_tensor, right=True) - 1
    # replace_indices = torch.where(replace_indices >= current_indices, replace_indices + 1, replace_indices)
    #
    # # 应用掩码来选择性替换值
    # perturbed_tensor[mask] = replace_values[replace_indices[mask]]

    for i in range(len(replace_values)):
        # 使用容差范围检查相等性
        value_mask = (perturbed_tensor >= replace_values[i] - epsilon_tolerance) & (perturbed_tensor <= replace_values[i] + epsilon_tolerance)
        combined_mask = value_mask & mask
        if value_mask.any():
            available_values = replace_values[replace_values != replace_values[i]]
            replace_indices = torch.randint(0, len(available_values), (combined_mask.sum().item(),), device=device)
            perturbed_tensor[combined_mask] = available_values[replace_indices]

    # 强制结果符合精度要求
    perturbed_tensor = torch.round(perturbed_tensor / precision) * precision

    return perturbed_tensor

def add_optimized_tensor_noise_Yang(tensor, epsilon, sigma, sensitivity):
    delta2_sigma = np.sqrt(-2 * np.log(sigma) + 2 * np.sqrt(-tensor.numel() * np.log(sigma)) + tensor.numel())

    # alpha = sensitivity**2
    # beta = 2 * np.sqrt(delta2_sigma) * sensitivity
    # B = ((-beta + np.sqrt(beta**2 + 8 * alpha * epsilon))**2) / (4 * alpha**2)

    B = (-delta2_sigma + np.sqrt(delta2_sigma**2 + 2 * epsilon)) / sensitivity**2

    N = torch.normal(0, 1, size=tensor.shape, device=tensor.device)

    Z = N * (1 / B)
    return tensor + Z


def perturb_tensor_adaptive_iterative(tensor, total_epsilon, sensitivity, total_rounds, current_round, gamma, is_first_layer, N = 2):
    # 裁剪在函数外
    epsilon_per_round = total_epsilon / total_rounds
    alpha_factor = 1.0 if is_first_layer else -1.0

    initial_budget_per_layer = epsilon_per_round / N

    budget_update_unit = epsilon_per_round / N

    cumulative_update = current_round * (gamma * alpha_factor * budget_update_unit)

    final_epsilon_for_layer = initial_budget_per_layer + cumulative_update

    if final_epsilon_for_layer <= 1e-9:
        final_epsilon_for_layer = 1e-9

    scale = sensitivity / final_epsilon_for_layer

    noise = torch.distributions.laplace.Laplace(0, scale).sample(tensor.shape).to(tensor.device)

    return tensor + noise

def _unflatten_params_func(flat_params, params):
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view_as(p))
        offset += numel


def _update_hessian_and_threshold_func_targeted(model, loss_fn, inputs, targets, device,
                                                hessian_approx_d=1e-3,):
    param_samples = 5000
    # --- 1. 数据采样 ---
    # 已经在外部实现

    # --- 2. 目标参数选择 ---
    params_to_consider = [p for name, p in model.named_parameters()
                          if p.requires_grad and ('weight' in name or 'bias' in name)]

    if not params_to_consider:
        return torch.tensor(1.0, device=device)

    flat_params = torch.cat([p.view(-1) for p in params_to_consider])
    total_params = len(flat_params)

    if total_params <= param_samples:
        # 如果总参数量小于等于采样数，则全部计算
        sample_indices = range(total_params)
    else:
        # 从所有目标参数索引中随机选择 param_samples 个
        sample_indices = random.sample(range(total_params), param_samples)

    sampled_diag_h_values = []

    # --- 3. 完整计算 ---
    with torch.no_grad():
        outputs = model(inputs)
        loss_center = loss_fn(outputs, targets).mean()

        for i in sample_indices:
            original_val = flat_params[i].item()

            flat_params[i] = original_val + hessian_approx_d
            _unflatten_params_func(flat_params, params_to_consider)
            loss_plus = loss_fn(model(inputs), targets).mean()

            flat_params[i] = original_val - hessian_approx_d
            _unflatten_params_func(flat_params, params_to_consider)
            loss_minus = loss_fn(model(inputs), targets).mean()

            flat_params[i] = original_val

            second_derivative = (loss_plus - 2 * loss_center + loss_minus) / (hessian_approx_d ** 2)
            sampled_diag_h_values.append(torch.abs(second_derivative))
            # <<< 添加调试打印 >>>
            if i == sample_indices[0]:  # 只打印第一个采样参数的信息
                print(f"    - hessian_approx_d (d): {hessian_approx_d}")
                print(f"    - d^2: {hessian_approx_d ** 2}")
                print(f"    - Loss Center: {loss_center.item():.6f}")
                print(f"    - Loss Plus:   {loss_plus.item():.6f}")
                print(f"    - Loss Minus:  {loss_minus.item():.6f}")
                print(f"    - Numerator:   {(loss_plus - 2 * loss_center + loss_minus).item():.8f}")
                print(f"    - Second Derivative (H_ii): {second_derivative.item():.4f}")

    _unflatten_params_func(flat_params, params_to_consider)

    if not sampled_diag_h_values:
        return torch.tensor(1.0, device=device)  # 如果没有采样到，返回默认值

    # --- 4. 根据采样结果估算整体曲率并计算最终阈值 ---
    mean_sampled_curvature = torch.mean(torch.stack(sampled_diag_h_values))
    estimated_total_curvature = mean_sampled_curvature * total_params
    hessian_sum_sqrt = torch.sqrt(estimated_total_curvature)

    return hessian_sum_sqrt if hessian_sum_sqrt >= 1e-6 else torch.tensor(1e-6, device=device)


def _update_hessian_and_threshold_func(model, loss_fn, data_loader, device, hessian_approx_d=1e-4):
    inputs, targets = next(iter(data_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    flat_params = torch.cat([p.view(-1) for p in params])
    diag_h = torch.zeros_like(flat_params)

    with torch.no_grad():
        outputs = model(inputs)
        loss_center = loss_fn(outputs, targets)
        if loss_center.dim() > 0:
            loss_center = loss_center.mean()

        for i in range(len(flat_params)):
            original_val = flat_params[i].item()
            # --- 正向扰动 ---
            flat_params[i] = original_val + hessian_approx_d
            _unflatten_params_func(flat_params, params)
            loss_plus = loss_fn(model(inputs), targets)
            if loss_plus.dim() > 0:
                loss_plus = loss_plus.mean()

            # --- 反向扰动 ---
            flat_params[i] = original_val - hessian_approx_d
            _unflatten_params_func(flat_params, params)
            loss_minus = loss_fn(model(inputs), targets)
            if loss_minus.dim() > 0:
                loss_minus = loss_minus.mean()

            # 恢复参数
            flat_params[i] = original_val

            second_derivative = (loss_plus - 2 * loss_center + loss_minus) / (hessian_approx_d ** 2)
            diag_h[i] = second_derivative

    # 5. 确保模型参数恢复到最初的状态
    _unflatten_params_func(flat_params, params)

    # 6. 计算自适应裁剪阈值 C_adaptive = sqrt(Σ |H_ii|)
    hessian_sum_sqrt = torch.sqrt(torch.sum(torch.abs(diag_h)))

    return hessian_sum_sqrt if hessian_sum_sqrt >= 1e-6 else torch.tensor(1e-6, device=device)


# 示例用法
if __name__ == "__main__":
    # epsilon = 0.1
    epsilon = 0.1
    sensitivity = 2

    import math
    # sensitivity = torch.sqrt(torch.numel(example_tensor))
    sensitivity = math.sqrt(torch.numel(example_tensor))
    print("sensitivity: ", sensitivity)
    optimized_tensor = add_optimized_tensor_noise_Yang(example_tensor, epsilon, 0.00001, sensitivity)
    print("\noptimized_tensor: ")
    print(optimized_tensor)

