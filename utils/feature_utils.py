import torch

def extract_features(model, device, data_loader):
    model.eval()
    features = []
    targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            features.append(output.cpu())
            targets.append(target)
    features = torch.cat(features)
    targets = torch.cat(targets)
    return features, targets
