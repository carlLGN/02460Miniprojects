import torch

def SUM_AGGREGATE(x, edge_index):
    """Sum aggregation function"""
    r, c = edge_index
    neighbor_features = x[c]

    out = torch.zeros_like(x)

    return out.index_add_(0, r, neighbor_features)

def MEAN_AGGREGATE(x, edge_index):
    """Mean aggregation function"""
    r, c = edge_index
    neighbor_features = x[c]

    out = torch.zeros_like(x)
    counts = torch.zeros(x.size(0), 1).to(x.device)
    out.index_add_(0, r, neighbor_features)
    counts.index_add_(0, r, torch.ones(neighbor_features.size(0), 1).to(x.device))

    return out / counts.clamp(min=1)

def ADD_UPDATE(x, message):
    """Update function"""
    return x + message

def CONCAT_UPDATE(x, message):
    """Update function"""
    return torch.cat([x, message], dim=-1)