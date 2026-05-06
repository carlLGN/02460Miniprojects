import torch


def ELBO(edge_logits, neg_edge_logits, x_hat_logits, target_x, mu, logvar, beta=0.1):
    """Calculate the Evidence Lower Bound (ELBO) loss for a Graph (beta)VAE model."""
    kl = 0

    positive_labels = torch.ones_like(edge_logits)
    negative_labels = torch.zeros_like(neg_edge_logits)

    edge_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.cat([edge_logits, neg_edge_logits], dim=0),
        torch.cat([positive_labels, negative_labels], dim=0)
    )

    target_indices = torch.argmax(target_x, dim=1)
    feature_loss = torch.nn.functional.cross_entropy(x_hat_logits, target_indices)

    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)) #mean so it matches scale of edge/feature loss.
    return edge_loss + feature_loss + beta * kl


def graph_level_ELBO(edge_logits, mask, target_adj, mu, logvar, beta=1.0, pos_weight=1.0):
    """ELBO for the graph-level VAE.

    Reconstructs every upper-triangular entry of each graph's adjacency, so no
    negative sampling is needed.
    """
    masked_logits = edge_logits[mask]
    masked_targets = target_adj[mask]

    weight = torch.where(
        masked_targets > 0.5,
        torch.full_like(masked_logits, pos_weight),
        torch.ones_like(masked_logits),
    )
    edge_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        masked_logits, masked_targets, weight=weight
    )

    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return edge_loss + beta * kl, edge_loss.detach(), kl.detach()