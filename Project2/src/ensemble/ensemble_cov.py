import torch
import torch.nn as nn
import torch.distributions as td
import os

from Project2.src.geodesics.curve_energy import curve_energy


def compute_ensemble_cov(args, model, mnist_test_loader, device, experiment_path, fixed_pairs):
    """
    Computes the Average CoV for Geodesic and Euclidean distances 
    across multiple training reruns for a specific ensemble size.
    """
    all_geodesic_distances = torch.zeros(len(fixed_pairs), args.num_reruns)
    all_euclidean_distances = torch.zeros(len(fixed_pairs), args.num_reruns)

    for run_idx in range(args.num_reruns):
        # Load the model state for the specific rerun
        run_folder = os.path.join(experiment_path, f"run_{run_idx}")
        model_path = os.path.join(run_folder, "model.pt")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        for pair_idx, (idx_i, idx_j) in enumerate(fixed_pairs):
            # 1. Prepare data
            img_i = mnist_test_loader.dataset[idx_i][0].to(device).unsqueeze(0)
            img_j = mnist_test_loader.dataset[idx_j][0].to(device).unsqueeze(0)

            with torch.no_grad():
                z_i = model.encoder(img_i).mean.squeeze(0)
                z_j = model.encoder(img_j).mean.squeeze(0)

            # 2. Calculate Euclidean distance
            all_euclidean_distances[pair_idx, run_idx] = torch.norm(z_i - z_j)

            # 3. Optimize path for Geodesic distance
            t = torch.linspace(0, 1, args.num_t).to(device).view(-1, 1)
            path = (1 - t) * z_i + t * z_j
            inner_points = path[1:-1].detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([inner_points], lr=1e-2)

            for _ in range(100):
                optimizer.zero_grad()
                full_path = torch.cat([z_i.unsqueeze(0), inner_points, z_j.unsqueeze(0)], dim=0)
                energy = curve_energy(full_path, model.decoders)
                energy.backward()
                optimizer.step()

            final_path = torch.cat([z_i.unsqueeze(0), inner_points, z_j.unsqueeze(0)], dim=0).detach()

            # 4. Monte Carlo estimate of Geodesic distance
            with torch.no_grad():
                dist = torch.tensor(0.0, device=device)
                n_samples = 10

                for i in range(final_path.shape[0] - 1):
                    z_curr = final_path[i].unsqueeze(0)
                    z_next = final_path[i+1].unsqueeze(0)
                    segment_dist = 0.0

                    for _ in range(n_samples):
                        l = torch.randint(0, len(model.decoders), (1,)).item()
                        k = torch.randint(0, len(model.decoders), (1,)).item()
                        f_l = model.decoders[l](z_curr).mean
                        f_k = model.decoders[k](z_next).mean
                        segment_dist += torch.norm(f_l - f_k)

                    dist += segment_dist / n_samples

                all_geodesic_distances[pair_idx, run_idx] = dist
                
        print(f"Finished evaluating run {run_idx} for {args.num_decoders} decoders")

    # 5. Final CoV Calculation
    # We calculate the CoV across runs for each pair, then average those CoVs
    geo_cov = (all_geodesic_distances.std(dim=1) / all_geodesic_distances.mean(dim=1)).mean()
    euc_cov = (all_euclidean_distances.std(dim=1) / all_euclidean_distances.mean(dim=1)).mean()

    print(f"\nFinal Results for {args.num_decoders} Decoders:")
    print(f"Average Geodesic CoV: {geo_cov.item():.4f}")
    print(f"Average Euclidean CoV: {euc_cov.item():.4f}")

    return geo_cov, euc_cov