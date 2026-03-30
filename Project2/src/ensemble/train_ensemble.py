import torch
import torch.nn as nn
import os

from Project2.src.ensemble.ensemble_vae import *


from Project2.src.geodesics.curve_energy import curve_energy
from Project2.src.vae import *
from Project2.src.helpers import *
from Project2.src.train_vae import train



if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.mode == "train":
        base_folder = os.path.join(script_dir, f"{args.experiment_folder}_decoders_{args.num_decoders}")
        os.makedirs(base_folder, exist_ok=True)

        for run in range(args.num_reruns):
            print(f"\n--- Starting Retraining {run + 1}/{args.num_reruns} ---")
            
            # Create a specific folder for this run
            run_folder = os.path.join(base_folder, f"run_{run}")
            os.makedirs(run_folder, exist_ok=True)

            # 1. Initialize fresh components for every retraining
            decoder_list = nn.ModuleList([
                GaussianDecoder(new_decoder(M)) for _ in range(args.num_decoders)
            ])

            model = EnsembleVAE(
                GaussianPrior(M),
                decoder_list,
                GaussianEncoder(new_encoder(M)),
            ).to(device)

            # 2. Fresh optimizer for the new model
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            # 3. Train the model from scratch
            train(
                model,
                optimizer,
                mnist_train_loader,
                args.epochs_per_decoder,
                args.device,
            )

            # 4. Save this specific run's state
            torch.save(
                model.state_dict(),
                os.path.join(run_folder, "model.pt"),
            )
            print(f"Model saved to {run_folder}/model.pt")

    elif args.mode == "sample":

        decoder_list = nn.ModuleList([
            GaussianDecoder(new_decoder(M)) for _ in range(args.num_decoders)
        ])

        model = EnsembleVAE(
            GaussianPrior(M),
            decoder_list,
            GaussianEncoder(new_encoder(M)),
        ).to(device)

        model.load_state_dict(torch.load(f"{args.experiment_folder}/model.pt", map_location=device))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            z = model.encoder(data).mean
            
            recon = model.decoders[0](z).mean 
            save_image(torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png")

    elif args.mode == "eval":
        decoder_list = nn.ModuleList([
            GaussianDecoder(new_decoder(M)) for _ in range(args.num_decoders)
        ])

        model = EnsembleVAE(
            GaussianPrior(M),
            decoder_list,
            GaussianEncoder(new_encoder(M)),
        ).to(device)

        model.load_state_dict(torch.load(f"{args.experiment_folder}/model.pt", map_location=device))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":
        decoder_list = nn.ModuleList([
            GaussianDecoder(new_decoder(M)) for _ in range(args.num_decoders)
        ])

        model = EnsembleVAE(
            GaussianPrior(M),
            decoder_list,
            GaussianEncoder(new_encoder(M)),
        ).to(device)

        # fixed pairs of images to evaluate geodesic and euclidean distances on
        # We will evaluate on the same 10 pairs across all reruns to ensure consistency in our comparisons
    
        fixed_pairs = [(0, 1), (10, 20), (50, 60), (100, 110), (5, 15), 
               (30, 40), (70, 80), (90, 100), (12, 22), (45, 55)]
        
        all_geodesic_distances = torch.zeros(len(fixed_pairs), args.num_reruns)
        all_euclidean_distances = torch.zeros(len(fixed_pairs), args.num_reruns)

        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        folder_name = f"{args.experiment_folder}_decoders_{args.num_decoders}"
        base_folder = os.path.join(current_script_dir, folder_name)

        for run_idx in range(args.num_reruns):
            # Construct the path to the specific run
            model_path = os.path.join(base_folder, f"run_{run_idx}", "model.pt")

            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

             # For each run, calculate distance for all 10 fixed pairs
            for pair_idx, (idx_i, idx_j) in enumerate(fixed_pairs):
                # Get data for these specific indices
                img_i = mnist_test_loader.dataset[idx_i][0].to(device).unsqueeze(0)
                img_j = mnist_test_loader.dataset[idx_j][0].to(device).unsqueeze(0)

                with torch.no_grad():
                    z_i = model.encoder(img_i).mean.squeeze(0)
                    z_j = model.encoder(img_j).mean.squeeze(0)

                #calculate Euclidean distance in latent space
                all_euclidean_distances[pair_idx, run_idx] = torch.norm(z_i - z_j)

                #calculate geodesic distance
                t = torch.linspace(0, 1, args.num_t).to(device).view(-1, 1)
                path = (1 - t) * z_i + t * z_j
                inner_points = path[1:-1].detach().clone().requires_grad_(True)
                optimizer = torch.optim.Adam([inner_points], lr=1e-2)

                # optimise to minimise curve energy
                for _ in range(100):
                    optimizer.zero_grad()
                    full_path = torch.cat([z_i.unsqueeze(0), inner_points, z_j.unsqueeze(0)], dim=0)
                    energy = curve_energy(full_path, model.decoders)
                    energy.backward()
                    optimizer.step()

                final_path = torch.cat([z_i.unsqueeze(0), inner_points, z_j.unsqueeze(0)], dim=0).detach()

                # monte carlo estimate of geodesic distance using the final path and random decoder samples
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
                
            print(f"finished run {run_idx}")

        # 4. Final CoV Calculation
        # For each pair, calculate std/mean across the 10 runs, then average those CoVs
        geo_cov = (all_geodesic_distances.std(dim=1) / all_geodesic_distances.mean(dim=1)).mean()
        euc_cov = (all_euclidean_distances.std(dim=1) / all_euclidean_distances.mean(dim=1)).mean()
        
        print(f"\nFinal Results for {args.num_decoders} Decoders:")
        print(f"Average Geodesic CoV: {geo_cov.item():.4f}")
        print(f"Average Euclidean CoV: {euc_cov.item():.4f}")





"""
Nu burde ensemble VAE virke. 
Når du kører den specificerer du også --n-decoder X for at bestemme hvor mange decoders du vil have.
Den træner en enkelt encoder og X decoders, og gemmer dem i en torch state dict. 
Vae class er også blevet ændret til at kunne håndtere flere decoders.
"""