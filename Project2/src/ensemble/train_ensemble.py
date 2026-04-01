
import torch
import torch.nn as nn
import os
import csv
from Project2.src.ensemble.ensemble_vae import EnsembleVAE
from Project2.src.ensemble.ensemble_cov import compute_ensemble_cov

from Project2.src.vae import GaussianPrior, GaussianDecoder, GaussianEncoder
import Project2.src.helpers as help
from Project2.src.train_vae import train
from Project2.src.geodesics.plot_geodesics import plot_latent_geodesics



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
        choices=["train", "sample", "eval", "geodesics", "plot"],
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
        default=20,
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
    parser.add_argument(
        "--use-fixed",
        type=bool,
        default=True,
        help="whether to use the fixed points defined in helpers. Turn to false in combination with varying --num-curves (default: %(default)s)."
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
    train_data = help.subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = help.subsample(
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

    # uv run -m Project2.src.ensemble.train_ensemble train --epochs-per-decoder 200 --num-decoders 4 --device cuda
    # re-runs
    # epochs
    # num decoders
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.mode == "train":
        root_experiments_dir = os.path.join(script_dir, "experiments")
        os.makedirs(root_experiments_dir, exist_ok=True)

        experiment_name = f"{args.experiment_folder}_decoders_{args.num_decoders}"
        base_folder = os.path.join(root_experiments_dir, experiment_name)
        os.makedirs(base_folder, exist_ok=True)

        for run in range(args.num_reruns):
            print(f"\n--- Starting Retraining {run + 1}/{args.num_reruns} ---")
            
            # Create a specific folder for this run
            run_folder = os.path.join(base_folder, f"run_{run}")
            os.makedirs(run_folder, exist_ok=True)

            # 1. Initialize fresh components for every retraining
            decoder_list = nn.ModuleList([
                GaussianDecoder(help.new_decoder(M)) for _ in range(args.num_decoders)
            ])

            model = EnsembleVAE(
                GaussianPrior(M),
                decoder_list,
                GaussianEncoder(help.new_encoder(M)),
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
            save_path = os.path.join(run_folder, "model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    elif args.mode == "sample":

        decoder_list = nn.ModuleList([
            GaussianDecoder(help.new_decoder(M)) for _ in range(args.num_decoders)
        ])

        model = EnsembleVAE(
            GaussianPrior(M),
            decoder_list,
            GaussianEncoder(help.new_encoder(M)),
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
            GaussianDecoder(help.new_decoder(M)) for _ in range(args.num_decoders)
        ])

        model = EnsembleVAE(
            GaussianPrior(M),
            decoder_list,
            GaussianEncoder(help.new_encoder(M)),
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

    # uv run -m Project2.src.ensemble.train_ensemble geodesics
    elif args.mode == "geodesics":
        # 1. Fixed pairs of images to evaluate geodesic and euclidean distances on
        fixed_pairs = help.FIXED_PAIRS_SHORT

        # 2. Path to the experiments parent directory
        experiments_dir = os.path.join(script_dir, "experiments")

        # Initialize a list to store our results for the CSV
        csv_results = []
        # 3. Loop through all experiment subfolders
        for folder_name in sorted(os.listdir(experiments_dir)):
            base_folder = os.path.join(experiments_dir, folder_name)

            # Only process directories that match the expected naming pattern
            if os.path.isdir(base_folder) and "decoders_" in folder_name:
                # Extract the number of decoders from the end of the folder name
                try:
                    num_decoders = int(folder_name.split("_")[-1])
                except ValueError:
                    print(f"Skipping {folder_name}: Could not parse the number of decoders.")
                    continue

                print(f"\nEvaluating CoV for experiment: {folder_name} (Decoders: {num_decoders})")

                # 4. Initialize the specific model architecture for this run
                decoder_list = nn.ModuleList([
                    GaussianDecoder(help.new_decoder(M)) for _ in range(num_decoders)
                ])

                model = EnsembleVAE(
                    GaussianPrior(M),
                    decoder_list,
                    GaussianEncoder(help.new_encoder(M)),
                ).to(device)

                # Temporarily update args so compute_ensemble_cov uses the correct decoder count
                args.num_decoders = num_decoders

                # 5. Execute the computation
                geo_cov, euc_cov = compute_ensemble_cov(
                    args=args,
                    model=model,
                    mnist_test_loader=mnist_test_loader, 
                    device=device,
                    experiment_path=base_folder,
                    fixed_pairs=fixed_pairs
                )

                csv_results.append([
                    num_decoders, 
                    round(float(geo_cov), 6), 
                    round(float(euc_cov), 6)
                ])

        if csv_results:

            csv_results.sort(key=lambda x: x[0])
            
            csv_path = os.path.join(experiments_dir, "ensemble_cov_results.csv")
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
  
                writer.writerow(["num_decoders", "geodesic_cov", "euclidean_cov"])
  
                writer.writerows(csv_results)
                
            print(f"\nSuccessfully saved summary results to {csv_path}")
        else:
            print("\nNo valid experiment folders found to evaluate.")

    elif args.mode =="plot":
        if args.use_fixed:
            fixed_pairs = help.FIXED_PAIRS
        else:
            fixed_pairs = None

        all_z = []
        all_labels = []

        decoder_list = nn.ModuleList([
            GaussianDecoder(help.new_decoder(M)) for _ in range(args.num_decoders)
            ])

        model = EnsembleVAE(
            GaussianPrior(M),
            decoder_list,
            GaussianEncoder(help.new_encoder(M)),
            ).to(device)

        model_path = "????" #TODO
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)

                z = model.encoder(x).mean 
                all_z.append(z)
                all_labels.append(y)

        z_points = torch.cat(all_z, dim=0)
        labels = torch.cat(all_labels, dim=0).numpy()

        
        plot_latent_geodesics(
            z_points=z_points, 
            labels=labels, 
            decoders=model.decoders, 
            n_pairs=args.num_curves,
            fixed_pairs=fixed_pairs
        )




"""
Nu burde ensemble VAE virke. 
Når du kører den specificerer du også --n-decoder X for at bestemme hvor mange decoders du vil have.
Den træner en enkelt encoder og X decoders, og gemmer dem i en torch state dict. 
Vae class er også blevet ændret til at kunne håndtere flere decoders.
"""