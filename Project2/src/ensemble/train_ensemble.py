import torch
import torch.nn as nn
import os

from Project2.src.ensemble.ensemble_vae import *


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


    if args.mode == "train":
        base_folder = f"{args.experiment_folder}_decoders_{args.num_decoders}"
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
        model.load_state_dict(torch.load(f"{args.experiment_folder}/model.pt", map_location=device))
        model.eval()

"""
Nu burde ensemble VAE virke. 
Når du kører den specificerer du også --n-decoder X for at bestemme hvor mange decoders du vil have.
Den træner en enkelt encoder og X decoders, og gemmer dem i en torch state dict. 
Vae class er også blevet ændret til at kunne håndtere flere decoders.
"""