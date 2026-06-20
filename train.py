import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ivysaurus_model import IvysaurusModel

# RUN ON GPU:1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

###########################################################

dimensions = 24
nClasses = 5
nTrackVars = 21
nShowerVars = 10

batchSize = 64
learningRate = 1e-4

###########################################################


def to_nchw(arr):
    """(N, H, W, 1) -> (N, 1, H, W) float32 tensor."""
    arr = np.asarray(arr, dtype=np.float32)
    return torch.from_numpy(np.transpose(arr, (0, 3, 1, 2)))


class IvysaurusDataset(Dataset):
    def __init__(self, grids, trackVars, showerVars, y):
        # grids: dict of name -> (N, H, W, 1) numpy arrays
        self.grids = {k: to_nchw(v) for k, v in grids.items()}
        self.trackVars = torch.from_numpy(trackVars.astype(np.float32))
        self.showerVars = torch.from_numpy(showerVars.astype(np.float32))
        # CrossEntropyLoss wants class indices, not one-hot
        self.labels = torch.from_numpy(np.argmax(y, axis=1).astype(np.int64))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
        sample = {k: v[i] for k, v in self.grids.items()}
        sample["trackVars"] = self.trackVars[i]
        sample["showerVars"] = self.showerVars[i]
        return sample, self.labels[i]


# Order matters: matches model.forward signature
GRID_KEYS = [
    "startU", "startU_mask", "endU", "endU_mask",
    "startV", "startV_mask", "endV", "endV_mask",
    "startW", "startW_mask", "endW", "endW_mask",
]


def run_model(model, batch, device):
    args = [batch[k].to(device) for k in GRID_KEYS]
    args.append(batch["trackVars"].to(device))
    args.append(batch["showerVars"].to(device))
    return model(*args)


###########################################################

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Load data ---
    suffix = "Contained" if args.is_contained else "Exiting"
    trainFileNames = glob.glob(
        f'/home/imawby/Ivysaurus/files/filtered_0_{suffix}.npz')
    print(trainFileNames)

    # mapping: model input name -> (.npz key prefix)
    # In the original .npz the keys are e.g. startGridU_train / startGridU_valid_train
    grid_map = {
        "startU": "startGridU", "startU_mask": "startGridU_valid",
        "endU":   "endGridU",   "endU_mask":   "endGridU_valid",
        "startV": "startGridV", "startV_mask": "startGridV_valid",
        "endV":   "endGridV",   "endV_mask":   "endGridV_valid",
        "startW": "startGridW", "startW_mask": "startGridW_valid",
        "endW":   "endGridW",   "endW_mask":   "endGridW_valid",
    }

    def empty_grids():
        return {k: np.empty((0, dimensions, dimensions, 1)) for k in GRID_KEYS}

    grids_train, grids_test = empty_grids(), empty_grids()
    trackVars_train = np.empty((0, nTrackVars))
    showerVars_train = np.empty((0, nShowerVars))
    trackVars_test = np.empty((0, nTrackVars))
    showerVars_test = np.empty((0, nShowerVars))
    y_train = np.empty((0, nClasses))
    y_test = np.empty((0, nClasses))

    for fname in trainFileNames:
        print("Reading file:", fname, ", This may take a while...")
        data = np.load(fname)

        for k, prefix in grid_map.items():
            grids_train[k] = np.concatenate(
                (grids_train[k], data[f"{prefix}_train"]), axis=0)
            grids_test[k] = np.concatenate(
                (grids_test[k], data[f"{prefix}_test"]), axis=0)

        trackVars_train = np.concatenate((trackVars_train, data['trackVars_train']), axis=0)
        trackVars_test = np.concatenate((trackVars_test, data['trackVars_test']), axis=0)
        showerVars_train = np.concatenate((showerVars_train, data['showerVars_train']), axis=0)
        showerVars_test = np.concatenate((showerVars_test, data['showerVars_test']), axis=0)
        y_train = np.concatenate((y_train, data['y_train']), axis=0)
        y_test = np.concatenate((y_test, data['y_test']), axis=0)

    print("y_train:", y_train.shape, "y_test:", y_test.shape)

    train_ds = IvysaurusDataset(grids_train, trackVars_train, showerVars_train, y_train)
    test_ds = IvysaurusDataset(grids_test, trackVars_test, showerVars_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batchSize, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batchSize, shuffle=False,
                             num_workers=4, pin_memory=True)

    # --- Class weights ---
    indexVector = np.argmax(y_train, axis=1)
    counts = [np.count_nonzero(indexVector == c) for c in range(nClasses)]
    maxParticle = max(counts)
    classWeights = np.array([maxParticle / c for c in counts], dtype=np.float32)
    print("Class Weights:", classWeights)
    class_weights_t = torch.from_numpy(classWeights).to(device)

    # --- Model / optim / loss ---
    model = IvysaurusModel(dimensions, nClasses, nTrackVars, nShowerVars).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learningRate)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    # ReduceLROnPlateau on val_loss, factor 0.1, patience 2, min_lr 1e-6
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.1, patience=2, min_lr=1e-6)

    filePath = (f'/home/imawby/Ivysaurus/models/'
                f'my_model_{"contained" if args.is_contained else "exiting"}'
                f'_VGG_BN_VARS.pt')

    best_val_acc = -1.0

    # --- Training loop ---
    for epoch in range(args.n_epochs):

        # ---- train ----
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch, labels in train_loader:
            labels = labels.to(device)
            optimiser.zero_grad()
            logits = run_model(model, batch, device)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ---- validate ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch, labels in test_loader:
                labels = labels.to(device)
                logits = run_model(model, batch, device)
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{args.n_epochs} - "
              f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        # checkpoint: save best on val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), filePath)
            print(f"  val_acc improved to {val_acc:.4f}, saved model to {filePath}")


##########################################################################################################

def parse_cli():
    parser = argparse.ArgumentParser(description="Ivysaurus PID")
    parser.add_argument("--is_contained", action="store_true",
                        help="Training for contained particles?")
    parser.add_argument("--n_epochs", type=int, required=True,
                        help="Number of epochs")
    return parser.parse_args()


##########################################################################################################

if __name__ == "__main__":
    args = parse_cli()
    main(args)
