import numpy as np
if __name__ == '__main__':
    from utilities import FrameDataSet

    frame = np.load("/mnt/MainNAS/temp/Persönliche Verzeichnisse/PS/save.npy", allow_pickle=True)
    time_rows = frame[:, 0]
    labels = frame[:, 1]
    from torch.utils.data import Dataset, DataLoader
    frame_dataset = FrameDataSet(time_rows, labels)
    batch_size = 32
    moving_mnist_dataloader = DataLoader(frame_dataset, batch_size=batch_size, shuffle=True)
    print("Welcome to STRIPE")



