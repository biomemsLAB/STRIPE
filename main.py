import numpy as np
from utilities import WindowedFrameDataset
if __name__ == '__main__':
    print("Welcome to STRIPE")
    frame = np.load("/mnt/MainNAS/temp/Persönliche Verzeichnisse/PS/save.npy", allow_pickle=True)
    time_rows = frame[:, 0]
    labels = frame[:, 1]
    from torch.utils.data import DataLoader
    windowed_frame_dataset = WindowedFrameDataset(time_rows, labels)
    batch_size = 32
    windowed_frame_dataloader = DataLoader(windowed_frame_dataset, batch_size=batch_size, shuffle=True)





