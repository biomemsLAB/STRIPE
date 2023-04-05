import numpy as np
from utilities import WindowedFrameDataset
if __name__ == '__main__':
    print("Welcome to STRIPE")

    # create one big numpy array
    path_to_dir_raw_data=''
    path_to_save_numpy_array_raw=''
    path_to_save_numpy_array_normalized=''

    from utilities import preprocessing_for_multiple_recordings, save_frame_to_disk, load_frame_from_disk, normalize_frame
    frame_of_multiple_recordings = preprocessing_for_multiple_recordings(path_to_dir_raw_data)
    save_frame_to_disk(frame_of_multiple_recordings, path_to_save_numpy_array_raw)
    #frame_of_multiple_recordings = load_frame_from_disk(path_to_save_numpy_array_raw)
    frame_normalized = normalize_frame(frame_of_multiple_recordings, scaler_type='minmax')
    save_frame_to_disk(frame_normalized, path_to_save_numpy_array_normalized)


    frame = np.load("/mnt/MainNAS/temp/Persönliche Verzeichnisse/PS/save.npy", allow_pickle=True)
    time_rows = frame[:, 0]
    labels = frame[:, 1]
    from torch.utils.data import DataLoader
    windowed_frame_dataset = WindowedFrameDataset(time_rows, labels)
    batch_size = 32
    windowed_frame_dataloader = DataLoader(windowed_frame_dataset, batch_size=batch_size, shuffle=True)





