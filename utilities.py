import numpy as np
def create_directory_structure(path):
    """
    Checks if a certain directory structure exists at the given path, and creates the structure if it doesn't.
    :param path: path to working directory.
    :return: None
    """
    import os
    # Define the directory structure you want to create
    directory_structure = [
        "data/raw/train",
        "data/raw/test",
        "data/raw/val",
        "data/raw/one",
        "data/save/after_normalization",
        "data/save/before_normalization"
    ]

    # Check if each directory in the structure exists, and create it if it doesn't
    for directory in directory_structure:
        directory_path = os.path.join(path, directory)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")
        else:
            print(f"Directory already exists: {directory_path}")

def paths(path, scaler_type=None):
    import os
    path_work_dir = path
    path_raw_train = "data/raw/train"
    path_raw_test = "data/raw/test"
    path_raw_val = "data/raw/val"
    path_raw_one = "data/raw/one"
    path_save_after_norm = "data/save/after_normalization"
    path_save_before_norm = "data/save/before_normalization"

    filename_train_before_norm = 'train_frame_raw.npy'
    filename_test_before_norm = 'test_frame_raw.npy'
    filename_val_before_norm = 'val_frame_raw.npy'
    filename_one_before_norm = 'one_frame_raw.npy'

    final_path_raw_train = os.path.join(path_work_dir, path_raw_train)
    final_path_raw_test = os.path.join(path_work_dir, path_raw_test)
    final_path_raw_val = os.path.join(path_work_dir, path_raw_val)
    final_path_raw_one = os.path.join(path_work_dir, path_raw_one)

    if scaler_type is None:
        final_name_train_before_norm = os.path.join(path_work_dir, path_save_before_norm, filename_train_before_norm)
        final_name_test_before_norm = os.path.join(path_work_dir, path_save_before_norm, filename_test_before_norm)
        final_name_val_before_norm = os.path.join(path_work_dir, path_save_before_norm, filename_val_before_norm)
        final_name_one_before_norm = os.path.join(path_work_dir, path_save_before_norm, filename_one_before_norm)
        return final_path_raw_train, final_path_raw_test, final_path_raw_val, final_path_raw_one, final_name_train_before_norm, final_name_test_before_norm, final_name_val_before_norm, final_name_one_before_norm

    elif scaler_type is not None:
        filename_train_after_norm = ''.join(['train_frame_norm_', scaler_type, '.npy'])
        filename_test_after_norm = ''.join(['test_frame_norm_', scaler_type, '.npy'])
        filename_val_after_norm = ''.join(['val_frame_norm_', scaler_type, '.npy'])
        filename_one_after_norm = ''.join(['one_frame_norm_', scaler_type, '.npy'])

        final_name_train_after_norm = os.path.join(path_work_dir, path_save_after_norm, filename_train_after_norm)
        final_name_test_after_norm = os.path.join(path_work_dir, path_save_after_norm, filename_test_after_norm)
        final_name_val_after_norm = os.path.join(path_work_dir, path_save_after_norm, filename_val_after_norm)
        final_name_one_after_norm = os.path.join(path_work_dir, path_save_after_norm, filename_one_after_norm)
        return final_path_raw_train, final_path_raw_test, final_path_raw_val, final_path_raw_one, final_name_train_after_norm, final_name_test_after_norm, final_name_val_after_norm, final_name_one_after_norm

def import_recording_h5(path):
    """
    Import recording h5 file from MEArec.
    :param path: path to file
    :return: signal_raw, timestamps, ground_truth, channel_positions, template_locations
    """
    import h5py  # hdf5
    import numpy as np
    h5 = h5py.File(path, 'r')
    signal_raw = np.array(h5["recordings"])
    timestamps = np.array(h5["timestamps"])
    ground_truth = []
    for i in range(len(h5["spiketrains"].keys())):
        ground_truth.append(np.array(h5["spiketrains"][str(i)]["times"]))
    channel_positions = np.array(h5["channel_positions"]) #indexes of columns x: 1 y: 2 z: 0
    template_locations = np.array(h5["template_locations"]) #indexes of columns x: 1 y: 2 z: 0
    return signal_raw, timestamps, ground_truth, channel_positions, template_locations

def create_labels_for_spiketrain(timestamps, times):
    """
    Assign ground truth label of times to the nearest value of timestamps.
    :param timestamps: all timestamps
    :param times: ground truth timestamps of occurring spikes
    :return: labels: Returns list of length of timestamps with 1s at positions of times and 0s at the other positions.
    """
    import bisect
    import numpy as np
    labels = np.zeros(len(timestamps), dtype=int)
    times_sorted = np.sort(timestamps)
    for i, t in enumerate(times):
        index = bisect.bisect_left(times_sorted, t)
        if index == 0:
            nearest_timestamp = times_sorted[0]
        elif index == len(times_sorted):
            nearest_timestamp = times_sorted[-1]
        else:
            left_timestamp = times_sorted[index - 1]
            right_timestamp = times_sorted[index]
            if t - left_timestamp < right_timestamp - t:
                nearest_timestamp = left_timestamp
            else:
                nearest_timestamp = right_timestamp
        nearest_index = np.searchsorted(timestamps, nearest_timestamp)
        labels[nearest_index] = 1
    return labels

def create_labels_of_all_spiketrains(ground_truth, timestamps):
    """
    Create labels for all ground_truth spiketrains using create_labels_for_spiketrain()
    :param ground_truth:
    :param timestamps:
    :return: labels_of_all_spiketrains: Returns numpy array of all ground_truth spiketrains with 1s for a spike and
        0s otherwise.
    """
    import numpy as np
    labels_of_all_spiketrains = []
    for i in range(len(ground_truth)):
        labels = create_labels_for_spiketrain(timestamps, ground_truth[i])
        labels_of_all_spiketrains.append(labels)
    return np.array(labels_of_all_spiketrains)

def assign_neuron_locations_to_electrode_locations(electrode_locations, neuron_locations, threshold):
    """
    Assigns the index of a neuron location to the index of an electrode location if
    the distance between them is less than or equal to the threshold value.
    :param electrode_locations:
    :param neuron_locations:
    :param threshold: The maximum distance between an electrode location and a neuron location for them
        to be considered a match.
    :return:
    """
    import pandas as pd
    import numpy as np

    electrode_locations_df = pd.DataFrame(electrode_locations)
    neuron_locations_df = pd.DataFrame(neuron_locations)

    # Compute the distance between each electrode location and each neuron location
    distances = np.sqrt(((electrode_locations_df.values[:, np.newaxis, :] - neuron_locations_df.values)**2).sum(axis=2))

    # Create an empty DataFrame to store the results
    assignments = pd.DataFrame(index=electrode_locations_df.index, columns=neuron_locations_df.index, dtype=bool)

    # Assign each channel position to its closest neuron_locations (if within the threshold distance)
    for i, point_idx in enumerate(neuron_locations_df.index):
        mask = distances[:, i] <= threshold
        assignments.iloc[:, i] = mask

    return assignments

def merge_data_to_location_assignments(assignments, signal_raw, labels_of_all_spiketrains, timestamps):
    """
    Assigns the label vectors to the raw data. For the merging of multiple spiketrains to one electrode the
    np.logical_or() is used. For electrodes without an assignment to spiketrains empty spiketrains are generated.
    Additionally, timestamps are added.
    :param assignments: A DataFrame representing the local assignment between neurons and electrodes.
        With rows corresponding to electrodes and columns corresponding to neurons. Each cell in the
        DataFrame is True if the corresponding channel position is within the threshold distance of the
        corresponding neuron, and False otherwise. If a channel position is not assigned to any neuron position,
        the corresponding cells are False.
    :param signal_raw: A numpy array representing the recorded signal, with rows
        corresponding to electrodes of the MEA and columns corresponding to timestamps.
    :param labels_of_all_spiketrains: A numpy array representing the labels, with rows
        corresponding to spiketrains of the different neurons and columns corresponding to timestamps.
    :param timestamps:
    :return: merged_data: A numpy array representing the merged data. It's build like nested lists. The structure:
        [[[raw_data of the first electrode],[labels of the first electrode],[timestamps of the first electrode]],
        [[raw_data of the second electrode],[labels of the second electrode],[timestamps of the second electrode]], etc.]
    """
    import numpy as np

    assignments2 = np.array(assignments, dtype=bool)
    merged_data = []

    for i in range(assignments2.shape[0]):  # iterate over rows in assignments
        row = assignments2[i]  # equals electrode
        merged = np.zeros(len(labels_of_all_spiketrains[0]))  # generating empty spiketrains
        for j, value in enumerate(row):  # iterate over "columns" in rows
            if value:
                merged = np.logical_or(merged, labels_of_all_spiketrains[j, :])
        merged_data.append([signal_raw[i], merged.astype(int), timestamps])
    return np.array(merged_data)

def devide_3_vectors_into_equal_windows_with_step(x1, x2, x3, window_size, step_size=None):
    """
    Devides vectors x1, x2, x3 into windows with one window_size. step_size is used to generate more windows with overlap.
    :param x1: Input list to be devided.
    :param x2: Input list to be devided.
    :param x3: Input list to be devided.
    :param window_size: Size of each window.
    :param step_size: If the step_size is not provided, it defaults to the window_size.
        If the step_size is set to True, it is set to half of the window_size.
        If the step_size is set to any other value, it is used directly as the step_size.
    :return: Returns for every input a list of lists. Each included list represents a window.
    """
    if step_size is None:
        step_size = window_size
    elif step_size is True:
        step_size = window_size // 2
    x1_windows = []
    x2_windows = []
    x3_windows = []
    for i in range(0, len(x1) - window_size + 1, step_size):
        x1_windows.append(x1[i:i + window_size])
        x2_windows.append(x2[i:i + window_size])
        x3_windows.append(x3[i:i + window_size])
    return x1_windows, x2_windows, x3_windows

def application_of_windowing(merged_data, window_size, step_size=None):
    """
    Application of windowing
    :param merged_data:
    :param window_size:
    :param step_size:
    :return: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
        and columns corresponding to signal_raw, labels, timestamps and electrode number.
    """
    import numpy as np

    frame = []
    for i in range(len(merged_data)):
        win1, win2, win3 = devide_3_vectors_into_equal_windows_with_step(merged_data[i][0], merged_data[i][1], merged_data[i][2], window_size, step_size)
        for l in range(len(win1)):
            frame.append(np.array([win1[l], win2[l], win3[l], i], dtype=object))
    return np.array(frame)

def application_of_windowing_v2(merged_data, window_size, step_size=None, feature_calculation=False):
    """
    Version 2 of application_of_windowing(). Currently designed just for one window_size and one step_size. Devides
    merged data into windows and calculate features for windows. Additionally, set a label for each window.
    :param merged_data: A numpy array representing the merged data from merge_data_to_location_assignments().
    :param window_size: Size of each window in counts.
    :param step_size: Size (in counts) of offset between windows.
        If the step_size is not provided, it defaults to the window_size.
        If the step_size is set to True, it is set to half of the window_size.
        If the step_size is set to any other value, it is used directly as the step_size.
    :param feature_calculation: bool. Defines the boolean value if calculate_features() is active or not.
    :return: A custom ndarray representing the merged data, which is devided by windows. With rows corresponding to windows
        and columns corresponding to signal_raw (signals), labels, timestamps, electrode number, features and label_per_window.
    """
    import numpy as np

    if step_size is None:
        step_size = window_size
    elif step_size is True:
        step_size = window_size // 2
    elif step_size is not None:
        step_size = int(step_size)

    # calculate number of features dynamically based on the returned feature vector from calculate_features()
    sample_data = merged_data[0][0:window_size]
    features_size = calculate_features(window_data=sample_data, calculate=feature_calculation).shape[0]

    # defining empty custom ndarray
    num_windows = sum((data.shape[1] - window_size) // step_size + 1 for data in merged_data)
    frame = np.zeros((num_windows,), dtype=[
        ('signals', np.float32, (window_size,)),
        ('labels', np.float32, (window_size,)),
        ('timestamps', np.float32, (window_size,)),
        ('electrode_number', np.int32),
        ('features', np.float32, (features_size,)),
        ('label_per_window', np.int32)
    ])

    curr_idx = 0
    for i, data in enumerate(merged_data):
        # calculate windows
        num_windows_i = (data.shape[1] - window_size) // step_size + 1
        win1 = np.lib.stride_tricks.as_strided(
            data[0], shape=(num_windows_i, window_size), strides=(data[0].strides[0] * step_size, data[0].strides[0]))
        win2 = np.lib.stride_tricks.as_strided(
            data[1], shape=(num_windows_i, window_size), strides=(data[1].strides[0] * step_size, data[1].strides[0]))
        win3 = np.lib.stride_tricks.as_strided(
            data[2], shape=(num_windows_i, window_size), strides=(data[2].strides[0] * step_size, data[2].strides[0]))

        for j in range(num_windows_i):
            # apply windowing to resulting frame
            frame[curr_idx]['signals'] = win1[j]
            frame[curr_idx]['labels'] = win2[j]
            frame[curr_idx]['timestamps'] = win3[j]
            frame[curr_idx]['electrode_number'] = i
            # calculate features for each window
            frame[curr_idx]['features'] = calculate_features(window_data=win1[j], calculate=feature_calculation)
            frame[curr_idx]['label_per_window'] = label_a_window_from_labels_of_a_window(win2[j])
            curr_idx += 1

    return frame

def calculate_features(window_data, calculate=False):
    """
    Calculates features for input data.
    :param window_data: input data for calculation
    :param calculate: bool.
        If calculate is set to True, it calculates the defined features and returns the feature array.
        If calculate is set to False, it returns a np.zeros array.
    :return: feature array or np.zeros array with size 1
    """
    import numpy as np
    if calculate is True:
        # Assuming 3 features for example purposes
        num_features = 3
        features = np.zeros((num_features,))
        features[0] = np.mean(window_data)
        features[1] = np.min(window_data)
        features[2] = np.max(window_data)
        return features
    elif calculate is False:
        return np.zeros((1,))

def label_a_window_from_labels_of_a_window(window_data):
    """
    Finds the max value of input data and returns it as integer. Input data of labels of a window should only be 0s and 1s.
    :param window_data: input data for calculation
    :return: label. It represents the label of the input window.
    """
    import numpy as np
    label = int(np.max(window_data))
    return label

def count_indexes_up_to_value(arr, value):
    import numpy as np
    # Find the indexes where the array values are less than or equal to the specified value
    indexes = np.where(arr <= value)[0]
    # Count the number of indexes
    count = len(indexes)
    return count
def get_window_size_in_index_count(timestamps, window_size_in_sec):
    """
    calculate window size in index counts from defined windowsize (in sec)
    :param timestamps: all timestamps (used for calculation)
    :param window_size_in_sec: windowsize in seconds
    :return: window_size_in_count
    """
    window_size_in_count = count_indexes_up_to_value(timestamps, window_size_in_sec)
    return window_size_in_count - 1

def preprocessing_for_one_recording(path, window_size_in_sec=0.001):
    """
    preprocessing pipeline for one recording (without normalization)
    :param path: path to recording file
    :param window_size_in_sec: window size in seconds (default = 0.001)
    :return: frame: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
        and columns corresponding to signal_raw, labels, timestamps and electrode number.
    """
    signal_raw, timestamps, ground_truth, electrode_locations, neuron_locations = import_recording_h5(path)
    labels_of_all_spiketrains = create_labels_of_all_spiketrains(ground_truth, timestamps)
    assignments = assign_neuron_locations_to_electrode_locations(electrode_locations, neuron_locations, 20)
    merged_data = merge_data_to_location_assignments(assignments, signal_raw.transpose(), labels_of_all_spiketrains, timestamps)
    window_size_in_counts = get_window_size_in_index_count(timestamps, window_size_in_sec)
    frame = application_of_windowing_v2(merged_data=merged_data, window_size=window_size_in_counts, step_size=None, feature_calculation=False)
    print('preprocessing finished for:', path)
    return frame

def preprocessing_for_multiple_recordings(path):
    """
    preprocessing pipeline for multiple recordings (without normalization)
    :param path: path to recording files. Only MEArec generated h5. recording files may be located here.
    :return: frame_of_multiple_recordings: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
        and columns corresponding to signal_raw, labels, timestamps and electrode number. No assignment to the recording!
    """
    from pathlib import Path
    import numpy as np
    recordings = [p for p in Path(path).iterdir()]
    frame_of_multiple_recordings = None
    print('preprocessing started for:', path)
    for rec in recordings:
        frame_of_one_recording = preprocessing_for_one_recording(rec)
        if frame_of_multiple_recordings is None:
            frame_of_multiple_recordings = frame_of_one_recording.copy()
        else:
            frame_of_multiple_recordings = np.hstack((frame_of_multiple_recordings, frame_of_one_recording))
    print('preprocessing finished for:', path)
    return frame_of_multiple_recordings

def save_frame_to_disk(frame, path_target):
    import numpy as np
    print('started saving frame to disk')
    np.save(path_target, frame, allow_pickle=True)
    print('frame saved to disk')

def load_frame_from_disk(path_source):
    import numpy as np
    print('started loading frame from disk')
    frame = np.load(path_source, allow_pickle=True)
    print('frame loaded from disk')
    return frame

def normalize_frame(frame, scaler_type='minmax'):
    """
    Normalizes the raw data in the input array using the specified scaler type
    :param frame: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
        and columns corresponding to signal_raw, labels, timestamps and electrode number. No assignment to the recording!
    :param scaler_type: possible Scalers from sklearn.preprocessing: StandardScaler, MinMaxScaler, RobustScaler
    :return: A numpy array representing the merged data, which is devided by windows. With rows corresponding to windows
        and columns corresponding to signal_raw (normalized), labels, timestamps and electrode number. No assignment to the recording!
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Scaler type {scaler_type} not supported. Please choose 'standard', 'minmax', or 'robust'")

    print(f"Normalization with scaler type '{scaler_type}' started")
    for i in frame:
        data_raw = i[0]
        data_norm = scaler.fit_transform(data_raw.reshape(-1,1))
        i[0] = data_norm.flatten()
    print(f"Normalization with scaler type '{scaler_type}' finished")
    return frame

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
class WindowedFrameDataset(Dataset):
    """
    A PyTorch dataset that represents windowed frames of data and their labels.

    Parameters
    ----------
    data : numpy.ndarray or list
        The input data array of shape (num_frames, num_channels, height, width).
    labels : numpy.ndarray or list
        The target labels array of shape (num_frames,).

    Attributes
    ----------
    data : numpy.ndarray or list
        The input data array.
    transform : torchvision.transforms
        The transform applied to the input data.
    labels : numpy.ndarray or list
        The target labels array.

    Methods
    -------
    __len__()
        Returns the length of the dataset.
    __getitem__(idx)
        Returns the data and label of the specified index.

    Notes
    -----
    This class takes a numpy array or a list of windowed frames of data and their labels, and transforms them into a PyTorch dataset.
    It also applies a transform to the input data to convert it to a tensor.

    Example
    -------
    # create a dataset
    time_rows = np.random.randn(100, 3, 32, 32)
    labels = np.random.randint(0, 2, size=(100,))
    windowed_frame_dataset = WindowedFrameDataset(time_rows, labels)

    # create a dataloader
    batch_size = 32
    windowed_frame_dataloader = DataLoader(windowed_frame_dataset, batch_size=batch_size, shuffle=True)
    """
    def __init__(self, data, labels):
        self.data = data.astype('float32')
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.labels = labels.astype('float32')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        return data, labels

def create_dataloader(frame, batch_size=32):
    from torch.utils.data import DataLoader
    time_rows = frame['signals']  #frame[:, 0]
    labels = frame['label_per_window'] #frame[:, 1]
    windowed_frame_dataset = WindowedFrameDataset(time_rows, labels)
    windowed_frame_dataloader = DataLoader(windowed_frame_dataset, batch_size=batch_size, shuffle=True)
    return windowed_frame_dataloader

def create_dataloader_simple(data, labels, batch_size=32):
    from torch.utils.data import DataLoader
    time_rows = data
    windowed_frame_dataset = WindowedFrameDataset(time_rows, labels)
    windowed_frame_dataloader = DataLoader(windowed_frame_dataset, batch_size=batch_size, shuffle=True)
    return windowed_frame_dataloader

def splitting_data_into_train_test_val_set(data, labels, test_and_val_size=0.4, val_size_of_test_and_val_size=0.5):
    """
    Splits data and labels into training, test and validation set.
    :param data: input set which contains data
    :param labels: input set which contains labels for data
    :param test_and_val_size: size of test and validation set combined. Rest equals training set.
    :param val_size_of_test_and_val_size: size of validation set corresponding to test_and_val_size. Rest equals test set.
    :return: training, test and validation set for data and labels
    """
    from sklearn.model_selection import train_test_split
    X = data
    y = labels
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_and_val_size)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=val_size_of_test_and_val_size)
    return x_train, y_train, x_test, y_test, x_val, y_val

def balancing_dataset_with_undersampling(data, labels):
    from imblearn.under_sampling import RandomUnderSampler
    print('balancing started')
    undersample = RandomUnderSampler(sampling_strategy='majority')
    data_result, labels_result = undersample.fit_resample(data, labels)
    print('balancing finished')
    return data_result, labels_result
