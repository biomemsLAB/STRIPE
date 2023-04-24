# here a script for operating several pipelines


from utilities import preprocessing_for_multiple_recordings, save_frame_to_disk, load_frame_from_disk
#path_for_h5_files =''
#frame_name =''
#frame = preprocessing_for_multiple_recordings(path_for_h5_files)
#save_frame_to_disk(frame, frame_name)

from utilities import normalize_frame
#frame_normalized = normalize_frame(frame=frame, scaler_type='minmax')

from utilities import dataset_pipeline_creating_even_larger_datasets
#path_source_npys = ''
#path_target_datasets = ''
#dataset_pipeline_creating_even_larger_datasets(path_source_npys, path_target_datasets)

# using frame just for one specific dataloader (train, validation or test)
from utilities import create_dataloader
#windowed_frame_dataloader_train = create_dataloader(frame)


def dataset_pipeline_generate_debug_dataset(path_to_working_dir, cropping_size=0.2):
    from utilities import cropping_dataset, loading_numpy_datasets_for_training, save_frame_to_disk
    import os
    data_train_balanced, label_train_balanced, data_test, label_test, data_val, label_val = loading_numpy_datasets_for_training(path_to_working_dir)

    data_train_balanced_debug, label_train_balanced_debug = cropping_dataset(data_train_balanced, label_train_balanced, cropping_size)
    data_test_debug, label_test_debug = cropping_dataset(data_test, label_test, cropping_size)
    data_val_debug, label_val_debug = cropping_dataset(data_val, label_val, cropping_size)

    print('train: spikes:', label_train_balanced_debug.sum(), 'total:', len(label_train_balanced_debug))
    print('test: spikes:', label_test_debug.sum(), 'total:', len(label_test_debug))
    print('val: spikes:', label_val_debug.sum(), 'total:', len(label_val_debug))

    path_debug = 'data/debug/prepared_for_training'

    save_frame_to_disk(data_train_balanced_debug, os.path.join(path_to_working_dir, path_debug, 'frames_x_train_res.npy'))
    save_frame_to_disk(label_train_balanced_debug, os.path.join(path_to_working_dir, path_debug, 'frames_y_train_res.npy'))
    save_frame_to_disk(data_test_debug, os.path.join(path_to_working_dir, path_debug, 'frames_x_test_crp.npy'))
    save_frame_to_disk(label_test_debug, os.path.join(path_to_working_dir, path_debug, 'frames_y_test_crp.npy'))
    save_frame_to_disk(data_val_debug, os.path.join(path_to_working_dir, path_debug, 'frames_x_val_crp.npy'))
    save_frame_to_disk(label_val_debug, os.path.join(path_to_working_dir, path_debug, 'frames_y_val_crp.npy'))
    print('files saved at:', os.path.join(path_to_working_dir, path_debug))
    return None

path_to_working_dir = '/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE'
#dataset_pipeline_generate_debug_dataset(path_to_working_dir)

print ('pipeline finished')