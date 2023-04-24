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
