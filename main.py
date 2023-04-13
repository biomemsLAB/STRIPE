
if __name__ == '__main__':
    print("Welcome to STRIPE")

    # init
    from utilities import create_directory_structure, paths
    path_to_working_dir = '/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE'
    create_directory_structure(path_to_working_dir)
    # getting paths
    final_path_raw_train, final_path_raw_test, final_path_raw_val, final_path_raw_one, final_name_train_before_norm, final_name_test_before_norm, final_name_val_before_norm, final_name_one_before_norm = paths(path_to_working_dir)

    # preprocessing
    from utilities import preprocessing_for_multiple_recordings, save_frame_to_disk, load_frame_from_disk, normalize_frame
    frame_of_multiple_recordings = preprocessing_for_multiple_recordings(final_path_raw_one)
    save_frame_to_disk(frame_of_multiple_recordings, final_name_one_before_norm)
    # normalization
    frame_of_multiple_recordings = load_frame_from_disk(final_name_one_before_norm)
    scaler_type = 'minmax'
    final_path_raw_train, final_path_raw_test, final_path_raw_val, final_path_raw_one, final_name_train_after_norm, final_name_test_after_norm, final_name_val_after_norm, final_name_one_after_norm = paths(path_to_working_dir, scaler_type=scaler_type)
    frame_normalized = normalize_frame(frame_of_multiple_recordings, scaler_type=scaler_type)
    save_frame_to_disk(frame_normalized, final_name_one_after_norm)

    # using this one frame for different dataloaders
    from utilities import splitting_data_into_train_test_val_set, create_dataloader_simple
    data_train, label_train, data_test, label_test, data_val, label_val = splitting_data_into_train_test_val_set(frame_of_multiple_recordings['signals'], frame_of_multiple_recordings['label_per_window'])

    train_dataloader = create_dataloader_simple(data_train, label_train)
    test_dataloader = create_dataloader_simple(data_test, label_test)
    val_dataloader = create_dataloader_simple(data_val, label_val)

    # using this frame just for one specific dataloader (train)
    # frame = frame_of_multiple_recordings  #np.load("/mnt/MainNAS/temp/Persönliche Verzeichnisse/PS/save.npy", allow_pickle=True)
    # from utilities import create_dataloader
    # windowed_frame_dataloader = create_dataloader(frame)

    from nn_handle import handle_model
    from custom_modles import DenseModel
    model = DenseModel(in_features=10, hidden_features=20, out_features=2)
    handler_object = handle_model(model, train_dataloader, val_dataloader, test_dataloader)
    handler_object.run(epochs=10)

    print('main finished')





