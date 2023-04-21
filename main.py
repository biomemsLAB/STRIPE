
if __name__ == '__main__':
    print("Welcome to STRIPE")

    # init
    from utilities import create_directory_structure, paths
    path_to_working_dir = '/home/psteigerwald/PycharmProjects/STRIPE'
    create_directory_structure(path_to_working_dir)
    # getting paths
    final_path_raw_train, final_path_raw_test, final_path_raw_val, final_path_raw_one, final_name_train_before_norm, final_name_test_before_norm, final_name_val_before_norm, final_name_one_before_norm = paths(path_to_working_dir)

    # preprocessing
    from utilities import preprocessing_for_multiple_recordings, save_frame_to_disk, load_frame_from_disk, normalize_frame
    #frame_of_multiple_recordings = preprocessing_for_multiple_recordings(final_path_raw_one)
    #save_frame_to_disk(frame_of_multiple_recordings, final_name_one_before_norm)
    # normalization
    #frame_of_multiple_recordings = load_frame_from_disk(final_name_one_before_norm)
    scaler_type = 'minmax'
    final_path_raw_train, final_path_raw_test, final_path_raw_val, final_path_raw_one, final_name_train_after_norm, final_name_test_after_norm, final_name_val_after_norm, final_name_one_after_norm = paths(path_to_working_dir, scaler_type=scaler_type)
    #frame_normalized = normalize_frame(frame_of_multiple_recordings, scaler_type=scaler_type)
    #save_frame_to_disk(frame_normalized, final_name_one_after_norm)
    # final_name_one_after_norm = '/home/psteigerwald/PycharmProjects/STRIPE/data/save/before_normalization/one_frame_raw.npy'
    #frame_of_multiple_recordings = load_frame_from_disk(final_name_one_before_norm)
    # frame_of_multiple_recordings = load_frame_from_disk("/home/psteigerwald/PycharmProjects/STRIPE/data/save/one_frame_raw.npy")
    frame_of_multiple_recordings = load_frame_from_disk("/home/psteigerwald/PycharmProjects/STRIPE/data/save/before_normalization/set_0001_new_sel_0000-0199_raw.npy")
    # using this one frame for different dataloaders
    from utilities import create_dataloader_simple
    from utilities import dataset_pipeline_for_training_process

    data_train_balanced, label_train_balanced, data_test, label_test, data_val, label_val = dataset_pipeline_for_training_process(frame_of_multiple_recordings)

    train_dataloader = create_dataloader_simple(data_train_balanced, label_train_balanced)
    test_dataloader = create_dataloader_simple(data_test, label_test)
    val_dataloader = create_dataloader_simple(data_val, label_val)

    # using this frame just for one specific dataloader (train)
    # frame = frame_of_multiple_recordings  #np.load("/mnt/MainNAS/temp/Persönliche Verzeichnisse/PS/save.npy", allow_pickle=True)
    # from utilities import create_dataloader
    # windowed_frame_dataloader = create_dataloader(frame)
    # dataset_pipeline_for_training_process
    from nn_handle import handle_model
    from custom_modles import DenseModel, ViT, CustomResNet, ResidualBlock, LSTM_Model
    dense_model = DenseModel(in_features=10, hidden_features=20, out_features=2)
    custom_resnet34 = CustomResNet(ResidualBlock, [3, 4, 6, 3])
    transformer_model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1, dim=64, depth=6, heads=8, mlp_dim=128)
    lstm_model = LSTM_Model(in_features=10, hidden_features=20, out_features=2)
    from nn_utilities import compare_models_acc_over_epoch
    path_to_save = "/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE/Results"
    compare_models_acc_over_epoch(train_dataloader, val_dataloader, test_dataloader, lstm_model, dense_model, transformer_model, epochs=100, learning_rate=0.0001, path_to_save=path_to_save)

    print('main finished')





