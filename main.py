
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
    frame_of_multiple_recordings = load_frame_from_disk(final_name_one_before_norm)
    # using this one frame for different dataloaders
    from utilities import splitting_data_into_train_test_val_set, create_dataloader_simple
    data_train, label_train, data_test, label_test, data_val, label_val = splitting_data_into_train_test_val_set(frame_of_multiple_recordings['signals'], frame_of_multiple_recordings['label_per_window'])

    print('train_sum:', label_train.sum(), 'length:', len(label_train))
    print('test_sum:', label_test.sum(), 'length:', len(label_test))
    print('val_sum:', label_val.sum(), 'length:', len(label_val))

    #balancing training data
    from utilities import balancing_dataset_with_undersampling
    data_train_balanced, label_train_balanced = balancing_dataset_with_undersampling(data_train, label_train)

    print('train_sum:', label_train_balanced.sum(), 'length:', len(label_train_balanced))
    print('test_sum:', label_test.sum(), 'length:', len(label_test))
    print('val_sum:', label_val.sum(), 'length:', len(label_val))

    train_dataloader = create_dataloader_simple(data_train_balanced, label_train_balanced)
    test_dataloader = create_dataloader_simple(data_test, label_test)
    val_dataloader = create_dataloader_simple(data_val, label_val)

    # using this frame just for one specific dataloader (train)
    # frame = frame_of_multiple_recordings  #np.load("/mnt/MainNAS/temp/Persönliche Verzeichnisse/PS/save.npy", allow_pickle=True)
    # from utilities import create_dataloader
    # windowed_frame_dataloader = create_dataloader(frame)

    from nn_handle import handle_model
    from custom_modles import DenseModel, ViT, CustomResNet, ResidualBlock
    dense_model = DenseModel(in_features=10, hidden_features=20, out_features=2)
    custom_resnet34 = CustomResNet(ResidualBlock, [3, 4, 6, 3])
    transformer_model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1, dim=64, depth=6, heads=8, mlp_dim=128)
    from nn_utilities import compare_models_acc_over_epoch
    compare_models_acc_over_epoch(train_dataloader, val_dataloader, test_dataloader, dense_model, transformer_model)

    print('main finished')





