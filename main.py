
if __name__ == '__main__':
    print("Welcome to STRIPE")

    # init
    from utilities import create_dataloader_simple, loading_numpy_datasets_for_training, create_directory_structure
    path_to_working_dir = '/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE/debug'
    create_directory_structure(path_to_working_dir)
    import os
    path_to_working_dir_debug = os.path.join(path_to_working_dir, 'debug')

    # loading preprocessed numpy arrays
    data_train_balanced, label_train_balanced, data_test, label_test, data_val, label_val = loading_numpy_datasets_for_training(path_to_working_dir)

    print('train: spikes:', label_train_balanced.sum(), 'total:', len(label_train_balanced))
    print('test: spikes:', label_test.sum(), 'total:', len(label_test))
    print('val: spikes:', label_val.sum(), 'total:', len(label_val))

    # creation of dataloader
    train_dataloader = create_dataloader_simple(data_train_balanced, label_train_balanced)
    test_dataloader = create_dataloader_simple(data_test, label_test)
    val_dataloader = create_dataloader_simple(data_val, label_val)

    from nn_handle import handle_model
    import torch
    from custom_models import DenseModel, ViT, CustomResNet, ResidualBlock, LSTM_Model
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    dense_model = DenseModel(in_features=10, hidden_features=50, out_features=2, device=device)
    custom_resnet34 = CustomResNet(ResidualBlock, [3, 4, 6, 3])
    transformer_model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1, dim=64, depth=6, heads=8, mlp_dim=128, device=device)
    lstm_model = LSTM_Model(in_features=10, hidden_features=64, out_features=2, num_layers=1, device=device)
    from nn_utilities import compare_models_acc_over_epoch
    path_to_save = "/mnt/MainNAS/BioMemsLaborNAS/Projekt_Ordner/STRIPE/Results/LSTM_DENSEV2_ViTV2"
    import torch
    torch.manual_seed(0)
    # Stuck Loss: https://datascience.stackexchange.com/questions/19578/why-my-training-and-validation-loss-is-not-changing
    compare_models_acc_over_epoch(train_dataloader, val_dataloader, test_dataloader, lstm_model, dense_model, transformer_model, epochs=2, learning_rate=0.0001, path_to_save=path_to_save, device=device)

    print('main finished')





