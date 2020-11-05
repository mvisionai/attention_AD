import os

class HyperParams:
    """Hyper-Parameters"""
    # data path

    main_image_directory = "/home/ben/PycharmProjects/attention_AD"
    model_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trained_models")
    chosen_epi_format = ["MP-RAGE", "MPRAGE", "MP-RAGE_REPEAT", "MPRAGE_Repeat", "MPRAGE_GRAPPA2"]
    strict_match = True
    yaml_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "init_yaml.yml")

    pre_trained_gan = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    # Parameters for our graph; we'll output images in a 4x4 configuration
    # os.kill(os.getpid(), signal.pthread_kill())
    autoenc_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "aee_summary")

    augment_data = False

    # (20,20,20)
    img_shape_tuple = (150, 150, 150)
    train_ad_fnames = None
    img_channel = 3
    train_mci_fnames = None
    train_nc_fnames = None
    source = "1.5T"
    augment_factor = 2
    target = "3.0T"
    training_frac = 90 / 100


    # Index for iterating over images
    pic_index = 0
    train_dir = os.path.join(main_image_directory, 'train')
    validation_dir = os.path.join(main_image_directory, 'validate')

    main_validation_dir = os.path.join(main_image_directory, 'validate')

    # Directory with our training AD dataset
    train_ad_dir = os.path.join(train_dir, 'AD')

    # Directory with our training MCI dataset
    train_mci_dir = os.path.join(train_dir, 'MCI')

    # Directory with our training NC dataset
    train_nc_dir = os.path.join(train_dir, 'NC')

    # Directory with our validation AD dataset
    validation_ad_dir = os.path.join(validation_dir, 'AD')

    # Directory with our validation MCI dataset
    validation_mci_dir = os.path.join(validation_dir, 'MCI')

    # Directory with our validation NC dataset
    validation_nc_dir = os.path.join(validation_dir, 'NC')

    main_validation_ad_dir = os.path.join(main_validation_dir, 'AD')

    # Directory with our validation MCI dataset
    main_validation_mci_dir = os.path.join(main_validation_dir, 'MCI')

    # Directory with our validation NC dataset
    main_validation_nc_dir = os.path.join(main_validation_dir, 'NC')
    SAVED_PATH = model_directory + "/model.ckpt"

    align_plots = os.path.join(model_directory,'aligned_images')
    latent_plots = os.path.join(align_plots, 'latent')
    recons_plots = os.path.join(align_plots, 'reconstruct')

    activation_dir = os.path.join(model_directory, 'trained')

    summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "summaries")
    train_summary = os.path.join(summary_path, "train")
    test_summary = os.path.join(summary_path, "test")
    restore_path= os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models")
    
    fold_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fold_csv")

    # dataset
    target_dataset = "ADNI"

    # setting
    RANDOM_STATE = 42
    NUM_EPOCHS = 150
    BATCH_SIZE =4
    VALID_BATCH_SIZE = 4
    DROP_OUT = 0.90


