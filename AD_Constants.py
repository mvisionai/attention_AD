import yaml
import  os
main_image_directory = "/HDD/public/data/ADNI_CROSS"  #DATA_EXP  #warp_adni  ADNI_NEW_DATASET
chosen_epi_format=["MP-RAGE","MPRAGE","MP-RAGE_REPEAT","MPRAGE_Repeat","MPRAGE_GRAPPA2"]
strict_match=True
yaml_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),"init_yaml.yml")

complex_data_directory="/home/arvid/Documents/Ben/data/ADNI_NEW_DATASET/Complex_dataset"

model_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)),"trained_models")

pre_trained_gan=["conv_1","conv_2","conv_3","conv_4","conv_5"]




# Parameters for our graph; we'll output images in a 4x4 configuration
# os.kill(os.getpid(), signal.pthread_kill())
nrows = 24
ncols = 4
augment_data=False
#classify_group=["AD",'NC']
#(200,248,248)
#(20,20,20)
#212,260,260
img_shape_tuple=(260,260,260)
img_bilateral_filter=(2480,4960,3)
train_ad_fnames = None
img_channel=1
train_mci_fnames = None
train_nc_fnames = None
source="1.5T"
augment_factor=2
target="3.0T"
training_frac=90/100
classify_group=["AD",'NC']
max_class_value=300

# Index for iterating over images
pic_index = 0
train_dir = os.path.join(main_image_directory, 'train')
validation_dir = os.path.join(main_image_directory, 'validate')

main_validation_dir = os.path.join(main_image_directory, 'validate')

test_dir = os.path.join(main_image_directory, 'test')

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


c_ad_dir = os.path.join(complex_data_directory, 'AD')

# Directory with our training MCI dataset
c_mci_dir = os.path.join(complex_data_directory, 'MCI')

# Directory with our training NC dataset
c_nc_dir = os.path.join(complex_data_directory, 'NC')



# Directory with our training AD dataset
test_ad_dir = os.path.join(test_dir, 'AD')

# Directory with our training MCI dataset
test_mci_dir = os.path.join(test_dir, 'MCI')

# Directory with our training NC dataset
test_nc_dir = os.path.join(test_dir, 'NC')


def yaml_values():

    with open(yaml_file, 'r') as stream:
        try:
            init_args = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        finally:
            stream.close()

    return init_args