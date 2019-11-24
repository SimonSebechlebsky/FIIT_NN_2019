# Task
The task we are solving is classification of 120 of dog breeds.
http://vision.stanford.edu/aditya86/ImageNetDogs/

# Docker setup
After cloning the repo, run `$ sh ./build_docker.sh` to build the docker image, 
and `$ sh ./run_docker.sh` or `sh ./run_nvidia_docker.sh` based on if you want to use GPU.

After that execute `$ docker exec -it <container_id> bash` to be able to run scripts.

# Data preprocessing
For data preprocessing run `python prepare_data.py`. This downloads and extracts the images from stanford website, 
splits them into test and train dataset, crops the images according to the annotations, resize them and pads them with
black pixels so all of them are 400x400px. You can find the preprocessed images in `/dog_breeds_data/test` 
and `/dog_breeds_data/train` after running the script

# Model
So far we're using simple Sequential model with 3 Inception layers, which you can find in `model.py`

# Training
To train the model run `python train.py`. You can set these hyperparamaters via commandline arguments:

- learning rate (`-lr, --learning-rate`)
- batch size (`-b, --batch-size`)
- epochs (`-e, --epochs`)
- name of the run (`-r, --run-name`) - logs and model checkpoints will be saved with this name

Accuracy and loss are logged in tensorboard format to `/logs` directory, and best model from the run will be saved in `/best_models`.