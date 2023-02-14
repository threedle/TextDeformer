# TextDeformer

## Installation

    conda create -y -n TextDeformer python=3.9
    conda activate TextDeformer
    pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
    conda install -y -c conda-forge igl
    pip install -r requirements.txt

## Usage
**NOTE:** This code **requires** a GPU to run.

``main.py`` is the primary script to use. You may pass arguments using the ``--config`` flag, which takes the path to a ``.yml`` file. See ``example_config.yml`` for an example. Alternatively, you may pass command line arguments manually, which override the arguments provided by the config file. Below, we provide example usage:
    
    # Use all arguments provided by the example config
    python main.py --config example_config.yml

    # Change the optimized mesh to hand.obj, change the base and target text prompts
    python main.py --config example_config.yml --mesh meshes/hand.obj --text_prompt 'an octupus' --base_text_prompt 'a hand'

    # Now, increase the batch size, learning rate, and the training resolution
    python main.py --config example_config.yml --mesh meshes/hand.obj --text_prompt 'an octopus' \
    --base_text_prompt 'a hand' --batch_size 50 --lr 0.005 --train_res 1024