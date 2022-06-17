# FSL
Frequency Spectrum Learning for Cross-domain Semantic Segmentation

## Abstract
xxx.

## Preparation

### Pre-requisites
* Python 3.7
* Pytorch >= 1.8.1
* CUDA 9.0 or higher

### Installation
0. Clone the repo:
```bash
$ git clone https://github.com/seabearlmx/FSL
$ cd FSL
```

### Datasets
By default, the datasets are put in ```<root_dir>/datasets```. 

* **GTA5**: Please follow the instructions [here](https://download.visinf.tu-darmstadt.de/data/from_games/) to download images and semantic segmentation annotations. The GTA5 dataset directory should have this basic structure:
```bash
<root_dir>/datasets/GTA5/                               % GTA dataset root
<root_dir>/datasets/GTA5/images/                        % GTA images
<root_dir>/datasets/GTA5/labels/                        % Semantic segmentation labels
...
```

* **Cityscapes**: Please follow the instructions in [Cityscape](https://www.cityscapes-dataset.com/) to download the images and validation ground-truths. The Cityscapes dataset directory should have this basic structure:
```bash
<root_dir>/datasets/Cityscapes/                         % Cityscapes dataset root
<root_dir>/datasets/Cityscapes/leftImg8bit              % Cityscapes images
<root_dir>/datasets/Cityscapes/leftImg8bit/val
<root_dir>/datasets/Cityscapes/gtFine                   % Semantic segmentation labels
<root_dir>/datasets/Cityscapes/gtFine/val
...
```

### Pre-trained models
Pre-trained models can be downloaded [here](https://github.com/seabearlmx/FSL/releases) and put in ```<root_dir>/FSL/pretrained_models```

## Running the code
Please follow the [here](https://github.com/seabearlmx/FSL/releases) to download model.

For evaluation, execute:
```bash
$ cd <root_dir>/padan
$ python evaluation_multi.py --restore-opt1="../checkpoints/FSL/gta2city_deeplab/gta2city"
```

### Training
For the experiments done in the paper, we used pytorch 1.8.1 and CUDA 9.0. To ensure reproduction, the random seed has been fixed in the code. Still, you may need to train a few times to reach the comparable performance.

By default, logs and snapshots are stored in ```<root_dir>/checkpoints/FSL``` with this structure:
```bash
<root_dir>/checkpoints/FSL/gta2city_deeplab
```

To train FSL:
```bash
$ cd <root_dir>/FSL
$ python train.py

```

### Testing
To test FSL:
```bash
$ cd <root_dir>/FSL
$ python evaluation_multi.py --restore-opt1="../checkpoints/FSL/gta2city_deeplab/gta2city"
```

## Acknowledgements
This codebase is heavily borrowed from [FDA]([https://github.com/wasidennis/AdaptSegNet](https://github.com/YanchaoYang/FDA)).

## License
FSL is released under the [MIT license](./LICENSE).
