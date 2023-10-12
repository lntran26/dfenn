# User Guide
Create dedicated environment and install dependencies:
```console
conda env create -f environment.yml
conda activate dfe-cnn
```

To use the code: 
```console
cd modules
python main.py --help
```
Output should look like this:
```console
 Usage: main.py [OPTIONS] COMMAND [ARGS]...                                                                                                                                         
                                                                                                                                                                                    
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                                                          │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                                                   │
│ --help                        Show this message and exit.                                                                                                                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ haperize                                                                                                                                                                         │
│ partition                                                                                                                                                                        │
│ process                                                                                                                                                                          │
│ scramble                                                                                                                                                                         │
│ simulate                                                                                                                                                                         │
│ train                                                                                                                                                                            │
│ validate                                                                                                                                                                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```
### Notes:
`simulate` command is currently empty. Refer to old commits for simulate.py script, or use simulated example data.
### Example commands:
Command for scrambling data
```console
python main.py scramble ../data/processed_data/const_demog/test/tensors ../data/processed_data/const_demog/test/tensors_scramble
python main.py scramble ../data/processed_data/const_demog/test/tensors ../data/processed_data/const_demog/test/tensors_scramble_free --free-scramble
```
Command for partitioning by row
```console
python main.py partition ../data/processed_data/const_demog/test/tensors ../data/processed_data/const_demog/test/tensors_partition_by_row --by-row
```
	
Command for processing data
```console
# from tree simulated data dict from two epoch demography
python main.py process ../data/trees/test_data_1B08_varied_scale_2.pickle ../data/processed_data/two_epoch/test/
# process train and test data to "AFS-like tensor" (entries are frequency spectrum values)
python main.py process ../data/trees/test_data.pickle ../data/processed_data/const_demog/test/ afs --afs	
python main.py process ../data/trees/train_data.pickle ../data/processed_data/const_demog/train/ afs --afs
```

Command for haperizing (I made this up xD) the tensor (haperize means convert to just one haplotype, all non-syn or syn only)
```console
python main.py haperize ../data/processed_data/const_demog/test/tensors ../data/processed_data/const_demog/test/tensors_haps_back
python main.py haperize ../data/processed_data/const_demog/test/tensors ../data/processed_data/const_demog/test/tensors_haps_front --pad-front
```
Command to train CNN
```console
python main.py train ../data/processed_data/const_demog/train/tensors_haps_pad_back  ../data/processed_data/const_demog/train/labels ../results/const_demog_trained_300_haps_pad_back
```
Command to validate the trained CNN (test on simulated test data)
```console
python main.py validate ../data/processed_data/const_demog/test/tensors_haps_pad_back  ../data/processed_data/const_demog/test/labels ../results/const_demog_trained_300_haps_pad_back ../results/plots/test_const_demog_haps_pad_back
# validate using swapped haps data
python main.py validate ../data/processed_data/const_demog/test/tensors_haps_pad_front  ../data/processed_data/const_demog/test/labels ../results/const_demog_trained_300_haps_pad_back ../results/plots/test_const_demog_haps_pad_back_test_on_haps_pad_front
# validate using free scramble haps
python main.py validate ../data/processed_data/const_demog/test/tensors_haps_pad_back_scramble_free  ../data/processed_data/const_demog/test/labels ../results/const_demog_trained_300_haps_pad_back ../results/plots/test_const_demog_haps_pad_back_test_on_haps_pad_back_scramble_free
```
