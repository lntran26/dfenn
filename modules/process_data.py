import numpy as np
from matplotlib import pyplot as plt
import allel


def ts_to_tensor(ts):
    """
    Input: a simulated tree sequence
    max_snps: cut off snps window size
    Output: a tensor representing non-syn, syn SNP vs ancestral state
    """

    # get haplotype matrix, which has 0 for ancestral and 1 for derived
    haps = ts.genotype_matrix().T

    # get selection coefficients for all snp positions (columns)
    # selection_coeffs = [
    #     ext.selection_coeff_from_mutation(ts, mut) for mut in ts.mutations()
    # ]
    selection_coeffs = []

    for mut in ts.mutations():
        selection_coeff = sum(
            [m.get("selection_coeff") for m in mut.metadata["mutation_list"]]
        )
        selection_coeffs.append(selection_coeff)
    # currently not using position
    # positions = [variant.site.position for variant in ts.variants()]

    # get position in haps that is neutral
    neu_positions = []
    # for i, (_, s) in enumerate(zip(positions, selection_coeffs)):
    for i, s in enumerate(selection_coeffs):
        if s == 0:
            neu_positions.append(i)

    # save position of fixed SNPs to be removed here
    fixed_positions = []
    # make two dims with the same shape and values as the haplotype matrix
    # ancestral entries of 0 will stay 0 in both dims
    # dim_1: 1 only if a position is non-syn SNP (selection coeff != 0), so need to
    # set the 1s at neutral positions to 0
    # dim_2: 1 only if a position is syn SNP (selection coeff is 0 and is a 1 in haps),
    # so need to set the 1s at non-neutral positions to 0
    dim_1, dim_2 = haps.copy(), haps.copy()

    # iterate through all SNP positions (columns) in matrix
    for idx in range(haps.shape[1]):
        # get position in haps that have fixed snp (1s in the whole column)
        if np.all(haps[:, idx] == 1):
            fixed_positions.append(idx)
        # if a neutral position, then set dim_1 value to 0 if it's currently 1 (not ancestral)
        if idx in neu_positions:
            dim_1[:, idx][np.where(haps[:, idx] == 1)] = 0
        # if not neutral position (non-syn SNP), then set dim_2 value to 0 if it's currently 1 (not ancestral)
        else:
            dim_2[:, idx][np.where(haps[:, idx] == 1)] = 0

    # stacking dim_1 and dim_2 together to make snp tensor
    snp_tensor = np.stack((dim_1, dim_2), axis=-1)

    # drop columns that are all 1s (fixed)
    new_tensor = np.delete(snp_tensor, fixed_positions, 1)
    # if have position vector in the future will have to
    # remove the fixed positions accordingly as well

    return new_tensor


# To-do: vcf format to long tensor for infer.py


def split_long_tensor(tensor, max_snps: int):
    """
    Breaking a long tensor into more reasonable windows
    Pad a window with 0s if smalle that max_snps
    Input: a single snp tensor to be split up
    Output: a list of broken down tensors, with padding for tensors
    smaller than max_snps
    """

    def _split_tensor(tensor, max_snps: int):
        """Splitting tensor into equal sized chunks based on max_snps"""
        for i in range(0, tensor.shape[1], 300):
            yield tensor[:, i : i + max_snps, :]

    cropped_tensor_list = list(_split_tensor(tensor, max_snps))

    # padding
    last_tensor = cropped_tensor_list[-1]
    if last_tensor.shape[1] < max_snps:
        # padd to the max snp size
        pad_width = max_snps - last_tensor.shape[1]
        padded_tensor = np.pad(
            last_tensor,
            ((0, 0), (0, pad_width), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        # replace this tensor in the list and return
        cropped_tensor_list[-1] = padded_tensor

    return cropped_tensor_list


def process_labels(param: tuple):
    # regardless of tuple length, right now only concerns the first two values
    # there params[0] is the mean and params[1] is the shape

    # calculate scale param from mean and shape, convert to log scale
    scale_log = np.log10(12378 * param[0] / param[1])
    # convert shape to log scale and use absolute value
    # remove absolute value conversion for now

    return (scale_log, abs(np.log10(param[1])))


def scramble_snp_tensor(snp_tensor, scramble_row=True, free_scramble=False, seed=None):
    """if scramble_row is False: will only scramble columns at the first level"""

    if seed is not None:
        np.random.seed(seed)

    # Create a copy of the image tensor to avoid mutation
    scrambled_tensor = np.copy(snp_tensor)

    # Shuffle along the columns
    np.random.shuffle(scrambled_tensor.transpose(1, 0, 2))

    # Shuffle within the columns
    if scramble_row:
        np.random.shuffle(scrambled_tensor)

    # Shuffle everything
    if free_scramble:
        h, w, c = scrambled_tensor.shape
        scrambled_tensor = scrambled_tensor.reshape(h * w, c)
        np.random.shuffle(scrambled_tensor)
        scrambled_tensor = scrambled_tensor.reshape(h, w, c)

    return scrambled_tensor


def partition_snp_tensor(snp_tensor, by_row=False, by_peaks=False):
    """Partition tensors into blocks of red, green, black"""
    if by_peaks:
        partitioned_tensor = np.copy(snp_tensor)
        partitioned_tensor.sort(axis=0)

    else:
        "put all red and green on either side of the tensor image"
        row, col, channel = snp_tensor.shape
        # count total # or red, green, black
        # red: 1s in the front, green: 1s in the second, black: not red or green
        num_red = np.count_nonzero(snp_tensor[:, :, 0])
        num_green = np.count_nonzero(snp_tensor[:, :, 1])
        # num_black = row * col - (num_red + num_green)

        # create a new tensor from these counts
        red_channel = np.concatenate([np.ones(num_red), np.zeros(row * col - num_red)])
        green_channel = np.concatenate(
            [np.zeros(row * col - num_green), np.ones(num_green)]
        )
        new_tensor = np.stack((red_channel, green_channel), axis=-1)

        # reshape from 1D to 2D, use transpose trick to organize red and green
        # pixels into columns instead of rows
        partitioned_tensor = (
            new_tensor.reshape(snp_tensor.shape)
            if by_row
            else new_tensor.reshape(col, row, channel).transpose(1, 0, 2)
        )

    return partitioned_tensor


def make_hap_tensor_from_snp_tensor(snp_tensor, pad_front=False):
    """Input is a snp_tensor with 2 dim, each for non-syn and syn snp
    Now we combine it to just a hap dim, and pad the other with 0s"""

    hap_tensor = snp_tensor.copy()
    # change dim_1 of the copied tensor to be 1 also for where there is 1 in dim_2
    dim_1 = hap_tensor[:, :, 0]
    dim_2 = hap_tensor[:, :, 1]
    dim_1[np.where(dim_2 == 1)] = 1
    # dim_1 is now the same as haps matrix

    # make a dim filled with 0s
    row, col, _ = snp_tensor.shape
    blank_dim = np.zeros((row, col))

    # stack the haps dim with a blank dim together to create a new haps-like snp tensor
    hap_tensor = (
        np.stack((blank_dim, dim_1), axis=-1)
        if pad_front
        else np.stack((dim_1, blank_dim), axis=-1)
    )

    return hap_tensor


def prep_data(data: dict, max_snps: int = 300, afs: bool = False):
    """Input:
    data is a dictionary with simulated tree sequence and label values
    from SLiM simulation
    max_snps: window to crop long tensor
    Output: data_in (tensor) and data_out (label) for training and testing"""
    X_input, y_label = [], []

    for param in data:
        tensor = ts_to_tensor(data[param])
        if afs:
            afs_tensor = snp_tensor_to_afs_tensor(tensor)
            X_input.append(afs_tensor)
            y_label.append(process_labels(param))
        else:
            tensor_list = split_long_tensor(tensor, max_snps)
            for tensor in tensor_list:
                X_input.append(tensor)
                y_label.append(process_labels(param))

    return np.array(X_input), np.array(y_label)


def visualize_snp_tensor(snp_tensor, pixel_size=5):
    """Expecting input is a 3d snp tensor, where the shape is (h, w, 2)"""

    # Get SNP tensor size
    row, col, _ = snp_tensor.shape

    # Add a 3rd dimension with zeros (will be the position vector once implemented)
    dim_3 = np.zeros((row, col))
    img = np.stack((snp_tensor[:, :, 0], snp_tensor[:, :, 1], dim_3), axis=-1).astype(
        np.uint8
    )

    # convert pixel to "plasma" color scheme
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i, j, :] == np.array([1, 0, 0])).all():
                img[i, j, :] = np.array([248, 149, 64]) # orange
            elif (img[i, j, :] == np.array([0, 1, 0])).all():
                img[i, j, :] = np.array([126, 3, 168]) # purple
            elif (img[i, j, :] == np.array([0, 0, 0])).all():
                img[i, j, :] = np.array([192,192,192]) # black to silver
    # Get image size
    height, width, channels = img.shape

    # Calculate the figure size based on pixel size and image dimensions
    fig_width = pixel_size * (width / float(height))
    fig_height = pixel_size

    # Set the DPI for the figure based on desired pixel size
    dpi = 10 * pixel_size

    # Create a new figure with the adjusted size and DPI
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    # Display the image array as a grayscale image
    ax.imshow(img)

    # Remove the axis labels and ticks
    ax.axis("off")

    # Show the plot
    plt.show()


def snp_tensor_to_afs_tensor(snp_tensor):
    """Processing a snp tensor into a synonymous and a non-synonymous SFS"""
    # make a copy of the tensor
    tensor_copied = snp_tensor.copy()

    # get the non-syn and syn hap matrix
    non_syn_hap = tensor_copied[:, :, 0].T
    syn_hap = tensor_copied[:, :, 1].T
    
    def hap_to_fs(hap, sample_size):
        # convert from hap to allel fs
        fs = allel.sfs(allel.HaplotypeArray(hap).count_alleles()[:, 1])[1:]
        
        # some syn and non_syn array may not have the same shape due to some entry
        # missing (zero count)
        # need to preprocess before being able to concatenate:
        # patch 0s to end of fs with dimension mismatch
        if len(fs) < sample_size - 1:
            fs = np.concatenate((fs, np.zeros(sample_size - 1 - len(fs))))
            
        return fs
    
    # get FS from hap
    fs_non_syn = hap_to_fs(non_syn_hap, non_syn_hap.shape[1])
    fs_syn = hap_to_fs(syn_hap, syn_hap.shape[1])
 
    # stacking the two FS instead of concatenating like before
    afs_tensor = np.stack((fs_non_syn, fs_syn))
    
    return afs_tensor / afs_tensor.sum() # normalize tensor