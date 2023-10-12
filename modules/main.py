import os

os.environ[
    "TF_CPP_MIN_LOG_LEVEL"
] = "3"  # this is to silence some of TF warning messages
import typer
from typing_extensions import Annotated
import pickle
import tensorflow as tf
import keras_tuner as kt
from train_cnn import *
from process_data import *
from validate import plot_all_gamma_results

app = typer.Typer()


@app.command()
def simulate(species: str, demog: str, dist_name: str, num: int):
    # simulate
    pass


@app.command()
def process(
    data_path: str,
    outdir: str,
    prefix: str,
    max_snps: Annotated[int, typer.Option()] = 300,
    afs: Annotated[bool, typer.Option()] = False,
):
    # load data from path
    data = pickle.load(open(f"{data_path}", "rb"))

    if afs:
        data_in, data_out = prep_data(data, afs = True)
    
    else:
        # process data from simulated trees dict to tensor_in and label_out
        data_in, data_out = prep_data(data, max_snps)

    # output to saved path
    # make outdir if necessary
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    pickle.dump(data_in, open(f"{outdir}/{prefix}_tensors", "wb"))
    pickle.dump(data_out, open(f"{outdir}/{prefix}_labels", "wb"))


@app.command()
def train(
    data_path: str,
    data_label_path: str,
    save_model_path: str,
    tune: Annotated[bool, typer.Option()] = False,
    # validate: Annotated[bool, typer.Option()] = False,
    # val_data_path: Annotated[str, typer.Option()] = None,
    # val_label_path: Annotated[str, typer.Option()] = None,
):
    # load training data
    train_in = pickle.load(open(f"{data_path}", "rb"))
    train_out = pickle.load(open(f"{data_label_path}", "rb"))

    if tune:
        model = run_tuning(model_builder, train_in, train_out)

    else:
        # request a model
        input_shape = train_in.shape[1:]
        n_outputs = 2
        model, kwargs = create_dfe_cnn(input_shape, n_outputs)

        # set training data, epochs and validation data
        kwargs.update(
            x=train_in,
            y=train_out,
            batch_size=10,
            epochs=30,
            # validation_data=(test_in, test_out),
        )

        # call fit, including any arguments supplied alongside the model
        callback = callbacks.EarlyStopping(monitor="val_loss", patience=5)
        model.fit(**kwargs, callbacks=[callback])

    # save trained model
    model.save(f"{save_model_path}")


@app.command()
def validate(
    val_data_in_path: str, val_data_out_path: str, model_path: str, results_path: str
):
    # load model
    loaded_model = tf.keras.models.load_model(model_path)
    # load data
    test_in = pickle.load(open(f"{val_data_in_path}", "rb"))
    test_out = pickle.load(open(f"{val_data_out_path}", "rb"))

    plot_all_gamma_results(loaded_model, test_in, test_out, results_path)


@app.command()
def scramble(
    data_in_path: str,
    data_out_path: str,
    scramble_row: Annotated[bool, typer.Option()] = True,
    free_scramble: Annotated[bool, typer.Option()] = False,
    seed: Annotated[int, typer.Option()] = None,
):
    # read in data
    data_in = pickle.load(open(f"{data_in_path}", "rb"))

    # scramble data and store in a new variable
    scramble_data = []
    # iterate through the data array and alter each tensor
    for tensor in data_in:
        scramble_tensor = scramble_snp_tensor(
            tensor, scramble_row=scramble_row, free_scramble=free_scramble, seed=seed
        )
        scramble_data.append(scramble_tensor)

    # save output scrambled data as a numpy array
    pickle.dump(np.array(scramble_data), open(f"{data_out_path}", "wb"))


@app.command()
def partition(
    data_in_path: str,
    data_out_path: str,
    by_row: Annotated[bool, typer.Option()] = False,
    by_peaks: Annotated[bool, typer.Option()] = False,
):
    # read in data
    data_in = pickle.load(open(f"{data_in_path}", "rb"))

    # partition data and store in a new variable
    partition_data = []
    # iterate through the data array and alter each tensor
    for tensor in data_in:
        partition_tensor = partition_snp_tensor(
            tensor, by_row=by_row, by_peaks=by_peaks
        )
        partition_data.append(partition_tensor)

    # save output scrambled data as a numpy array
    pickle.dump(np.array(partition_data), open(f"{data_out_path}", "wb"))


@app.command()
def haperize(
    data_in_path: str,
    data_out_path: str,
    pad_front: Annotated[bool, typer.Option()] = False,
):
    # read in data
    data_in = pickle.load(open(f"{data_in_path}", "rb"))

    # partition data and store in a new variable
    haperize_data = []
    # iterate through the data array and alter each tensor
    for tensor in data_in:
        hap_tensor = make_hap_tensor_from_snp_tensor(tensor, pad_front=pad_front)
        haperize_data.append(hap_tensor)

    # save output scrambled data as a numpy array
    pickle.dump(np.array(haperize_data), open(f"{data_out_path}", "wb"))


if __name__ == "__main__":
    app()
