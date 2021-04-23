# ml4convection

ml4convection is an end-to-end package that uses U-nets, a type of machine learning, to predict the spatial coverage of thunderstorms (henceforth, "convection") from satellite data.  Inputs to the U-net are a time series of multispectral brightness-temperature maps from the Himawari-8 satellite.  Available spectral bands are band 8 (central wavelength of 6.25 μm), 9 (6.95 μm), 10 (7.35 μm), 11 (8.60 μm), 13 (10.45 μm), 14 (11.20 μm), and 16 (13.30 μm).  Labels are created by applying an echo-classification algorithm to reflectivity maps from four weather radars in Taiwan.  The echo-classification algorithm is a modified version of Storm-labeling in 3 Dimensions (SL3D); you can read about the original version [here](https://doi.org/10.1175/MWR-D-16-0089.1) and all except one of the modifications [here](https://doi.org/10.1175/MWR-D-19-0372.1).  Before the end of April 2021, we (Ryan Lagerquist, Jebb Stewart, Imme Ebert-Uphoff, and Christina Kumler) plan to submit a journal article on this work to *Monthly Weather Review*, titled "Using deep learning to nowcast the spatial coverage of convection from Himawari-8 satellite data".  If you would like to see the manuscript before it is accepted for publication, please contact me at ryan dot lagerquist at noaa dot gov.

Detailed documentation (each file and method) can be found... TODO.

Documentation for important scripts, which you can run from the Unix command line, is provided below.  Please note that this package is not intended for Windows and I provide no support for Windows.  Also, though I have included many unit tests (every file ending in `_test.py`), I provide no guarantee that the code is free of bugs.  If you choose to use this package, you do so at your own risk.

# Setting up a U-net

Before training a U-net (or any model in Keras), you must set up the model.  "Setting up" includes four things: choosing the architecture, choosing the loss function, choosing the metrics (evaluation scores other than the loss function, which, in addition to the loss function, are used to monitor the model's performance after each training epoch), and compiling the model.  For each lead time (0, 30, 60, 90, 120 minutes), I have created a script that sets up the **chosen** U-net (based on the hyperparameter experiment presented in the *Monthly Weather Review* paper).  These scripts, which you can find in the directory `ml4convection/scripts`, are as follows:

 - `make_best_architecture_0minutes.py`
 - `make_best_architecture_30minutes.py`
 - `make_best_architecture_60minutes.py`
 - `make_best_architecture_90minutes.py`
 - `make_best_architecture_120minutes.py`

Each script will set up the model (`model_object`) and print the model's architecture in a text-only flow chart to the command window, using the command `model_object.summary()`.  If you want to save the model (which is still untrained) to a file, add the following command, replacing `output_path` with the desired file name.

`model_object.save(filepath=output_path, overwrite=True, include_optimizer=True)`

# Training the U-net

Once you have set up a U-net, you can train the U-net, using the script `train_neural_net.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `train_neural_net.py` from a Unix terminal.  For some input arguments I have suggested a default (where I include an actual value), and for some I have not.  In this case, the lead time is 3600 seconds (60 minutes) and the lag times are 0 and 1200 and 2400 seconds (0 and 20 and 40 minutes).  Thus, if the forecast issue time is 1200 UTC, the valid time will be 1300 UTC, while the predictors (brightness-temperature maps) will come from 1120 and 1140 and 1200 UTC.

`python train_neural_net.py \`
    `    --training_predictor_dir_name="your directory name here" \`
    `    --training_target_dir_name="your directory name here" \`
    `    --validn_predictor_dir_name="your directory name here" \`
    `    --validn_target_dir_name="your directory name here" \`
    `    --input_model_file_name="file with untrained, but set-up, model" \`
    `    --output_model_dir_name="where you want trained model to be saved" \`
    `    --band_numbers 8 9 10 11 13 14 16 \`
    `    --lead_time_seconds=3600 \`
    `    --lag_times_seconds 0 1200 2400 \`
    `    --include_time_dimension=0 \`
    `    --first_training_date_string="20160101" \`
    `    --last_training_date_string="20161224" \`
    `    --first_validn_date_string="20170101" \`
    `    --last_validn_date_string="20171224" \`
    `    --normalize=1 \`
    `    --uniformize=1 \`
    `    --add_coords=0 \`
    `    --num_examples_per_batch=60 \`
    `    --max_examples_per_day_in_batch=8 \`
    `    --use_partial_grids=1 \`
    `    --omit_north_radar=1 \`
    `    --num_epochs=1000 \`
    `    --num_training_batches_per_epoch=64 \`
    `    --num_validn_batches_per_epoch=32 \`
    `    --plateau_lr_multiplier=0.6`
