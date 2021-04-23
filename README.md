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

```
python train_neural_net.py \
    --training_predictor_dir_name="your directory name here" \
    --training_target_dir_name="your directory name here" \
    --validn_predictor_dir_name="your directory name here" \
    --validn_target_dir_name="your directory name here" \
    --input_model_file_name="file with untrained, but set-up, model" \
    --output_model_dir_name="where you want trained model to be saved" \
    --band_numbers 8 9 10 11 13 14 16 \
    --lead_time_seconds=3600 \
    --lag_times_seconds 0 1200 2400 \
    --include_time_dimension=0 \
    --first_training_date_string="20160101" \
    --last_training_date_string="20161224" \
    --first_validn_date_string="20170101" \
    --last_validn_date_string="20171224" \
    --normalize=1 \
    --uniformize=1 \
    --add_coords=0 \
    --num_examples_per_batch=60 \
    --max_examples_per_day_in_batch=8 \
    --use_partial_grids=1 \
    --omit_north_radar=1 \
    --num_epochs=1000 \
    --num_training_batches_per_epoch=64 \
    --num_validn_batches_per_epoch=32 \
    --plateau_lr_multiplier=0.6
```

More details on the input arguments are provided below.

 - `training_predictor_dir_name` is a string, naming the directory with predictor files (containing brightness-temperature maps).  Files therein will be found by `example_io.find_predictor_file` and read by `example_io.read_predictor_file`, where `example_io.py` is in the directory `ml4convection/io`.  `example_io.find_predictor_file` will only look for files named like `[training_predictor_dir_name]/[yyyy]/predictors_[yyyymmdd]_radar[k].nc` and `[training_predictor_dir_name]/[yyyy]/predictors_[yyyymmdd]_radar[k].nc.gz`, where `[yyyy]` is the 4-digit year; `[yyyymmdd]` is the date; and `[k]` is the radar number, ranging from 1-3.  An example of a good file name, assuming the top-level directory is `foo`, is `foo/2016/predictors_20160101_radar1.nc`.
 - `training_target_dir_name` is a string, naming the directory with target files (containing labels, which are binary convection masks, containing 0 or 1 at each grid point).  Files therein will be found by `example_io.find_target_file` and read by `example_io.read_target_file`.  `example_io.find_target_file` will only look for files named like `[training_target_dir_name]/[yyyy]/targets_[yyyymmdd]_radar[k].nc` and `[training_target_dir_name]/[yyyy]/targets_[yyyymmdd]_radar[k].nc.gz`.
 - `validn_predictor_dir_name` is the same as `training_predictor_dir_name` but for validation data.
 - `validn_target_dir_name` is the same as `training_target_dir_name` but for validation data.
 - `input_model_file_name` is a string, containing the full path to the untrained but set-up model.  This file will be read by `neural_net.read_model`, where `neural_net.py` is in the directory `ml4convection/machine_learning`.
 - `output_model_dir_name` is a string, naming the output directory.  The trained model will be saved here.
 - `band_numbers` is a list of band numbers to use in the predictors.  I suggest using all bands (8, 9, 10, 11, 13, 14, 16).
 - `lead_time_seconds` is the lead time in seconds.
 - `lag_times_seconds` is a list of lag times for the predictors.
 - `include_time_dimension` is a Boolean flag (0 or 1), determining whether or not the spectral bands and lag times will be represented on separate axes.  For vanilla U-nets, always make this 0; for temporal U-nets and U-net++ models, always make this 1.
 - `first_training_date_string` is a string containing the first date in the training period, in the format `yyyymmdd`.
 - `last_training_date_string` is a string containing the last date in the training period, in the format `yyyymmdd`.
 - `first_validn_date_string` is the same as `first_training_date_string` but for validation data.
 - `last_validn_date_string` is the same as `last_training_date_string` but for validation data.
 - `normalize` is a Boolean flag (0 or 1), determining whether or not predictors will be normalized to *z*-scores.  Please always make this 1.
 - `uniformize` is a Boolean flag (0 or 1), determining whether or not predictors will be uniformized before normalization.  Please always make this 1.
 - `add_coords` is a Boolean flag (0 or 1), determining whether or not latitude-longitude coordinates will be used as predictors.  Please always make this 0.
 - `num_examples_per_batch` is the number of examples per training or validation batch.  Based on hyperparameter experiments presented in the *Monthly Weather Review* paper, I suggest making this 60.
 - `max_examples_per_day_in_batch` is the maximum number of examples in a given batch that can come from the same day.  The smaller you make this, the less temporal autocorrelation there will be in each batch.  However, smaller numbers also increase the training time, because they increase the number of daily files from which data must be read.
 - `use_partial_grids` is a Boolean flag (0 or 1), determining whether the model will be trained on the full Himawari-8 grid or partial radar-centered grids.  Please always make this 1.
 - `omit_north_radar` is a Boolean flag (0 or 1), determining whether or not the northernmost radar in Taiwan will be omitted from training.  Please always make this 1.
 - `num_epochs` is the number of training epochs.  I suggest making this 1000, as early stopping always occurs before 1000 epochs.
 - `num_training_batches_per_epoch` is the number of training batches per epoch.  I suggest making this 64 (so that 64 training batches per epoch are used to update model weights), but you might find a better value.
 - `num_validn_batches_per_epoch` is the number of validation batches per epoch.  I suggest making this 32 (so that 32 validation batches per epoch are used to compute metrics other than the loss function).
 - `plateau_lr_multiplier` is a floating-point value ranging from 0 to 1 (non-inclusive).  During training, if the validation loss has not improved over the last 10 epochs (*i.e.*, validation performance has reached a "plateau"), the learning rate will be multiplied by this value.

# Applying the trained U-net

Once you have trained the U-net, you can apply it to make predictions on new data.  This is called the "inference phase," as opposed to the "training phase".  You can do this with the script `apply_neural_net.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `apply_neural_net.py` from a Unix terminal.

```
python apply_neural_net.py \
    --input_model_file_name="file with trained model" \
    --input_predictor_dir_name="your directory name here" \
    --input_target_dir_name="your directory name here" \
    --apply_to_full_grids=[0 or 1] \
    --overlap_size_px=90 \
    --first_valid_date_string="date in format yyyymmdd" \
    --last_valid_date_string="date in format yyyymmdd" \
    --output_dir_name="your directory name here" \
```

More details on the input arguments are provided below.

 - `input_model_file_name` is a string, containing the full path to the trained model.  This file will be read by `neural_net.read_model`.
 - `input_predictor_dir_name` is a string, naming the directory with predictor files.  Files therein will be found by `example_io.find_predictor_file` and read by `example_io.read_predictor_file`, as for the input arguments `training_predictor_dir_name` and `validn_predictor_dir_name` to `train_neural_net.py`.
 - `input_target_dir_name` is a string, naming the directory with target files.  Files therein will be found by `example_io.find_target_file` and read by `example_io.read_target_file`, as for the input arguments `training_target_dir_name` and `validn_target_dir_name` to `train_neural_net.py`.
 - `apply_to_full_grids` is a Boolean flag (0 or 1), determining whether the model will be applied to full or partial grids.  If the model was trained on full grids, `apply_to_full_grids` will be ignored and the model will be applied to full grids regardless.  Thus, `apply_to_full_grids` is used only if the model was trained on partial (radar-centered) grids.
 - `overlap_size_px` is an integer, determining the overlap size (in pixels) between adjacent partial grids.  This argument is used only if the model was trained on partial grids and `apply_to_full_grids` is 1.  I suggest making `overlap_size_px` 90.
 - `first_valid_date_string` and `last_valid_date_string` are the first and last days in the inference period.  In other words, the model will be used to make the predictions for all days from `first_valid_date_string` to `last_valid_date_string`.
 - `output_dir_name` is a string, naming the output directory.  Predictions will be saved here.

# Plotting predictions and inputs (radar/satellite data)

Once you have run `apply_neural_net.py` to make predictions, you can plot the predictions with the script `plot_predictions.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `plot_predictions.py` from a Unix terminal.

```
python plot_predictions.py \
    --input_prediction_dir_name="your directory name here" \
    --first_date_string="date in format yyyymmdd" \
    --last_date_string="date in format yyyymmdd" \
    --use_partial_grids=[0 or 1] \
    --smoothing_radius_px=2 \
    --daily_times_seconds 0 7200 14400 21600 28800 36000 43200 50400 57600 64800 72000 79200 \
    --plot_deterministic=0 \
    --probability_threshold=-1 \
    --output_dir_name="your directory name here" \
```

More details on the input arguments are provided below.

 - `input_prediction_dir_name` is a string, naming the directory with prediction files.  Files therein will be found by `prediction_io.find_file` and read by `prediction_io.read_file`, where `prediction_io.py` is in the directory `ml4convection/io`.  `prediction_io.find_file` will only look for files named like `[input_prediction_dir_name]/[yyyy]/predictions_[yyyymmdd]_radar[k].nc` and `[input_prediction_dir_name]/[yyyy]/predictions_[yyyymmdd]_radar[k].nc.gz`, where `[yyyy]` is the 4-digit year; `[yyyymmdd]` is the date; and `[k]` is the radar number, ranging from 1-3.  An example of a good file name, assuming the top-level directory is `foo`, is `foo/2016/predictions_20160101_radar1.nc`.
 - `first_date_string` and `first_date_string` are the first and last days to plot.  In other words, the model will be used to plot predictions for all days from `first_date_string` to `last_valid_date_string`.
 - `use_partial_grids` is a Boolean flag (0 or 1), indicating whether you want to plot predictions on the full Himawari-8 grid or partial (radar-centered) grids.  If `use_partial_grids` is 1, `plot_predictions.py` will plot partial grids centered on every radar (but in separate plots, so you will get one plot per time step per radar).
 - `smoothing_radius_px`, used for full-grid predictions (if `use_partial_grids` is 0), is the *e*-folding radius for Gaussian smoothing (pixels).  Each probability field will be filtered by this amount before plotting.  Smoothing is useful for full-grid predictions created from a model trained on partial grids.  In this case the full-grid predictions are created by sliding the partial grid to various "windows" inside the full grid, and sometimes there is a sharp cutoff at the boundary between two adjacent windows.
 - `daily_times_seconds` is a list of daily times at which to plot predictions.  The data are available at 10-minute (600-second) time steps, but you may not want to plot the predictions every 10 minutes.  In the above code example, the list of times provided will force the script to plot predictions at {0000, 0200, 0400, 0600, 0800, 1000, 1200, 1400, 1600, 1800, 2000, 2200} UTC every day.
 - `plot_deterministic` is a Boolean flag (0 or 1), indicating whether you want to plot deterministic (binary) or probabilistic predictions.
 - `probability_threshold` is the probability threshold (ranging from 0 to 1) used to convert probabilistic to binary predictions.  This argument is ignored if `plot_deterministic` is 0, which is why I make it -1 in the above code example.
 - `output_dir_name` is a string, naming the output directory.  Plots will be saved here as JPEG images.

If you want to just plot satellite data, use the script `plot_satellite.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `plot_satellite.py` from a Unix terminal.

```
python plot_satellite.py \
    --input_satellite_dir_name="your directory name here" \
    --first_date_string="date in format yyyymmdd" \
    --last_date_string="date in format yyyymmdd" \
    --band_numbers 8 9 10 11 13 14 16 \
    --daily_times_seconds 0 7200 14400 21600 28800 36000 43200 50400 57600 64800 72000 79200 \
    --output_dir_name="your directory name here" \
```

More details on the input arguments are provided below.

 - `input_satellite_dir_name` is a string, naming the directory with prediction files.  Files therein will be found by `satellite_io.find_file` and read by `satellite_io.read_file`, where `satellite_io.py` is in the directory `ml4convection/io`.  `satellite_io.find_file` will only look for files named like `[input_satellite_dir_name]/[yyyy]/satellite_[yyyymmdd].nc` and `[input_satellite_dir_name]/[yyyy]/satellite_[yyyymmdd].nc.gz`, where `[yyyy]` is the 4-digit year and `[yyyymmdd]` is the date.  An example of a good file name, assuming the top-level directory is `foo`, is `foo/2016/satellite_20160101.nc`.
 - `first_date_string` and `first_date_string` are the first and last days to plot.  In other words, the model will be used to plot brightness-temperature maps for all days from `first_date_string` to `last_valid_date_string`.
 - `band_numbers` is a list of band numbers to plot.  `plot_satellite.py` will create one image per time step per band.
 - `daily_times_seconds` is a list of daily times at which to plot brightness-temperature maps, same as the input for `plot_predictions.py`.
 - `output_dir_name` is a string, naming the output directory.  Plots will be saved here as JPEG images.
