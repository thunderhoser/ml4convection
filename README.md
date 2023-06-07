# ml4convection

ml4convection is an end-to-end package that uses U-nets, a type of machine learning, to predict the spatial coverage of thunderstorms (henceforth, "convection") from satellite data.  Inputs to the U-net are a time series of multispectral brightness-temperature maps from the Himawari-8 satellite.  Available spectral bands are band 8 (central wavelength of 6.25 μm), 9 (6.95 μm), 10 (7.35 μm), 11 (8.60 μm), 13 (10.45 μm), 14 (11.20 μm), and 16 (13.30 μm).  Labels are created by applying an echo-classification algorithm to reflectivity maps from four weather radars in Taiwan.  The echo-classification algorithm is a modified version of Storm-labeling in 3 Dimensions (SL3D); you can read about the original version [here](https://doi.org/10.1175/MWR-D-16-0089.1) and all except one of the modifications [here](https://doi.org/10.1175/MWR-D-19-0372.1).  A journal article on this has been published in *Monthly Weather Review*, titled "Using deep learning to nowcast the spatial coverage of convection from Himawari-8 satellite data".  You can find it [here](https://doi.org/10.1175/MWR-D-21-0096.1).

Documentation for important scripts, which you can run from the Unix command line, is provided below.  Please note that this package is not intended for Windows and I provide no support for Windows.  Also, though I have included many unit tests (every file ending in `_test.py`), I provide no guarantee that the code is free of bugs.  If you choose to use this package, you do so at your own risk.

# Pre-processing

If you need to pre-process your own data, the steps are as follows:
 1. Process satellite data (get raw data into Ryan's NetCDF format).
 2. Process radar data (get raw data into Ryan's NetCDF format).
 3. Compute normalization parameters (means and standard deviations), which will be used to normalize satellite data while converting satellite data to predictors.
 4. Create predictor files (from satellite data).
 5. Run echo classification (on radar data, to classify each pixel as convection or not).  Echo classification will be used in creating target files (next step).  The end result is a binary grid (1 for convection, 0 for no-convection) at each time step.
 6. Create target files (from radar data).

## 1. Processing satellite data

You will use the script `process_satellite_data.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `process_satellite_data.py` from a Unix terminal.

```
python process_satellite_data.py \
    --input_satellite_dir_name="your directory name here" \
    --first_date_string="20160101" \
    --last_date_string="20160131" \
    --allow_missing_days=1 \
    --output_satellite_dir_name="your directory name here"
```

More details on the input arguments are provided below.

 - `input_satellite_dir_name` is a string, pointing to the directory with raw files (from Taiwan CWB).  Files therein will be found by `twb_satellite_io.find_file` and read by `twb_satellite_io.read_file`, where `twb_satellite_io.py` is in the directory `ml4convection/io`.  `twb_satellite_io.find_file` will only look for files named like `[input_satellite_dir_name]/[yyyy-mm]/[yyyy-mm-dd_HHMM].B[nn].GSD.Cnt` or `[input_satellite_dir_name]/[yyyy-mm]/[yyyy-mm-dd_HHMM].B[nn].GDS.Cnt`, where `[yyyy]` is the 4-digit year; `[mm]` is the 2-digit month; `[dd]` is the 2-digit day of month; `[HH]` is the 2-digit hour; `[MM]` is the 2-digit minute; and `[nn]` is the satellite-number.  An example of a good file name, assuming the top-level directory is `foo`, is `foo/2016-01/2016-01-05_1010.B13.GSD.Cnt`.  This file contains data for band 13.
 - `first_date_string` is a string (format `yyyymmdd`) containing the first date in the period you want to process.
 - `last_date_string` is a string (format `yyyymmdd`) containing the last date in the period you want to process.
 - `allow_missing_days` is a Boolean flag (0 for False, 1 for True).  This determines what happens if any date in the time period is missing (*i.e.*, the raw data cannot be found in `input_satellite_dir_name`).  If `allow_missing_days == 1`, the script `process_satellite_data.py` will just process the dates it finds and ignore the dates it can't find.  But if `allow_missing_days == 0` and there is a missing date, the script will throw an error and stop.
 - `output_satellite_dir_name` is a string, pointing the directory where you want processed NetCDF files.  Files will be written to this directory by `satellite_io.write_file`, to specific locations determined by `satellite_io.find_file`.  The files will be named like `[output_satellite_dir_name]/[yyyy]/satellite_[yyyymmdd].nc`, so one NetCDF file per date.

After this basic processing, you should quality-control (QC) the satellite data as well.  For more on QC (the specific methodology and why it is needed), see Section 2b of the paper in *Monthly Weather Review*, [here](https://doi.org/10.1175/MWR-D-21-0096.1).  You will use the script `qc_satellite_data.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `qc_satellite_data.py` from a Unix terminal.  Please do not change the arguments `half_window_size_px`, `min_temperature_diff_kelvins`, `min_region_size_px`.  The values given below correspond to those in the MWR paper.

```
python qc_satellite_data.py \
    --input_satellite_dir_name="your directory name here" \
    --first_date_string="20160101" \
    --last_date_string="20160131" \
    --half_window_size_px=2 \
    --min_temperature_diff_kelvins=1 \
    --min_region_size_px=1000 \
    --output_satellite_dir_name="your directory name here"
```

More details on the input arguments are provided below.

 - `input_satellite_dir_name` is a string, pointing to the directory with non-quality-controlled files.  This could be the output directory from `process_satellite_data.py`.  Either way, files in `input_satellite_dir_name` should be named like `[output_satellite_dir_name]/[yyyy]/satellite_[yyyymmdd].nc`, so one NetCDF file per date.
 - `first_date_string` is a string (format `yyyymmdd`) containing the first date in the period you want to process.
 - `last_date_string` is a string (format `yyyymmdd`) containing the last date in the period you want to process.
 - `output_satellite_dir_name` is a string, pointing the directory where you want processed NetCDF files.  Files will be written to this directory by `satellite_io.write_file`, to specific locations determined by `satellite_io.find_file`.  The files will be named like `[output_satellite_dir_name]/[yyyy]/satellite_[yyyymmdd].nc`, so one NetCDF file per date.

## 2. Processing radar data

**IMPORTANT**: There are only two situations in which you will need radar data, which are used to create the labels (*i.e.*, correct convection masks) and are *not* used to create the predictors (satellite images).  These situations are:
 - You want to train your own model.
 - You want to run one of my pre-trained models in inference mode (*i.e.*, to make predictions for new satellite data; this does not require labels), but you also want to evaluate the new predictions (this does require labels, because evaluation requires correct answers to compare with the predictions).

You will use the script `process_radar_data.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `process_radar_data.py` from a Unix terminal.

```
python process_radar_data.py \
    --input_radar_dir_name="your directory name here" \
    --first_date_string="20160101" \
    --last_date_string="20160131" \
    --allow_missing_days=1 \
    --output_radar_dir_name="your directory name here"
```

More details on the input arguments are provided below.

 - `input_radar_dir_name` is a string, pointing to the directory with raw files (from Taiwan CWB).  Files therein will be found by `twb_radar_io.find_file` and read by `twb_radar_io.read_file`, where `twb_radar_io.py` is in the directory `ml4convection/io`.  `twb_radar_io.find_file` will only look for files named like `[input_radar_dir_name]/[yyyymmdd]/MREF3D21L.[yyyymmdd].[HHMM].gz`.  An example of a good file name, assuming the top-level directory is `foo`, is `foo/20160105/MREF3D21L.20160105.1010.gz`.
 - `first_date_string` is a string (format `yyyymmdd`) containing the first date in the period you want to process.
 - `last_date_string` is a string (format `yyyymmdd`) containing the last date in the period you want to process.
 - `allow_missing_days` is a Boolean flag (0 for False, 1 for True).  This determines what happens if any date in the time period is missing (*i.e.*, the raw data cannot be found in `input_radar_dir_name`).  If `allow_missing_days == 1`, the script `process_radar_data.py` will just process the dates it finds and ignore the dates it can't find.  But if `allow_missing_days == 0` and there is a missing date, the script will throw an error and stop.
 - `output_radar_dir_name` is a string, pointing the directory where you want processed NetCDF files.  Files will be written to this directory by `radar_io.write_file`, to specific locations determined by `radar_io.find_file`.  The files will be named like `[output_radar_dir_name]/[yyyy]/satellite_[yyyymmdd].nc`, so one NetCDF file per date.

## 3. Computing normalization parameters

Following common practice in machine learning, we train the U-nets with normalized values ($z$-scores) instead of physical values (satellite brightness temperatures).  The normalization is done separately for each Himawari-8 band (channels 8, 9, 10, 11, 13, 14, and 16).  Normalization should always be done *with respect to the training set* -- in other words, when normalizing the validation and testing data, you use the normalization parameters from the training data, rather than recomputing normalization parameters for the validation and testing data.  Recomputing normalization parameters for each new dataset can lead to large model errors, *e.g.*, if the new normalization parameters differ a lot from those in the training data.

**IMPORTANT**: There is only one situation in which you will need to recompute normalization parameters: if you want to train your own model.  If you just want to use my pre-trained models in inference mode, you can use my normalization file, [here](https://drive.google.com/file/d/1v8gCYZDsVC9HOSwSA0cpcB5ayx9dgDDb/view?usp=sharing).

If you need to recompute normalization parameters, use the script `get_normalization_params.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `get_normalization_params.py` from a Unix terminal.

```
python get_normalization_params.py \
    --input_satellite_dir_name="your directory name here" \
    --first_date_string="20160101" \
    --last_date_string="20161224" \
    --num_values_per_band=200000 \
    --output_file_name="your file name here"
```

More details on the input arguments are provided below.

 - `input_satellite_dir_name` is a string, pointing to the directory with processed (and better yet, quality-controlled) satellite files.  This could be the output directory from `process_satellite_data.py` or `qc_satellite_data.py`.  Either way, files in `input_satellite_dir_name` should be named like `[output_satellite_dir_name]/[yyyy]/satellite_[yyyymmdd].nc`, so one NetCDF file per date.
 - `first_date_string` is a string (format `yyyymmdd`) containing the first date in the training dataset, which is the only dataset used to compute normalization parameters.  Note that my training set was Jan 1 2016 - Dec 24 2016.
 - `last_date_string` is a string (format `yyyymmdd`) containing the last date in the training dataset.
 - `num_values_per_band` is the number of sample values, per Himawari-8 channel, used to compute normalization parameters.  I recommend leaving this at 200 000.
 - `output_file_name` is a string, pointing to where you want the output file.  This file (containing normalization parameters) will be in Pickle format.

## 4. Creating predictor files

**IMPORTANT**: There are two formats for predictor files: full-grid and partial-grid.  There is only one situation in which you will need to create partial-grid files: if you want to train your own model.  If you just want to use my pre-trained models in inference mode, you need only full-grid predictor files.

*A little more explanation*: Full-grid predictor files contain satellite data on the full grid, spanning 18-29 $^{\circ}$N and 115-126.5 $^{\circ}$E.  Since the grid spacing is a uniform 0.0125$^{\circ}$, the full grid is 881 rows $\times$ 921 columns, or 881 latitudes $\times$ 921 longitudes.  Partial-grid predictor files contain satellite data on radar-centered patches of the full grid.  These radar-centered patches, which are 205 rows $\times$ 205 columns, are needed only for training.  For a complete explanation (*e.g.*, if you're wondering "how can a model be trained with 205 $\times$ 205 grids but then applied to 881 $\times$ 921 grids at inference time?), see Sections 4a ("Training with patches") and 4b ("Inference with sliding windows") of the MWR paper.

To create predictor files, use the script `create_predictors.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `create_predictors.py` from a Unix terminal.  Please leave `spatial_downsampling_factor` as 1; this will keep the grid spacing at 0.0125$^{\circ}$.  Also, please leave `half_grid_size_px` as 102; this ensures that, if creating partial-grid files, the partial radar-centered grids will be 205 $\times$ 205 pixels.

```
python create_predictors.py \
    --input_satellite_dir_name="your directory name here" \
    --use_partial_grids=[0 or 1] \
    --half_grid_size_px=102 \
    --spatial_downsampling_factor=1 \
    --first_date_string="20160101" \
    --last_date_string="20160131" \
    --input_normalization_file_name="your file name here" \
    --output_predictor_dir_name="your directory name here"
```

More details on the input arguments are provided below.

 - `input_satellite_dir_name` is a string, pointing to the directory with processed (and better yet, quality-controlled) satellite files.  This could be the output directory from `process_satellite_data.py` or `qc_satellite_data.py`.  Either way, files in `input_satellite_dir_name` should be named like `[input_satellite_dir_name]/[yyyy]/satellite_[yyyymmdd].nc`, so one NetCDF file per date.
 - `use_partial_grids` is a Boolean flag.  If 1, the script will create partial-grid predictor files.  If 0, the script will create full-grid predictor files.
 - `first_date_string` is a string (format `yyyymmdd`) containing the first date in the period you want to process.
 - `last_date_string` is a string (format `yyyymmdd`) containing the last date in the period you want to process.
 - `input_normalization_file_name` is a string pointing to the file with normalization parameters (*i.e.*, one created by the script `get_normalization_params`.py, which was discussed above).  This is a Pickle file.
 - `output_predictor_dir_name` is a string, pointing the directory where you want predictor files.  Files will be written to this directory by `example_io.write_predictor_file`, to specific locations determined by `example_io.find_predictor_file`.  Full-grid files will be named like `[output_predictor_dir_name]/[yyyy]/predictors_[yyyymmdd].nc`, so one NetCDF file per date.  Partial-grid files will be named like `[output_predictor_dir_name]/[yyyy]/predictors_[yyyymmdd]_radar[n].nc` -- where `[n]` is an integer from 0 to 3 -- so one NetCDF file per date per radar.

## 5. Running echo classification

**IMPORTANT**: "Echo classification" is the process of classifiying radar echoes according to type.  Some echo-classification algorithms have many categories (*e.g.*, hail, graupel, snow, ice pellets, convective rain, stratiform rain, anvil, etc.), but our algorithm has only two categories: convective or non-convective.  There are only two situations in which you will need to run echo classification:
 - You want to train your own model.
 - You want to run one of my pre-trained models in inference mode, but you also want to evaluate the new predictions, which requires labels (correct answers).

Use the script `run_echo_classification.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `run_echo_classification.py` from a Unix terminal.  Please do not change the arguments `min_height_fraction_for_peakedness`, `thin_height_grid`, `min_size_pixels`.  The values given below correspond to those in the MWR paper.

```
python run_echo_classification.py \
    --input_radar_dir_name="your directory name here" \
    --first_date_string="20160101" \
    --last_date_string="20160131" \
    --min_height_fraction_for_peakedness=0.59 \
    --thin_height_grid=1 \
    --min_size_pixels=10 \
    --output_dir_name="your directory name here"
```

More details on the input arguments are provided below.

 - `input_radar_dir_name` is a string, pointing to the directory with processed radar files.  This could be the output directory from `process_radar_data.py`.  Either way, files in `input_radar_dir_name` should be named like `[input_radar_dir_name]/[yyyy]/reflectivity_[yyyymmdd].nc`, so one NetCDF file per date.
 - `first_date_string` is a string (format `yyyymmdd`) containing the first date in the period you want to process.
 - `last_date_string` is a string (format `yyyymmdd`) containing the last date in the period you want to process.
 - `output_dir_name` is a string, pointing the directory where you want output files (containing a binary grid at each time step, with 1 for convective pixels and 0 for non-convective pixels).  Files will be written to this directory by `radar_io.write_echo_classifn_file`, to specific locations determined by `radar_io.find_file`.  Files will be named like `[output_dir_name]/[yyyy]/echo_classification_[yyyymmdd].nc`, so one NetCDF file per date.

## 6. Creating target files

**IMPORTANT**: There are only two situations in which you will need to create target files (containing labels, *i.e.*, correct answers):
 - You want to train your own model.  In this case you will need partial-grid target files.
 - You want to run one of my pre-trained models in inference mode, but you also want to evaluate the new predictions.  In this case you will need full-grid target files.

To create target files, use the script `create_targets.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `create_targets.py` from a Unix terminal.  Please leave `spatial_downsampling_factor` as 1; this will keep the grid spacing at 0.0125$^{\circ}$.  Also, please leave `half_grid_size_px` as 102; this ensures that, if creating partial-grid files, the partial radar-centered grids will be 205 $\times$ 205 pixels.

```
python create_targets.py \
    --input_echo_classifn_dir_name="your directory name here" \
    --input_mask_file_name="your file name here" \
    --use_partial_grids=[0 or 1] \
    --half_grid_size_px=102 \
    --spatial_downsampling_factor=1 \
    --first_date_string="20160101" \
    --last_date_string="20160131" \
    --output_target_dir_name="your directory name here"
```

More details on the input arguments are provided below.

 - `input_echo_classifn_dir_name` is a string, pointing to the directory with processed echo-classification files (containing the binary masks).  This could be the output directory from `run_echo_classification.py`.  Either way, files in `input_echo_classifn_dir_name` should be named like `[input_echo_classifn_dir_name]/[yyyy]/echo_classification_[yyyymmdd].nc`, so one NetCDF file per date.
 - `input_mask_file_name` is a string, pointing to the file containing the "radar mask".  This is a binary mask over the full grid (881 $\times$ 921 pixels), indicating which pixels are within 100 km of the nearest radar.  Echo classifications will be used only at these pixels.  Subjectively (*i.e.*, by visual inspection), we have deemed that echo classifications are not accurate enough at locations $>$ 100 km from the nearest radar.  Instead of creating your own, you can find the file [here](https://drive.google.com/file/d/1lDsNbsI8_mjzSR58nu3tg6BDyRAiyRZG/view?usp=sharing).
 - `use_partial_grids` is a Boolean flag.  If 1, the script will create partial-grid target files.  If 0, the script will create full-grid target files.
 - `first_date_string` is a string (format `yyyymmdd`) containing the first date in the period you want to process.
 - `last_date_string` is a string (format `yyyymmdd`) containing the last date in the period you want to process.
 - `output_target_dir_name` is a string, pointing the directory where you want target files.  Files will be written to this directory by `example_io._write_target_file`, to specific locations determined by `example_io.find_target_file`.  Full-grid files will be named like `[output_target_dir_name]/[yyyy]/targets_[yyyymmdd].nc`, so one NetCDF file per date.  Partial-grid files will be named like `[output_target_dir_name]/[yyyy]/targets_[yyyymmdd]_radar[n].nc` -- where `[n]` is an integer from 0 to 3 -- so one NetCDF file per date per radar.

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
 - `first_date_string` and `last_date_string` are the first and last days to plot.  In other words, the model will be used to plot predictions for all days from `first_date_string` to `last_date_string`.
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
 - `first_date_string` and `last_date_string` are the first and last days to plot.  In other words, the model will be used to plot brightness-temperature maps for all days from `first_date_string` to `last_date_string`.
 - `band_numbers` is a list of band numbers to plot.  `plot_satellite.py` will create one image per time step per band.
 - `daily_times_seconds` is a list of daily times at which to plot brightness-temperature maps, same as the input for `plot_predictions.py`.
 - `output_dir_name` is a string, naming the output directory.  Plots will be saved here as JPEG images.

If you want to just plot radar data, use the script `plot_radar.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `plot_radar.py` from a Unix terminal.

```
python plot_radar.py \
    --input_reflectivity_dir_name="your directory name here" \
    --input_echo_classifn_dir_name="your directory name here" \
    --first_date_string="date in format yyyymmdd" \
    --last_date_string="date in format yyyymmdd" \
    --plot_all_heights=[0 or 1] \
    --daily_times_seconds 0 7200 14400 21600 28800 36000 43200 50400 57600 64800 72000 79200 \
    --expand_to_satellite_grid=[0 or 1] \
    --output_dir_name="your directory name here" \
```

More details on the input arguments are provided below.

 - `input_reflectivity_dir_name` is a string, naming the directory with reflectivity files.  Files therein will be found by `radar_io.find_file` and read by `radar_io.read_reflectivity_file`, where `radar_io.py` is in the directory `ml4convection/io`.  `radar_io.find_file` will only look for files named like `[input_reflectivity_dir_name]/[yyyy]/reflectivity_[yyyymmdd].nc` and `[input_reflectivity_dir_name]/[yyyy]/reflectivity_[yyyymmdd].nc.gz`, where `[yyyy]` is the 4-digit year and `[yyyymmdd]` is the date.  An example of a good file name, assuming the top-level directory is `foo`, is `foo/2016/reflectivity_20160101.nc`.
 - `input_echo_classifn_dir_name` is a string, naming the directory with echo-classification files.  If you specify this input argument, then `plot_radar.py` will plot black dots on top of the reflectivity map, one for each convective grid point.  If you leave this input argument alone, echo classification will not be plotted.  Files in `input_echo_classifn_dir_name` will be found by `radar_io.find_file` and read by `radar_io.read_echo_classifn_file`.  `radar_io.find_file` will only look for files named like `[input_echo_classifn_dir_name]/[yyyy]/echo_classification_[yyyymmdd].nc` and `[input_echo_classifn_dir_name]/[yyyy]/echo_classification_[yyyymmdd].nc.gz`, where `[yyyy]` is the 4-digit year and `[yyyymmdd]` is the date.  An example of a good file name, assuming the top-level directory is `foo`, is `foo/2016/echo_classification_20160101.nc`.
 - `first_date_string` and `last_date_string` are the first and last days to plot.  In other words, the model will be used to plot reflectivity maps for all days from `first_date_string` to `last_date_string`.
 - `plot_all_heights` is a Boolean flag (0 or 1).  If 1, the script will plot reflectivity at all heights, thus producing one plot per time step per height.  If 0, the script will plot composite (column-maximum) reflectivity, thus producing one plot per time step.
 - `daily_times_seconds` is a list of daily times at which to plot reflectivity maps, same as the input for `plot_predictions.py`.
 - `expand_to_satellite_grid` is a Boolean flag (0 or 1).  If 1, the script will plot reflectivity on the Himawari-8 grid, which is slightly larger than the radar grid.  In this case values around the edge of the grid will all be 0 dBZ.
 - `output_dir_name` is a string, naming the output directory.  Plots will be saved here as JPEG images.

# Evaluating the trained U-net

## Computing "basic" evaluation scores (one file per day)

Evaluation scripts are split into those that compute "basic" scores and those that compute "advanced" scores.  Basic scores are written to one file per day, whereas advanced scores are written to one file for a whole time period (*e.g.*, the validation period, which is Jan 1 2017 - Dec 24 2017 in the *Monthly Weather Review* paper).  For any time period *T*, basic scores can be aggregated over *T* to compute advanced scores.  This documentation does not list all the basic and advanced scores (there are many), but below is an example:

 - The fractions skill score (FSS) is an advanced score, defined as 1 - SSE / SSE<sub>ref</sub>.
 - SSE (the actual sum of squared errors) and SSE<sub>ref</sub> (the reference sum of squared errors) are basic scores, each with one value per time step.
 - To compute the FSS for a period *T*, SSE and SSE<sub>ref</sub> are summed over *T* and then the following equation is applied: FSS = 1 - SSE / SSE<sub>ref</sub>.

If you want to compute basic ungridded scores (averaged over the whole domain), use the script `compute_basic_scores_ungridded.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `compute_basic_scores_ungridded.py` from a Unix terminal.

```
python compute_basic_scores_ungridded.py \
    --input_prediction_dir_name="your directory name here" \
    --first_date_string="date in format yyyymmdd" \
    --last_date_string="date in format yyyymmdd" \
    --time_interval_steps=[integer] \
    --use_partial_grids=[0 or 1] \
    --smoothing_radius_px=2 \
    --matching_distances_px 1 2 3 4 \
    --num_prob_thresholds=21 \
    --prob_thresholds -1 \
    --output_dir_name="your directory name here" \
```

More details on the input arguments are provided below.

 - `input_prediction_dir_name` is a string, naming the directory with prediction files.  Files therein will be found by `prediction_io.find_file` and read by `prediction_io.read_file`, as for the input argument to `plot_predictions.py`.
 - `first_date_string` and `last_date_string` are the first and last days for which to compute scores.  In other words, the script will compute basic scores for all days from `first_date_string` to `last_date_string`.
 - `time_interval_steps` is used to reduce computing time.  If you want to compute scores for every *k*<super>th</super> time step, make `time_interval_steps` be *k*.
 - `use_partial_grids` is a Boolean flag (0 or 1), indicating whether you want to compute scores for predictions on the Himawari-8 grid or partial (radar-centered) grids.
 - `smoothing_radius_px`, used for full-grid predictions (if `use_partial_grids` is 0), is the *e*-folding radius for Gaussian smoothing (pixels).  Each probability field will be filtered by this amount before computing scores.  I suggest making this 2.
 - `matching_distances_px` is a list of neighbourhood distances (pixels) for evaluation.  Basic scores will be computed for each neighbourhood distance, and one set of files will be written for each neighbourhood distance.
 - `num_prob_thresholds` is the number of probability thresholds at which to compute scores based on binary (rather than probabilistic) forecasts.  These thresholds will be equally spaced from 0.0 to 1.0.  If you instead want to specify probability thresholds, make `num_prob_thresholds` -1 and use the argument `prob_thresholds`.
 - `prob_thresholds` is a list of probability thresholds (between 0.0 and 1.0) at which to compute scores based on binary (rather than probabilistic) forecasts.   you instead want equally spaced thresholds from 0.0 to 1.0, make `prob_thresholds` -1 and use the argument `num_prob_thresholds`.
 - `output_dir_name` is a string, naming the output directory.  Basic scores will be saved here as NetCDF files.

If you want to compute basic gridded scores (one set of scores for each grid point), use the script `compute_basic_scores_gridded.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `compute_basic_scores_gridded.py` from a Unix terminal.

```
python compute_basic_scores_gridded.py \
    --input_prediction_dir_name="your directory name here" \
    --first_date_string="date in format yyyymmdd" \
    --last_date_string="date in format yyyymmdd" \
    --smoothing_radius_px=2 \
    --matching_distances_px 1 2 3 4 \
    --climo_file_names "climatology/climo_neigh-distance-px=1.p" "climatology/climo_neigh-distance-px=2.p" "climatology/climo_neigh-distance-px=3.p" "climatology/climo_neigh-distance-px=4.p" \
    --prob_thresholds -1 \
    --output_dir_name="your directory name here" \
```

More details on the input arguments are provided below.

 - `input_prediction_dir_name` is the same as for `compute_basic_scores_ungridded.py`.
 - `first_date_string` and `last_date_string` are the same as for `compute_basic_scores_ungridded.py`.
 - `smoothing_radius_px` is the same as for `compute_basic_scores_ungridded.py`.
 - `matching_distances_px` is the same as for `compute_basic_scores_ungridded.py`.
 - `climo_file_names` is a list of paths to climatology files, one for each matching distance.  Each file will be read by `climatology_io.read_file`, where `climatology_io.py` is in the directory `ml4convection/io`.  Each file specifies the climatology (*i.e.*, convection frequency in the training data at each pixel), which is ultimately used to compute the Brier skill score at each pixel.  The climatology is different for each matching distance, because a matching distance (radius) of *N* pixels turns each convective label (one pixel) into *πr*<super>2</super> labels (pixels).  The climatology depends on the matching distance and training period, which is why I have not included climatology files in this package.
 - `prob_thresholds` is the same as for `compute_basic_scores_ungridded.py`.
 - `output_dir_name` is the same as for `compute_basic_scores_ungridded.py`.

## Computing "advanced" evaluation scores (one file per period)

If you want to compute advanced ungridded scores (averaged over the whole domain), use the script `compute_advanced_scores_ungridded.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `compute_advanced_scores_ungridded.py` from a Unix terminal.

```
python compute_advanced_scores_ungridded.py \
    --input_basic_score_dir_name="your directory name here" \
    --first_date_string="date in format yyyymmdd" \
    --last_date_string="date in format yyyymmdd" \
    --num_bootstrap_reps=[integer] \
    --use_partial_grids=[0 or 1] \
    --desired_month=[integer] \
    --split_by_hour=[0 or 1] \
    --input_climo_file_name="your file name here" \
    --output_dir_name="your directory name here" \
```

More details on the input arguments are provided below.

 - `input_basic_score_dir_name` is a string, naming the directory with basic ungridded scores.  Files therein will be found by `evaluation.find_basic_score_file` and read by `evaluation.read_basic_score_file`, where `evaluation.py` is in the directory `ml4convection/utils`.  `evaluation.find_basic_score_file` will only look for files named like `[input_basic_score_dir_name]/[yyyy]/basic_scores_gridded=0_[yyyymmdd].nc` (if `use_partial_grids` is 0) or `[input_basic_score_dir_name]/[yyyy]/basic_scores_gridded=0_[yyyymmdd]_radar[k].nc` (if `use_partial_grids` is 1), where `[yyyy]` is the 4-digit year; `[yyyymmdd]` is the date; and `[k]` is the radar number, an integer from 1-3.  An example of a good file name, assuming the top-level directory is `foo`, is `foo/2016/basic_scores_gridded=0_20160101.nc` or `foo/2016/basic_scores_gridded=0_20160101_radar1.nc`.
 - `first_date_string` and `last_date_string` are the first and last days for which to aggregate basic scores into advanced scores.  In other words, the script will compute advanced scores for all days from `first_date_string` to `last_date_string`.
 - `num_bootstrap_reps` is the number of replicates (sometimes called "iterations") for bootstrapping, used to compute uncertainty.  If you do not want to boostrap, make `num_bootstrap_reps` 1.
 - `use_partial_grids` is a Boolean flag (0 or 1), indicating whether you want to compute scores for predictions on the Himawari-8 grid or partial (radar-centered) grids.
 - `desired_month` is an integer from 1 to 12, indicating the month for which you want compute advanced scores.  If you want to include all months, make this -1.
 - `split_by_hour` is a Boolean flag (0 or 1), indicating whether or not you want to compute one set of advanced scores for each hour of the day (0000-0059 UTC, 0100-0159 UTC, etc.).
 - `input_climo_file_name` is the path to the climatology file.  For more details on this (admittedly weird) input argument, see the documentation above for the input argument `climo_file_names` to the script `compute_basic_scores_gridded.py`.
 - `output_dir_name` is a string, naming the output directory.  Advanced scores will be saved here as NetCDF files.

If you want to compute advanced gridded scores (one set of scores for each grid point), use the script `compute_advanced_scores_gridded.py` in the directory `ml4convection/scripts`.  Below is an example of how you would call `compute_advanced_scores_gridded.py` from a Unix terminal.

```
python compute_advanced_scores_gridded.py \
    --input_basic_score_dir_name="your directory name here" \
    --first_date_string="date in format yyyymmdd" \
    --last_date_string="date in format yyyymmdd" \
    --num_subgrids_per_dim=3 \
    --output_dir_name="your directory name here" \
```

More details on the input arguments are provided below.

 - `input_basic_score_dir_name` is a string, naming the directory with basic gridded scores.  Files therein will be found by `evaluation.find_basic_score_file` and read by `evaluation.read_basic_score_file`.  `evaluation.find_basic_score_file` will only look for files named like `[input_basic_score_dir_name]/[yyyy]/basic_scores_gridded=1_[yyyymmdd].nc` (if `use_partial_grids` is 0) or `[input_basic_score_dir_name]/[yyyy]/basic_scores_gridded=1_[yyyymmdd]_radar[k].nc` (if `use_partial_grids` is 1), where `[yyyy]` is the 4-digit year; `[yyyymmdd]` is the date; and `[k]` is the radar number, an integer from 1-3.  An example of a good file name, assuming the top-level directory is `foo`, is `foo/2016/basic_scores_gridded=1_20160101.nc` or `foo/2016/basic_scores_gridded=1_20160101_radar1.nc`.
 - `first_date_string` and `last_date_string` are the same as for `compute_advanced_scores_ungridded.py`.
 - `num_subgrids_per_dim` (an integer) is the number of subgrids per spatial dimension.  For example, `num_subgrids_per_dim` is 3, the script will use 3 * 3 = 9 subgrids.  It will aggregate basic scores for one subgrid at a time.  Although this input argument is weird, it **greatly** reduces the memory requirements.
 - `output_dir_name` is the same as for `compute_advanced_scores_ungridded.py`.

## Plotting evaluation scores

ml4convection contains plotting code only for advanced evaluation scores (aggregated over a time period), not for basic scores (one set of scores per time step).

If you want to plot ungridded scores (averaged over the whole domain) with no separation by month or hour, use the script `plot_evaluation.py` in the directory `ml4convection/scripts`.  `plot_evaluation.py` creates an attributes diagram (evaluating probabilistic forecasts) and a performance diagram (evaluating binary forecasts at various probability thresholds).  Below is an example of how you would call `plot_evaluation.py` from a Unix terminal.

```
python plot_evaluation.py \
    --input_advanced_score_file_name="your file name here" \
    --best_prob_threshold=[float] \
    --confidence_level=0.95 \
    --output_dir_name="your directory name here" \
```

More details on the input arguments are provided below.

 - `input_advanced_score_file_name` is a string, giving the full path to the file with advanced ungridded scores.  This file will be read by `evaluation.read_advanced_score_file`, where `evaluation.py` is in the directory `ml4convection/utils`.
 - `best_prob_threshold` is the optimal probability threshold, which will be marked with a star in the performance diagram.  If you have not yet chosen the optimal threshold and want it to be determined "on the fly," make this argument -1.
 - `confidence_level` is the confidence level for plotting uncertainty.  This argument will be only if `input_advanced_score_file_name` contains bootstrapped scores.  For example, if the file contains scores for 1000 bootstrap replicates and `confidence_level` is 0.95, the 95% confidence interval will be plotted (ranging from the 2.5<super>th</super> to 97.5<super>th</super> percentile over all bootstrap replicates).
 - `output_dir_name` is a string, naming the output directory.  Plots will be saved here as JPEG files.

If you want to plot gridded scores (one set of scores per grid point), use the script `plot_gridded_evaluation.py` in the directory `ml4convection/scripts`.  `plot_gridded_evaluation.py` plots a gridded map for each of the following scores: Brier score, Brier skill score, fractions skill score, label climatology (event frequency in the training data, which isn't an evaluation score), model climatology (mean forecast probability, which also isn't an evaluation score), probability of detection, success ratio, frequency bias, and critical success index.  Below is an example of how you would call `plot_gridded_evaluation.py` from a Unix terminal.

```
python plot_gridded_evaluation.py \
    --input_advanced_score_file_name="your file name here" \
    --probability_threshold=[float] \
    --output_dir_name="your directory name here" \
```

More details on the input arguments are provided below.

 - `input_advanced_score_file_name` is a string, giving the full path to the file with advanced gridded scores.  This file will be read by `evaluation.read_advanced_score_file`.
 - `probability_threshold` is the probability threshold for binary forecasts.  This is required for plotting probability of detection (POD), success ratio, frequency bias, and critical success index (CSI), which are scores based on binary forecasts.
 - `output_dir_name` is a string, naming the output directory.  Plots will be saved here as JPEG files.

If you want to plot ungridded scores separated by month and hour, use the script `plot_evaluation_by_time.py` in the directory `ml4convection/scripts`.  `plot_evaluation_by_time.py` creates a monthly attributes diagram, hourly attributes diagram, monthly performance diagram, and hourly performance diagram.  `plot_evaluation_by_time.py` also plots fractions skill score (FSS), CSI, and frequency bias as a function of month and hour.  Thus, `plot_evaluation_by_time.py` plots 6 figures.  Below is an example of how you would call `plot_evaluation_by_time.py` from a Unix terminal.

```
python plot_evaluation_by_time.py \
    --input_dir_name="your directory name here" \
    --probability_threshold=[float] \
    --confidence_level=0.95 \
    --output_dir_name="your directory name here" \
```

More details on the input arguments are provided below.

 - `input_dir_name` is a string, naming the directory with advanced ungridded scores separated by month and hour.  Files therein will be found by `evaluation.find_advanced_score_file` and read by `evaluation.read_advanced_score_file`.  `evaluation.find_advanced_score_file` will only look for files named like `[input_dir_name]/advanced_scores_month=[mm]_gridded=0.p` and `[input_dir_name]/advanced_scores_hour=[hh]_gridded=0.p`, where `[mm]` is the 2-digit month and `[hh]` is the 2-digit hour.  An example of a good file name, assuming the directory is `foo`, is `foo/advanced_scores_month=03_gridded=0.p` or `foo/advanced_scores_hour=12_gridded=0.p`.
 - `probability_threshold` is the probability threshold for binary forecasts.  This is required for plotting CSI and frequency bias versus hour and month.
 - `confidence_level` is the confidence level for plotting uncertainty.  This argument will be used only if files in `input_dir_name` contain bootstrapped scores.  For example, if the files contain scores for 1000 bootstrap replicates and `confidence_level` is 0.95, the 95% confidence interval will be plotted (ranging from the 2.5<super>th</super> to 97.5<super>th</super> percentile over all bootstrap replicates).
 - `output_dir_name` is a string, naming the output directory.  Plots will be saved here as JPEG files.
