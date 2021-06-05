# PandAna 3

The PandAna 3 project is a re-working of PandAna to provide higher
efficiency for parallel calculations.

## Environment setup

PandAna3 requires a variety of installed Python packages. This
list is subject to change as the code is developed.

The recommended way to get all the required elements is through
`conda`. Instructions for installing `conda` are available at
https://docs.anaconda.com/anaconda/install.

The file `environment.yml` contains the items required for a 
minimal environment for PandAna 3. You can install this environment
using:

    conda env create -n pandana-base -f environment.yml

Activate the environment with:

    conda activate pandana-base

You may also wish to install addtional packages into the environment;
the online documentation for Conda is available for guidance.

## Testing

All tests can be run from the project top-level (this directory),
using the command `pytest`. Some tests require the file `small.h5`;
this file can be created by first running `tools/make_small_file`.


