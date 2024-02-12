import sys
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import load_img
from nilearn.input_data import NiftiLabelsMasker

# Pull command line arguments.
cleaned_epi_path,atlas_path,out_path = sys.argv[1:4]

# Load the cleaned EPI image.
cleaned_epi_img = load_img(cleaned_epi_path)

# Load the brain atlas.
atlas_img = load_img(atlas_path)

# Initialize the Labels Masker class.
labels_masker = NiftiLabelsMasker(labels_img=atlas_img, background_label=0, resampling_target="data")

# Extract and save the atlas time series.
timeseries = labels_masker.fit_transform(cleaned_epi_img, confounds=None)
np.savetxt(out_path, timeseries, delimiter='\t')



