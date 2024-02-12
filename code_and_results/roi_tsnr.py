import sys
import numpy as np
import pandas as pd
from nilearn import input_data

preproc_file, atlas_file, out_path = sys.argv[1:4]

masker = input_data.NiftiLabelsMasker(labels_img=atlas_file, standardize=False)
preproc_ts = masker.fit_transform(preproc_file)
pp_ts_mean = np.mean(preproc_ts, axis=0)
pp_ts_stdev = np.std(preproc_ts, axis=0)
pp_ts_tsnr = np.true_divide(pp_ts_mean, pp_ts_stdev)
df = pd.DataFrame([pp_ts_mean, pp_ts_stdev, pp_ts_tsnr], index=['tmean','tsd','tsnr']).T
df.to_csv(out_path)
