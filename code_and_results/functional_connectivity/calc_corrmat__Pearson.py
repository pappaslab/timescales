import sys
import numpy as np

# Parse command line arguments
timeseries_tsv,out_path = sys.argv[1:3]

# Load the previously extracted atlas ROI time series.
timeseries = np.loadtxt(timeseries_tsv, delimiter='\t')

# Calculate pairwise Pearson correlations between time series.
corrmat = np.corrcoef(timeseries, rowvar=False)

# Save the correlation matrix.
np.savetxt(out_path, corrmat, delimiter='\t')
