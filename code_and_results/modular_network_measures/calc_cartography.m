function res = calc_cartography(mat_path, part_dir, out_prefix)

%% Load BCT, NCT, and GenLouvain
addpath('/home/despoB/dlurie/Software/matlab/BCT/2017_01_15_BCT')
addpath('/home/despoB/dlurie/Software/matlab/NCT/2017_10_22_NCT')
addpath('/home/despoB/dlurie/Software/matlab/GenLouvain')

%% Load the adjacency matrix
corr_tab = readtable(mat_path, 'ReadVariableNames', false, 'Delimiter', ' ');
corr_mat = table2array(corr_tab);
corr_mat = weight_conversion(corr_mat, 'autofix');

%% Iterate over resolutions
formatspec_gamma = "gamma_%.1f"
gamma_range = 0.5:0.1:3.5

for y = gamma_range
    gamma = y
    
    %% Load consensus partition
    cpart_path = strcat(part_dir, compose(formatspec_gamma, gamma), '_GraphPartition.txt');
    fileID = fopen(cpart_path, 'r');
    formatspec_load = "%u";
    consensus_partition = fscanf(fileID, formatspec_load);
    consensus_partition = consensus_partition';
    
    %% Save matrix re-ordered by module
    [corr_mat_order, corr_mat_reordered] = reorder_mod(corr_mat, consensus_partition);
    reorder_path = strcat(out_prefix, compose(formatspec_gamma, gamma), '_ReorderedMatrix.txt');
    fileID = fopen(reorder_path, 'w');
    fprintf(fileID, '%.15e \r', corr_mat_reordered);
    fclose(fileID);

    %% Calculate nodal metrics
    wmd_z = module_degree_zscore(corr_mat, consensus_partition, 0);
    pc = participation_coef(corr_mat, consensus_partition);

    %% Save Within Module Degree Z-scores
    wmdz_path = strcat(out_prefix, compose(formatspec_gamma, gamma), '_WMDz.txt');
    fileID = fopen(wmdz_path, 'w');
    fprintf(fileID, '%.15e \r', wmd_z);
    fclose(fileID);

    %% Save Participation Coefficients
    pc_path = strcat(out_prefix, compose(formatspec_gamma, gamma), '_PC.txt');
    fileID = fopen(pc_path, 'w');
    fprintf(fileID, '%.15e \r', pc);
    fclose(fileID);
end
