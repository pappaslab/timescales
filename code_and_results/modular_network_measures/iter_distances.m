function res = iter_distances(mat_path, out_dir)

%% Load BCT, NCT, and GenLouvain
addpath('/home/despoB/dlurie/Software/matlab/BCT/2017_01_15_BCT')
addpath('/home/despoB/dlurie/Software/matlab/NCT/2017_10_22_NCT')
addpath('/home/despoB/dlurie/Software/matlab/GenLouvain')

%% Load the adjacency matrix
corr_tab = readtable(mat_path, 'ReadVariableNames', false, 'Delimiter', ',');
corr_mat = table2array(corr_tab);
corr_mat = weight_conversion(corr_mat, 'autofix');

%% Iterate over resolutions
formatspec = "gamma_%.1f";
gamma_range = 0.5:0.1:3.5;

for y = gamma_range
    gamma = y;
    %% Run the Louvain algorithm 1000 times
    n_nodes = size(corr_mat,1);
    n_iters = 1000;
    part_mat = zeros(n_nodes,n_iters);
    for idx = 1:n_iters;
            %iter_report = sprintf('Iteration %d of 1000.', idx);
            %disp(iter_report);
            n  = n_nodes;               % number of nodes
            M  = 1:n;                   % initial community affiliations
            Q0 = -1; Q1 = 0;            % initialize modularity values
            while Q1-Q0>1e-5;           % while modularity increases
                Q0 = Q1;                % perform community detection
                [M, Q1] = community_louvain(corr_mat, gamma, M, 'modularity');
            end   
            part_mat(:,idx) = M;
    end
    
    %% Compute partition similarities.
    [vi, mi] = partition_distance(part_mat);
    idx = logical(triu(ones(size(vi)), 1));
    vi_triu = vi(idx)';
    mi_triu = mi(idx)';
    mean_vi = mean(vi_triu);
    mean_mi = mean(mi_triu);
    
    %% Save VI
    vi_path = strcat(out_dir, compose(formatspec, gamma), '_meanVI.txt');
    fileID = fopen(vi_path, 'w');
    fprintf(fileID, '%.15e \r', mean_vi);
    fclose(fileID);
    
    %% Save MI
    mi_path = strcat(out_dir, compose(formatspec, gamma), '_meanMI.txt');
    fileID = fopen(mi_path, 'w');
    fprintf(fileID, '%.15e \r', mean_mi);
    fclose(fileID);
end