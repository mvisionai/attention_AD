function runSegmentation()

    % List of open inputs
    X = 1;
    nrun = X; % enter the number of runs here
    jobfile = {'/Users/antrax/Documents/MATLAB/spmAutomaticSegmentation/runSegmentation_job.m'};
    jobs = repmat(jobfile, 1, nrun);
    inputs = cell(0, nrun);
    for crun = 1:nrun
    end
    spm('defaults', 'FMRI');
    spm_jobman('run', jobs, inputs{:});
    
end


