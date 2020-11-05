function m = getJob(file, alignFolder, outputFolderName)
    
    [path, name, ext] = fileparts(file{1});
    pathToFile = fullfile(path, [name ext]);
    outputFolder = replace(path, alignFolder, outputFolderName);
    outputFile = fullfile(outputFolder, [name ext]);
    
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
        copyfile(pathToFile, outputFile);
    else
        rmdir(outputFolder, 's')
        mkdir(outputFolder);
        copyfile(pathToFile, outputFile);
    end
    

    matlabbatch{1}.spm.spatial.normalise.estwrite.subj.vol = {[outputFile ',1']};
    matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = {[outputFile ',1']};
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasreg = 0.0001;
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasfwhm = 60;
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.tpm = {strcat(spm('Dir'), '/tpm/TPM.nii')};
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.affreg = 'mni';
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.fwhm = 0;
    matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.samp = 3;
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.bb = [-78 -112 -70
                                                                 78 76 85];
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.vox = [2 2 2];
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.interp = 4;
    matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.prefix = 'w';
    
    matlabbatch{2}.spm.spatial.preproc.channel.vols(1) = cfg_dep('Normalise: Estimate & Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
    matlabbatch{2}.spm.spatial.preproc.channel.biasreg = 0.001;
    matlabbatch{2}.spm.spatial.preproc.channel.biasfwhm = 60;
    matlabbatch{2}.spm.spatial.preproc.channel.write = [0 0];
    matlabbatch{2}.spm.spatial.preproc.tissue(1).tpm = {strcat(spm('Dir'), '/tpm/TPM.nii,1')};
    matlabbatch{2}.spm.spatial.preproc.tissue(1).ngaus = 1;
    matlabbatch{2}.spm.spatial.preproc.tissue(1).native = [1 0];
    matlabbatch{2}.spm.spatial.preproc.tissue(1).warped = [0 0];
    matlabbatch{2}.spm.spatial.preproc.tissue(2).tpm = {strcat(spm('Dir'), '/tpm/TPM.nii,2')};
    matlabbatch{2}.spm.spatial.preproc.tissue(2).ngaus = 1;
    matlabbatch{2}.spm.spatial.preproc.tissue(2).native = [1 0];
    matlabbatch{2}.spm.spatial.preproc.tissue(2).warped = [0 0];
    matlabbatch{2}.spm.spatial.preproc.tissue(3).tpm = {strcat(spm('Dir'), '/tpm/TPM.nii,3')};
    matlabbatch{2}.spm.spatial.preproc.tissue(3).ngaus = 2;
    matlabbatch{2}.spm.spatial.preproc.tissue(3).native = [1 0];
    matlabbatch{2}.spm.spatial.preproc.tissue(3).warped = [0 0];
    matlabbatch{2}.spm.spatial.preproc.tissue(4).tpm = {strcat(spm('Dir'), '/tpm/TPM.nii,4')};
    matlabbatch{2}.spm.spatial.preproc.tissue(4).ngaus = 3;
    matlabbatch{2}.spm.spatial.preproc.tissue(4).native = [0 0];
    matlabbatch{2}.spm.spatial.preproc.tissue(4).warped = [0 0];
    matlabbatch{2}.spm.spatial.preproc.tissue(5).tpm = {strcat(spm('Dir'), '/tpm/TPM.nii,5')};
    matlabbatch{2}.spm.spatial.preproc.tissue(5).ngaus = 4;
    matlabbatch{2}.spm.spatial.preproc.tissue(5).native = [0 0];
    matlabbatch{2}.spm.spatial.preproc.tissue(5).warped = [0 0];
    matlabbatch{2}.spm.spatial.preproc.tissue(6).tpm = {strcat(spm('Dir'), '/tpm/TPM.nii,6')};
    matlabbatch{2}.spm.spatial.preproc.tissue(6).ngaus = 2;
    matlabbatch{2}.spm.spatial.preproc.tissue(6).native = [0 0];
    matlabbatch{2}.spm.spatial.preproc.tissue(6).warped = [0 0];
    matlabbatch{2}.spm.spatial.preproc.warp.mrf = 1;
    matlabbatch{2}.spm.spatial.preproc.warp.cleanup = 1;
    matlabbatch{2}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{2}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{2}.spm.spatial.preproc.warp.fwhm = 0;
    matlabbatch{2}.spm.spatial.preproc.warp.samp = 3;
    matlabbatch{2}.spm.spatial.preproc.warp.write = [0 0];


%     matlabbatch{1}.spm.spatial.preproc.channel.vols = {outputFile};
%     matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
%     matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
%     matlabbatch{1}.spm.spatial.preproc.channel.write = [0 0];
%     matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {strcat(spm('Dir'), '/tpm/TPM.nii,1')};
%     matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1;
%     matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 0];
%     matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [0 0];
%     matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {strcat(spm('Dir'), '/tpm/TPM.nii,2')};
%     matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1;
%     matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 0];
%     matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [0 0];
%     matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {strcat(spm('Dir'), '/tpm/TPM.nii,3')};
%     matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
%     matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 0];
%     matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [0 0];
%     matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {strcat(spm('Dir'), '/tpm/TPM.nii,4')};
%     matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
%     matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [0 0];
%     matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
%     matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {strcat(spm('Dir'), '/tpm/TPM.nii,5')};
%     matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
%     matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [0 0];
%     matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
%     matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {strcat(spm('Dir'), '/tpm/TPM.nii,6')};
%     matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
%     matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [0 0];
%     matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];
%     matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
%     matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
%     matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
%     matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
%     matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
%     matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
%     matlabbatch{1}.spm.spatial.preproc.warp.write = [0 0];
    
    m = matlabbatch;
end
