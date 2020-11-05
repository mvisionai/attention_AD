function runParallelSegmentation(baseDir)
    % i.e. runParallelSegmentation('/Users/<user>/Documents/Master/master-data/0-baseline')

    niiFiles = readNii(baseDir);
    [r, ~] = size(niiFiles);
    
    jobs = cell(r, 1);
    j = 1;
    for i = 1:r
        outputFolderName = '/1-Normalization-Segmentation';
        alignedFolderName = '/0-AC-PC-Aligned';
        
        r = strfind(niiFiles{i}, '_RAS.nii');
        s = strfind(niiFiles{i}, outputFolderName);
        
        if isempty(s)
            if ~isempty(r)
                jobs{j} = getJob({niiFiles{i}}, alignedFolderName, outputFolderName);
                j = j + 1;
            end
        end
    end
    
    jobs = jobs(1:j-1);
    [r, ~] = size(jobs); 
    
    % M specifies maximum number of workers
%     M = 4; 
%     poolobj = parpool(M);
%     parfor (i = 1:r)
%         spm('defaults', 'FMRI')
%         spm_jobman('run', jobs{i});
%     end
%     delete(poolobj)

    M = 4; 
    poolobj = parpool(M);
    
    % extracting brain for analysis
    parfor (i = 1:r)
        disp(['Counter Position: ' int2str(i)])
        file = erase(jobs{i,1}{1,1}.spm.spatial.normalise.estwrite.subj.vol{:}, ',1');
        
        disp(file)
        
        spm('defaults', 'FMRI')
        spm_jobman('run', jobs{i});
        
        
        [path, name, ext] = fileparts(file);
        c1 = niftiread(fullfile(path, ['c1w' name ext]));
        c2 = niftiread(fullfile(path, ['c2w' name ext]));
        c3 = niftiread(fullfile(path, ['c3w' name ext]));
        brain_mask = c1 + c2 + c3;

        index = find(brain_mask > 0);

        t1_hdr = spm_vol(fullfile(path, ['w' name ext]));
        t1 = spm_read_vols(t1_hdr);

        [x, y, z] = size(t1);

        brain = zeros(x, y, z);
        brain(index) = t1(index);
        %figure;
        % imagesc(brain(:,:,round(z/2)));
        % colormap gray

        output = t1_hdr;
        output.fname= fullfile(path, ['brain_' name ext]);
        spm_write_vol(output, brain);
        
        delete(file)
    end
    delete(poolobj)
end