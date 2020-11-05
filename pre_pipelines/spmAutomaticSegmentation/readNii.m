
function r = readNii(rootFolder)
    
    topPath = fullfile(rootFolder);
    
    % "fileNames" - output argument of VISITOR2 function
    fileNmes = dirwalk(topPath, @visitor2, '^.*\.nii$', '^.*\.nii$', '^.*\.nii$');

    % All files *.c, *.cpp, *.h
    r = vertcat({}, fileNmes{:});
    
    function fileNames = visitor2(rootPath, Listing, varargin)
        %VISITOR2 Visitor function
        %
        % Description:
        %   Select files on regexp paterns matching.
        %
        % Input:
        %   rootPath -- Path to visited directory. String
        %   Listing  -- Visited directory listing. Array of structs (output of function DIR)
        %   varargin -- Regexp paterns
        %

        % Check number of output arguments
        error(nargoutchk(0, 1, nargout))

        names = {Listing.name}';
        isDirs = [Listing.isdir];
        fileNames = names(~isDirs);

        pInds = zeros(numel(fileNames), 1);

        paterns = varargin;

        for i = 1:numel(paterns)
            matchNames = regexp(fileNames, paterns{i}, 'once', 'match');
            cInds = ~cellfun('isempty', matchNames);
            pInds = pInds | cInds;
        end

        fileNames = cellfun(@(x) fullfile(rootPath, x), fileNames(pInds), 'UniformOutput', 0);

    end
        
end

