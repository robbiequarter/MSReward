% ZIP_LOAD Load and format c3d files from zip archives created with 
%   Dexterit-E 3.0 and higher.
%
%   DATA = ZIP_LOAD opens all data files (.ZIP files) created by Dexterit-E
%   3.0 and higher that are in the currect directory and outputs the data
%   into the structure DATA.  Each element of DATA corresponds to a single
%   data file.    
%
%   DATA contains at least two fields:
%		.c3d - this field contains all of the trial data from the .C3D
%		files stored in the .ZIP file. The trials are sorted into the order
%		in which they were collected.
%		.filename - the filename of the .ZIP file
%
%   The format of the .c3d field is a structured array, each element of
%   which corresponds to a single trial.  The sub-fields of the .c3d
%   structure are of two types: time-varying data or a c3d Parameter Group.
%   Time-varying data are vectors of data corresponding to the field name.
%   Details of the time varying data can be found in the related Parameter
%   Group.  The Parameter Group fields have sub-fields of their own, most
%   of which are specific to each Parameter Group.
% 
%   DATA = ZIP_LOAD(ZIP_FILENAME) only opens ZIP_FILENAME.
%   ZIP_FILENAME can contain the '*' wildcard.
%   
%   DATA = ZIP_LOAD(ZIP_FILENAME1, ZIP_FILENAME2) opens ZIP_FILENAME1
%   and ZIP_FILENAME2 and outputs the data into the DATA structure.
%   ZIP_FILENAME1 and ZIP_FILENAME2 can both contain the % '*' wildcard.
%   Any number of filenames can be listed.  
%   
%   DATA = ZIP_LOAD('dir', DIRECTORY) looks for all .ZIP files in
%   DIRECTORY.
%   
%   DATA = ZIP_LOAD('c3d_filename', C3D_FILENAME) will load up only
%   those trials within specified zip files that correspond to
%   C3D_FILENAME.   C3D_FILENAME can contain the '*' wildcard.  
%  
%   NOTE: From the time Force/Torque sensors on KINARM End-Point robots
%   were introduced until Dexterit-E 3.4.2 there was a bug in the
%   calculation of TorqueX data of the Force/Torque sensor. This code
%   corrects those errors upon loading the data. If TorqueX data are not
%   found in the given data file then nothing is done. If the build TDK for
%   the given data file is  >=3.4.2 then nothing is done. 
%
%   NOTE: The TorqueY and TorqueZ data and all of the Force data from the
%   Force/Torque sensors are correct, only TorqueX is corrected.  

%   Copyright 2010-2021 BKIN Technologies Ltd

function c3dstruct = zip_load_impl(varargin)

x = 1;
num_files = 0;
newArgs = {};
c3d_filename = '*.c3d';		% by default, load all c3d files
startFolder = pwd;

while x <= length(varargin)
    % See if the user included a directory to look in
    if strncmpi(varargin{x}, 'dir', 3)
        x = x + 1;
        cd(varargin{x});
    elseif strncmpi(varargin{x}, 'c3d_filename', 12)
        x = x + 1;
        c3d_filename = varargin{x};
    elseif strncmpi(varargin{x}, 'ignore', 6)
        x = x + 1;
        newArgs = cat(2, newArgs, 'ignore');
        newArgs = cat(2, newArgs, varargin{x});
    elseif strncmpi(varargin{x}, 'keep', 4)
        x = x + 1;
        newArgs = cat(2, newArgs, 'keep');
        newArgs = cat(2, newArgs, varargin{x});
	else
		num_files = num_files + 1;
 		zipfiles{num_files} = varargin{x};
        varargin{x} = [];
    end
    x = x + 1;
end

% ensure that c3d_filename passed to c3d_load ends in .c3d
temp = strfind(c3d_filename, '.');
if isempty(temp)
	c3d_filename = [c3d_filename '.c3d'];
else
	c3d_filename = [c3d_filename(1:temp) 'c3d'];
end


if num_files > 0
	% check for '*' wild card in filename - expand file list if it exists
	for ii = num_files:-1:1
		if ~isempty(strfind(zipfiles{ii}, '*'))
			temp = dir(zipfiles{ii});
			zipfiles = [zipfiles strcat(pwd, '\', {temp.name})];
			%erase the filename with the wildcard
			zipfiles(ii) = [];		
		end
	end
	num_files = length(zipfiles);
	if num_files == 0
		disp(strvcat(' ','WARNING!!!  No zip files found.'));
		c3dstruct = [];
		return;
	end
else
	% Get all c3d files
	zipfiles = dir('*.zip');
	if isempty(zipfiles)
		disp(strvcat(' ','WARNING!!!  No zip files found in:', pwd));
		c3dstruct = [];
		return;
	end
    zipfiles = strcat(pwd, '\', {zipfiles.name});
end

cd(startFolder);
c3dstruct = [];
ME = [];

for x = 1:length(zipfiles)
	fprintf( 'Loading %s', zipfiles{x});
    start_time = tic;
	
    zipReader = [];
    
    try
        zipReader = ZipReader(zipfiles{x});

        if ~zipReader.containsFile('raw/common.c3d')
            fprintf('\nWARNING!!!  Not a Dexterit-E zip file: %s', zipfiles{x});
            continue
        end

        % get the common file so we can use it to place parameters in 
        % all other c3d structs.
        common_data = c3d_load(zipReader, 'raw/common.c3d', newArgs{:});

        % bulk load the c3d files
        zip_data.c3d = c3d_load(zipReader, c3d_filename, newArgs{:});    
        
        [path, fname, ext] = fileparts(zipfiles{x});
        fname = [fname ext];
        zip_data.filename = {fname};			% this returns a cell array, which is annoying, but is legacy and we don't want to remove so as to not break people's code
        zip_data.file_name = fname;			% this returns a string, which is more useful and consistent with file_label

        % If the end-user modified the name of the file in Dexterit-E UI, load and save it in a new field called filelabel
        filename = 'description.txt';
        if zipReader.containsFile(filename)
            fid = zipReader.fopen(filename);
            zip_data.file_label = freadstring(fid);
		end
		
		% if there are analysis results in the exam then load them, Use pwd because starting in R2018a, there is an analysis
        % folder on the MATLAB path
        analysis_files = zipReader.listAnalysis();
        if ~isempty(analysis_files)

            zip_data.analysis = c3d_load(zipReader, analysis_files{1}, newArgs{:});  % zip_data.analysis = c3d_load('analysis.c3d', newArgs{:});
            
            f = zipReader.fopen('analysis/app_version.txt');
            v1 = freadstring(f);

            zip_data.analysis.app_version = v1;
            f = zipReader.fopen('analysis/analysis_version.txt');
            v2 = freadstring(f);
            zip_data.analysis.analysis_version = v2;
        end

    catch ME
    end
    
    if ~isempty(zipReader)
        try
            zipReader.close();
        catch zrMR
        end   
    end
    
    if ~isempty(ME)
        rethrow(ME)
    end

    common_fields = fieldnames(common_data);

    % remove the ...HandX, ...HandY and FILENAME fields
    common_fields(strcmp('Right_HandX', common_fields)) = [];
    common_fields(strcmp('Right_HandY', common_fields)) = [];
    common_fields(strcmp('Left_HandX', common_fields)) = [];
    common_fields(strcmp('Left_HandY', common_fields)) = [];
    common_fields(strcmp('FILE_NAME', common_fields)) = [];

    %for each trial, add the common data back in
    for ii = 1:length(zip_data.c3d)
        for jj = 1:length(common_fields)
            zip_data.c3d(ii).(common_fields{jj}) = common_data.(common_fields{jj});
        end        
    end
    
    %Check for unimanual Left systems and correct the data
    for ii = 1:length(zip_data.c3d)
        if isfield(zip_data.c3d(ii).RIGHT_KINARM, 'IS_PRESENT') && ~zip_data.c3d(ii).RIGHT_KINARM.IS_PRESENT
            zip_data.c3d(ii).Left_HandX = zip_data.c3d(ii).Right_HandX;
            zip_data.c3d(ii).Left_HandY = zip_data.c3d(ii).Right_HandY;
            zip_data.c3d(ii).Right_HandX = zeros(length(zip_data.c3d(ii).Left_HandX), 1);
            zip_data.c3d(ii).Right_HandY = zeros(length(zip_data.c3d(ii).Left_HandX), 1);
        end
    end
    
    
    zip_data = correctXTorque(zip_data);
    c3dstruct = catStructs(c3dstruct, zip_data);
    end_time = toc(start_time);
    
	display(sprintf( ' %.2fs', end_time) );
end
% auto sort the trials into the order in which they were run
c3dstruct = sort_trials(c3dstruct);

% display(sprintf( 'Finished loading all exam files.') );
end

function out = catStructs(a, b)
% this function will concatenate two structures with different fields by
% creating the missing fields in each struct before concatenating them

	if isempty(a) 
		out = b;
	elseif isempty(b)
		out = a;
	else
		aNames = fieldnames(a);
		bNames = fieldnames(b);

		missingFromb = setdiff(aNames, bNames);
		missingFroma = setdiff(bNames, aNames);

		for ii = 1:length(missingFromb)
			b(1).(missingFromb{ii}) = [];
		end
		for ii = 1:length(missingFroma)
			a(1).(missingFroma{ii}) = [];
		end

		out = cat(1, a, b);
	end
	
end



function c3dstruct = c3d_load(zipReader, varargin)
    %C3D_LOAD Load and format c3d files.
    %   C3D_DATA = C3D_LOAD opens all .c3d files in the current directory and
    %   outputs the data into the C3D_DATA structure.%
    %   C3D_DATA is a structured array, each element of which corresponds to a
    %   single .c3d file.  The fields of the structure are of two types: 
    %   time-varying data or a c3d Parameter Group.  Time-varying data are
    %   vectors of data corresponding to the field name.  Details of the time
    %   varying data can be found in the related Parameter Group.  The
    %   Parameter Group fields have sub-fields of their own, most of which are
    %   specific to each Parameter Group.
    % 
    %   C3D_DATA = C3D_LOAD(FILENAME) opens FILENAME and
    %   outputs the data into the C3D_DATA structure.  FILENAME can contain the
    %   '*' wildcard.
    % 
    %   C3D_DATA = C3D_LOAD(FILENAME1, FILENAME2) opens FILENAME1 and FILENAME2
    %   and outputs the data into the C3D_DATA structure.  FILENAME1 and
    %   FILENAME2 can both contain the % '*' wildcard.  Any number of filenames
    %   can be listed.  
    % 
    %   C3D_DATA = C3D_LOAD('dir', DIRECTORY) looks for all .c3d files in
    %   DIRECTORY.
    % 
    %   C3D_DATA = C3D_LOAD('ignore', IGNORE) removes data fields containing
    %   the IGNORE string from the C3D_DATA structure.  IGNORE can either be a 
    %   string, or a cell array of strings.  This command is case insensitive.
    %   The special string 'PARAMETERS' can be used to ignore the c3d PARAMETER
    %   groups (i.e. all Parameter Groups will be removed from the C3D_DATA
    %   structure)
    % 
    %   C3D_DATA = C3D_LOAD('keep', KEEP) only keeps those data fields
    %   containing the KEEP string in the C3D_DATA structure.  KEEP can either
    %   be a string, or a cell array of strings.  This command is case
    %   insensitive.  The special string 'PARAMETERS' can be used to keep all
    %   of the c3d PARAMETER Groups (i.e. all Parameter Groups will be kept in
    %   the C3D_DATA structure)
    % 
    %   The above arguments can be combined in any combination and/or order,
    %   except for the 'ignore' and 'keep' arguments - both may not be present.
    %   For example, 
    %   C3D_DATA = C3D_LOAD(FILENAME1, 'dir', DIRECTORY, FILENAME2,...
    %   'keep',{'right', 'PARAMETERS') will load up the files FILENAME1 and
    %   FILENAME2 in the directory DIRECTORY, and will keep all data fields
    %   with the string 'right' in the field name and will keep all of the c3d
    %   Parameter Groups.

    % The decision to use genvarname is required many times - checking it once makes
    % the code run more efficiently.
    if verLessThan('matlab', '8.5.1')
        useGenVarName = true;
    else
        useGenVarName = false;
    end

    x = 1;
    num_files = 0;

    to_ignore = [];
    to_keep = [];

    while x <= length(varargin)
        if strncmpi(varargin{x}, 'ignore', 6)
            x = x + 1;
            to_ignore = varargin{x};   
        elseif strncmpi(varargin{x}, 'keep', 4)
            x = x + 1;
            to_keep = varargin{x};   
        else
            num_files = num_files + 1;
            c3dfiles{num_files} = varargin{x};
        end
        x = x + 1;
    end


    if num_files > 0
        % check for '*' wild card in filename - expand file list if it exists
        for ii = num_files:-1:1
            if ~isempty(strfind(c3dfiles{ii}, '*'))
                c3dfiles = zipReader.listTrials();
            end
        end
        num_files = length(c3dfiles);
        if num_files == 0
            disp(strvcat(' ','WARNING!!!  No c3d files found.'));
            c3dstruct = [];
            return;
        end
    else
        % Get all c3d files
        c3dfiles = dir('*.c3d');
        if isempty(c3dfiles)
            disp(strvcat(' ','WARNING!!!  No c3d files found in:', pwd));
            c3dstruct = [];
            return;
        end
        c3dfiles = {c3dfiles.name};
    end

    if ~isempty(to_ignore) && ~isempty(to_keep)
        disp('You can only specify the params/signals to keep OR to ignore, not both.');
        return;
    else
        if ~iscell(c3dfiles)
            c3dfiles = {c3dfiles};
        end

        % Read in each c3d file and organize into structure
        for x = 1:length(c3dfiles)

            c3d = c3d_load_single_file(zipReader, c3dfiles{x});
			
			if rem(x, 10) == 1
			    fprintf('.');
			end

            %check if data was loaded, otherwise proceed to next file
            if isempty(c3d.FileName)
                continue;				
            end

            load_parameters = 1;		%default is to load the c3d parameters
            % Get hand data
            c3dstruct(x).Right_HandX = c3d.Hand.RightX;
            c3dstruct(x).Right_HandY = c3d.Hand.RightY;
            c3dstruct(x).Left_HandX = c3d.Hand.LeftX;
            c3dstruct(x).Left_HandY = c3d.Hand.LeftY;

            % Get analog and kinematic signals
            indANA = find(strcmpi([c3d.ParameterGroup.name], 'Analog'));
            indLAB = find(strcmpi([c3d.ParameterGroup(indANA).Parameter.name], 'Labels'));
            % Get analog signal names and attach them to the analog signal data
            % (they are in the same order in the Parameters that they are in the
            % data matrix).
            sig_names = makeValidFieldName(c3d.ParameterGroup(indANA).Parameter(indLAB).data, useGenVarName);
            for y = 1:length(sig_names)
                %Typecast the force torque sensor status to uint32 so that the bits are interpreted
                %correctly. 
                %Convert the KINARM robot StatusBits to uint32.
                if strcmp(sig_names{y}, 'Right_FS_Status')
                    c3dstruct(x).Right_FS_Status = typecast(single( c3d.AnalogSignals(:, y) ), 'uint32');
                elseif strcmp(sig_names{y}, 'Left_FS_Status')
                    c3dstruct(x).Left_FS_Status = typecast(single( c3d.AnalogSignals(:, y) ), 'uint32');
                elseif strcmp(sig_names{y}, 'StatusBits')
                    c3dstruct(x).StatusBits = uint32( c3d.AnalogSignals(:, y) );
                else
                    c3dstruct(x).(sig_names{y}) = c3d.AnalogSignals(:, y);
                end
            end

            %keep only those data fields requested
            if ~isempty(to_keep)
                if ~iscell(to_keep)
                    to_keep = {to_keep};
                end
                fnames = fieldnames(c3dstruct);
                for ii = 1:length(fnames)
                    %for each field, check if it contains any of the 'keep'
                    %expressions 
                    if isempty( cell2mat( regexp(upper(fnames{ii}), upper(to_keep)) ) )
                        c3dstruct = rmfield(c3dstruct, fnames{ii});
                    end
                end
                load_parameters = ~isempty( cell2mat( regexp('PARAMETERS', upper(to_keep)) ) );
            end

            %remove those data fields requested
            if ~isempty(to_ignore)
                if ~iscell(to_ignore)
                    to_ignore = {to_ignore};
                end
                fnames = fieldnames(c3dstruct);
                for ii = 1:length(fnames)
                    %for each field, check if it contains any of the 'ignore' expressions 
                    if ~isempty( cell2mat( regexp(upper(fnames{ii}), upper(to_ignore)) ) )
                        c3dstruct = rmfield(c3dstruct, fnames{ii});
                    end
                end
                load_parameters = isempty( cell2mat( regexp('PARAMETERS', upper(to_ignore)) ) );
            end

            if load_parameters ==1
                % Go through all parameters and add them to structure, but ONLY if
                % the USED parameter for the Group is > 0
                AllParamGroupNames = [c3d.ParameterGroup.name];

                for y = 1:length(AllParamGroupNames)
                    add_ParamGroup = 1;		%default value....

                    if isempty(c3d.ParameterGroup(y).Parameter)  % Case added by Bretzke to push through Dex3.5 Beta testing (2015-01-14)
                       add_ParamGroup = 0;
                       disp(['Parameter group ' c3d.ParameterGroup(y).name ' is empty.']) 
                    else                
                        %is there a 'USED' parameter, and if so is it > 0?
                        USEDindex = strmatch('USED', [c3d.ParameterGroup(y).Parameter.name]);
                        if ~isempty(USEDindex)					
                            if c3d.ParameterGroup(y).Parameter(USEDindex).data == 0
                                add_ParamGroup = 0;		%do not add this Parameter Group
                            end
                        end
                    end

                    %add the ParameterGroup if USED > 0 (or if there is no 'USED'
                    %parameters)
                    if add_ParamGroup == 1
                        ParamGroupNameCell = makeValidFieldName(c3d.ParameterGroup(y).name(1), useGenVarName);
                        ParamGroupName = ParamGroupNameCell{1};
                        
                        clear DESCRIPTIONTEXT;		%clear this cell array before adding to it

                        %rename the 'POINT' ParameterGroup to 'HAND'
                        if strcmp(ParamGroupName, 'POINT')
                            ParamGroupName = 'HAND';
                        end
                        

                        for z = 1:length([c3d.ParameterGroup(y).Parameter.name])

                            if ~iscell(c3d.ParameterGroup(y).Parameter(z).name)
                                continue
                            end
                            ParameterNameCell = makeValidFieldName(c3d.ParameterGroup(y).Parameter(z).name, useGenVarName);
                            ParameterName = ParameterNameCell{1};

                            %if data is singleton cell array, remove cell array structure	
                            if ~isfield(c3d.ParameterGroup(y).Parameter(z), 'data')
                                c3d.ParameterGroup(y).Parameter(z).data = [];
                            end
                            data = c3d.ParameterGroup(y).Parameter(z).data;
                            if iscell(data) && length(data) == 1
                                c3dstruct(x).(ParamGroupName).(ParameterName) = data{1};
                            else
                                c3dstruct(x).(ParamGroupName).(ParameterName) = data;
                            end

                            %create cell array of Parameter descriptions, which is
                            %text that includes the ParameterName 
                            if isfield(c3d.ParameterGroup(y).Parameter(z), 'description') && ~isempty(c3d.ParameterGroup(y).Parameter(z).description)
                                description = c3d.ParameterGroup(y).Parameter(z).description{1};
                            else
                                description = '';
                            end
                            DESCRIPTIONTEXT(z) = {[ParameterName ' -- ' description]};

                        end

                        %Nominally, Parameter descriptions are not passed on to
                        %the final c3dstruct because in those cases the
                        %parameters are self-explanatory (e.g. 'UNITS',
                        %'DATA').  However, for some ParameterGroups, the data
                        %are self-explanatory, and instead it is the Parameters
                        %that are not.  In those cases, there is no Parameter
                        %called DESCRIPTIONS, so in those cases the description
                        %of each Parameter needs to be passed on.
                        DESCRIPTIONindex = strmatch('DESCRIPTIONS', [c3d.ParameterGroup(y).Parameter.name], 'exact');
                        if isempty(DESCRIPTIONindex)
                            c3dstruct(x).(ParamGroupName).DESCRIPTIONS = DESCRIPTIONTEXT;
                        end

                    end
                end

                %clean up event times.  Event times from Dexterit-E do not use the
                %first number of the two numbers stored for each event
                if isfield(c3dstruct(x), 'EVENTS') && ~isempty(c3dstruct(x).EVENTS)
                    if size(c3dstruct(x).EVENTS.TIMES,1) == 2 
                        c3dstruct(x).EVENTS.TIMES(1,:) = [];
                    end
                end

                %Check to see if events exist in the new Events and Ranges
                %section.  If so, then over-write any other events.
                if ~isempty(c3d.NEREvents)
                    c3dstruct(x).EVENTS = c3d.NEREvents;
                end	

                %Check to see if ranges exist in the new Events and Ranges
                %section.  If a Ranges section exists, then check if any have
                %'Video Frame' as the first part of the Label.  If so, then we
                %will assume that these have been used for recording the
                %confirmation of video display and that the start/stop times
                %were explicitly recorded, so those data are copied to a new
                %VIDEO_LATENCY field.  All other range information is copied to
                %a Ranges field. 
                if ~isempty(c3d.NERRanges)
                    Video_Frames = strncmp('Video Frame', c3d.NERRanges.LABELS, 11);
                    non_Video_Frames = not(Video_Frames);
                    if sum(non_Video_Frames) > 0
                        c3dstruct(x).Ranges.LABELS = c3d.NERRanges.LABELS(non_Video_Frames);
                        c3dstruct(x).Ranges.SEND_TIMES = c3d.NERRanges.STARTTIMES(non_Video_Frames);
                        c3dstruct(x).Ranges.ACK_TIMES = c3d.NERRanges.STOPTIMES(non_Video_Frames);
                        c3dstruct(x).Ranges.USED = sum(non_Video_Frames);
                    end
                    if sum(Video_Frames) > 0
                        c3dstruct(x).VIDEO_LATENCY.LABELS = c3d.NERRanges.LABELS(Video_Frames);
                        c3dstruct(x).VIDEO_LATENCY.SEND_TIMES = c3d.NERRanges.STARTTIMES(Video_Frames);
                        c3dstruct(x).VIDEO_LATENCY.ACK_TIMES = c3d.NERRanges.STOPTIMES(Video_Frames);
                        c3dstruct(x).VIDEO_LATENCY.USED = sum(Video_Frames);
                    end
                end	
            end

            % Add filename
            c3dstruct(x).FILE_NAME = char(c3d.FileName);

            % Find all of the fields with one string inside a cell array and just
            % "unwrap" that string so the field contains the string rather than the
            % cell array.
    %		c3dstructfields = fields(c3dstruct(x));
    %		for y = 1:length(c3dstructfields)
    %			if iscell(c3dstruct(x).(c3dstructfields{y})) && length(c3dstruct(x).(c3dstructfields{y})) == 1
    %				c3dstruct(x).(c3dstructfields{y}) = c3dstruct(x).(c3dstructfields{y}){1};
    %			end
    %		end

        end


    end

%     cd(olddir);
end

function newNames = makeValidFieldName(oldNames, useGenVarName)
	% Ensure that the  names are valid field names for MATLAB. Instead of using the
	% default for makeValidName, replace spaces with '_' and prefix invalid first
	% characters with 'x_'. 
	
	% this function assumes/requires a cell array as the input
	if isempty(oldNames)
		newNames = oldNames;
	else
		namesNoSpaces = regexprep(oldNames, ' ', '_');
		if useGenVarName 
			% if earlier than R2014a...
			% add valid prefix if first character is not a letter
			namesWithPrefix = cellfun(@(x) { [ regexprep(x(1), '[^a-zA-Z]', ['x_' x(1)] ) x(2:end) ] }, namesNoSpaces);
			% replace all non-valid characters with '_'
			newNames = regexprep(namesWithPrefix, '\W', '_');
        else
            nameCount = size(namesNoSpaces, 2);
            newNames = namesNoSpaces;
            tmpStruct = [];
            for i=1:nameCount
                try
                    % Try to use the name as a structure field. If this
                    % fails then call to generate a name. The name
                    % generation is quite slow, so we should only use it
                    % when required.
                    tmpStruct.(namesNoSpaces{i}) = 0;
                catch ME
                    % the following method was introduced in R2014a to replace genvarname
                    newNames(i) = matlab.lang.makeValidName(namesNoSpaces(i), 'prefix', 'x_');
                end
            end
		end
	end
end

function c3d = c3d_load_single_file(zipReader, FullFileName, varargin)

    %C3D_LOAD_SINGLE_FILE Load single c3d file.
    % 
    % 	Input:	FullFileName - file (including path) to be read
    % 
    % 	Output:   Data structure called 'c3d' with these fields:
    % 
    % 	c3d.Markers            3D-marker data [Nmarkers x NvideoFrames x Ndim(=3)]
    % 	c3d.VideoFrameRate     Frames/sec
    % 	c3d.AnalogSignals      Analog signals [Nsignals x NanalogSamples ]
    % 	c3d.AnalogFrameRate    Samples/sec
    % 	c3d.Event              Event(Nevents).time ..value  ..name
    % 	c3d.ParameterGroup     ParameterGroup(Ngroups).Parameters(Nparameters).data ..etc.
    % 	c3d.CameraInfo         MarkerRelated CameraInfo [Nmarkers x NvideoFrames]
    % 	c3d.ResidualError      MarkerRelated ErrorInfo  [Nmarkers x NvideoFrames]
    % 
    % 	AUTHOR(S) AND VERSION-HISTORY
    % 	Ver. 1.0 Creation (Alan Morris, Toronto, October 1998) [originally named "getc3d.m"]
    % 	Ver. 2.0 Revision (Jaap Harlaar, Amsterdam, april 2002)
    % 	LIMB Lab specifics (Jon Swaine, Kingston, December 2005)
    %   All subsequent modifications by BKIN Technologies

    BLOCK_SIZE = 512;

    c3d.Markers=[];
    c3d.VideoFrameRate=0;
    c3d.AnalogSignals=[];
    c3d.AnalogFrameRate=0;
    c3d.Event=[];							%Events stored in the c3d header
    c3d.NEREvents=[];						%Events stored in the new event and range section
    c3d.NERRanges=[];						%RANGES stored in the new event and range section
    c3d.ParameterGroup=[];
    c3d.CameraInfo=[];
    c3d.ResidualError=[];
    c3d.FileName = [];

    enable_waitbar = 0;

    % ###############################################
    % ##                                           ##
    % ##    open the file                          ##
    % ##                                           ##
    % ###############################################

    if isempty(strfind(FullFileName, '.c3d'))
        if isempty(strfind(FullFileName, '.'))
            FullFileName = [FullFileName '.c3d'];
        else
            disp(['WARNING!!! - ' FullFileName ' is not a c3d file']);
            return;
        end
    end


    ind=strfind(FullFileName,'/');
    if ind>0
        c3d.FileName=FullFileName(ind(length(ind))+1:length(FullFileName));
    else
        c3d.FileName=FullFileName;
    end

    % assume that data was saved on x86 processor (using IEEE little endian format)
    fid=zipReader.fopen(FullFileName);
    NrecordFirstParameterblock=fread(fid,1,'uint8');   % Reading record number of parameter section
    key=fread(fid,1,'int8');                           % key = 80;

    if key~=80
        h=errordlg(['File: ',FileName,' does not comply to the C3D format'],'application error');
        uiwait(h)
        fclose(fid);
        return
    end


    fseek(fid,BLOCK_SIZE*(NrecordFirstParameterblock-1)+3,'bof'); % jump to processortype - field
    proctype=fread(fid,1,'int8')-83;                       % proctype: 1(INTEL-PC); 2(DEC-VAX); 3(MIPS-SUN/SGI)

    if proctype==2
        fclose(fid);
        fid=fopen(FullFileName,'r','d'); % DEC VAX D floating point and VAX ordering
    end

    % ###############################################
    % ##                                           ##
    % ##    read header                            ##
    % ##                                           ##
    % ###############################################

    %NrecordFirstParameterblock=fread(fid,1,'int8');     % Reading record number of parameter section
    %key1=fread(fid,1,'int8');                           % key = 80;

    fseek(fid,2,'bof');

    Nmarkers=fread(fid,1,'int16');			        %number of markers
    NanalogSamplesPerVideoFrame=fread(fid,1,'int16');			%number of analog channels x #analog frames per video frame
    StartFrame=fread(fid,1,'int16');		        %# of first video frame
    EndFrame=fread(fid,1,'uint16');			        %# of last video frame

    % Value not used
    MaxInterpolationGap=fread(fid,1,'int16');		%maximum interpolation gap allowed (in frame)

    Scale=fread(fid,1,'float32');			        %floating-point scale factor to convert 3D-integers to ref system units

    NrecordDataBlock=fread(fid,1,'int16');			%starting record number for 3D point and analog data

    NanalogFramesPerVideoFrame=fread(fid,1,'int16');
    if NanalogFramesPerVideoFrame > 0
        NanalogChannels=NanalogSamplesPerVideoFrame/NanalogFramesPerVideoFrame;
    else
        NanalogChannels=0;
    end

    c3d.VideoFrameRate=fread(fid,1,'float32');
    c3d.AnalogFrameRate=c3d.VideoFrameRate*NanalogFramesPerVideoFrame;



    % ###############################################
    % ##                                           ##
    % ##    read events                            ##
    % ##                                           ##
    % ###############################################

    fseek(fid,298,'bof');
    EventIndicator=fread(fid,1,'int16');
    if EventIndicator==12345
        Nevents=fread(fid,1,'int16');
        fseek(fid,2,'cof'); % skip one position/2 bytes
        if Nevents>0
            for i=1:Nevents
                c3d.Event(i).time=fread(fid,1,'float');
            end
            fseek(fid,188*2,'bof');
            for i=1:Nevents
                c3d.Event(i).value=fread(fid,1,'int8');
            end
            fseek(fid,198*2,'bof');
            for i=1:Nevents
%                 c3d.Event(i).name=cellstr(fread(fid,4,'*char'));
                c3d.Event(i).name={fread(fid,4,'*char')};
            end
        end
    end


    % ###############################################
    % ##                                           ##
    % ##    read 1st parameter block               ##
    % ##                                           ##
    % ###############################################

    nStartParamRead = BLOCK_SIZE*(NrecordFirstParameterblock-1);
    if nStartParamRead == 0
        %for some reason the location of the parameters can sometimes be wrong,
        %assume it is at least at the first block.
        nStartParamRead = BLOCK_SIZE;
    end
    fseek(fid, nStartParamRead, 'bof');

    dat1=fread(fid,1,'int8');
    key2=fread(fid,1,'int8');                   % key = 80;
    nParameterBlocks=fread(fid,1,'int8');
    proctype=fread(fid,1,'int8')-83;            % proctype: 1(INTEL-PC); 2(DEC-VAX); 3(MIPS-SUN/SGI)


    Ncharacters=fread(fid,1,'int8');   			% characters in group/parameter name
    GroupNumber=fread(fid,1,'int8');			% id number -ve=group / +ve=parameter

    %sometimes this may be invalid, if it is then just don't read past the end
    %of the file.
    if (nParameterBlocks <= 0)
%         fileInfo  = dir(FullFileName);
%         nEndReadPos = fileInfo.bytes;
        nEndReadPos = flength(fid);
    else
        %find the extend of the reading we should do in the file.
        nEndReadPos = nStartParamRead + (nParameterBlocks * BLOCK_SIZE);
    end

    while GroupNumber ~= 0% The end of the parameter record is indicated by <0 characters for group/parameter name

        if GroupNumber<0 % Group data
            GroupNumber=abs(GroupNumber);
            GroupName=fread(fid,[1,Ncharacters],'*char');
%             c3d.ParameterGroup(GroupNumber).name=cellstr(GroupName);	%group name
            c3d.ParameterGroup(GroupNumber).name={GroupName};	%group name
            offset=fread(fid,1,'int16');							%offset in bytes
            deschars=fread(fid,1,'int8');       					%description characters
            if deschars > 0
                GroupDescription=fread(fid,[1,deschars],'*char');
            else
                GroupDescription = '';
            end
%             c3d.ParameterGroup(GroupNumber).description=cellstr(GroupDescription); %group description
            c3d.ParameterGroup(GroupNumber).description={GroupDescription}; %group description

            ParameterNumberIndex(GroupNumber)=0;
            fseek(fid,offset-3-deschars,'cof');


        else % parameter data
            clear dimension;
            ParameterNumberIndex(GroupNumber)=ParameterNumberIndex(GroupNumber)+1;
            ParameterNumber=ParameterNumberIndex(GroupNumber);              % index all parameters within a group

            if Ncharacters > 0
                ParameterName=fread(fid,[1,Ncharacters],'*char');				% name of parameter
            else
                ParameterName = '';
            end

            % read parameter name
            if size(ParameterName)>0
                c3d.ParameterGroup(GroupNumber).Parameter(ParameterNumber).name={ParameterName};	%save parameter name
            end

            % read offset
            offset=fread(fid,1,'uint16');							%offset of parameters in bytes
            filepos=ftell(fid);										%present file position
            nextrec=filepos+offset(1)-2;							%position of beginning of next record


            % read type
            type=fread(fid,1,'int8');     % type of data: -1=char/1=byte/2=integer*2/4=real*4
            c3d.ParameterGroup(GroupNumber).Parameter(ParameterNumber).datatype=type;


            % read number of dimensions
            dimnum=fread(fid,1,'int8');
            if dimnum==0
                datalength=abs(type);								%length of data record
            else
                mult=1;
                for j=1:dimnum
                    dimension(j)=fread(fid,1,'uint8');   % 17-Feb-2009 reads each dimension as an unsigned int to handle dimensions > 127
                    mult = mult * dimension(j);
                    c3d.ParameterGroup(GroupNumber).Parameter(ParameterNumber).dim(j)=dimension(j);  %save parameter dimension data
                end
                datalength=abs(type)*mult;							%length of data record for multi-dimensional array
            end

            if type==-1 %datatype=='char'

                wordlength=dimension(1);	%length of character word
                if dimnum==2 && datalength > 0
                    data=fread(fid,dimension,'*char');
                    c3d.ParameterGroup(GroupNumber).Parameter(ParameterNumber).data = cellstr(data')';
                elseif dimnum==1 && datalength>0
                    data=fread(fid,[1,wordlength],'*char');		%numerical data record of 1-D array
                    c3d.ParameterGroup(GroupNumber).Parameter(ParameterNumber).data={data};
                elseif dimnum > 2 && datalength > 0
                    data = fread(fid, datalength, '*char');
                    data = reshape(data, dimension);
                    if dimnum == 3  %character word data record for 3-D array
                       for j = 1:dimension(3)
                          for k = 1:dimension(2)
                             c3d.ParameterGroup(GroupNumber).Parameter(ParameterNumber).data{j,k} = strtrim(char(cellstr(data(:,k,j)))');
                          end
                       end
                    else  % greater than 3-D will not be formatted into cell strings and dimensions will remain in Fortran-style order
                        c3d.ParameterGroup(GroupNumber).Parameter(ParameterNumber).data = data;
                    end
                end


            elseif type==1    %1-byte for boolean

                Nparameters=datalength/abs(type);
                data=fread(fid,Nparameters,'int8');
                c3d.ParameterGroup(GroupNumber).Parameter(ParameterNumber).data=data;

            elseif type==2 && datalength>0			%integer

                Nparameters=datalength/abs(type);
                if (strcmp(c3d.ParameterGroup(GroupNumber).name, 'POINT') && ...
                    strcmp(c3d.ParameterGroup(GroupNumber).Parameter(ParameterNumber).name, 'FRAMES'))  || ... % Fixed by HBretzke 20-Jan-2009 to handle files with POINT.FRAMES > 32767
                    strcmp(c3d.ParameterGroup(GroupNumber).name, 'EVENTS')  % Fixed by HBretzke 17-Feb-2009 to handle files > 127 events
                    data=fread(fid,Nparameters,'uint16');
                else                
                    data=fread(fid,Nparameters,'int16'); 
                end
                if dimnum>1
                    c3d.ParameterGroup(GroupNumber).Parameter(ParameterNumber).data=reshape(data,dimension);
                else
                    c3d.ParameterGroup(GroupNumber).Parameter(ParameterNumber).data=data;
                end

            elseif type==4 && datalength>0

                Nparameters=datalength/abs(type);
                data=fread(fid,Nparameters,'float32');
                if dimnum > 1
                    c3d.ParameterGroup(GroupNumber).Parameter(ParameterNumber).data=reshape(data,dimension);
                else
                    c3d.ParameterGroup(GroupNumber).Parameter(ParameterNumber).data=data';
                end
            else
                % error
            end

    %%%        deschars=fread(fid,1,'int8');							%description characters
    % in order to handle descriptions longer than 128 characters, change 'int8'
    % to uint8.
            deschars=fread(fid,1,'uint8');      %description characters
            if deschars>0
               description=fread(fid,[1,deschars],'*char');
%                c3d.ParameterGroup(GroupNumber).Parameter(ParameterNumber).description=cellstr(description);
               c3d.ParameterGroup(GroupNumber).Parameter(ParameterNumber).description={description};
            end
            %moving ahead to next record
            fseek(fid,nextrec,'bof');
        end

        %if we have read to the end of the parameter block we can stop
        if (ftell(fid) >= nEndReadPos)
            break;
        end

        % check group/parameter characters and idnumber to see if more records present
        Ncharacters=fread(fid,1,'int8');   			% characters in next group/parameter name
        GroupNumber=fread(fid,1,'int8');			% id number -ve=group / +ve=parameter
    end


    % ###############################################
    % ##                                           ##
    % ##    read data block                        ##
    % ##                                           ##
    % ###############################################
    %  Get the coordinate and analog data

    seekToPos = (NrecordDataBlock-1) * BLOCK_SIZE;
    if flength(fid) > seekToPos
        fseek(fid, seekToPos, 'bof');
    end

    if enable_waitbar
        h = waitbar(0,[FileName,' is loading...']);
    end

    % To determine how much data to expect, check for the fix for >= 65535
    % frames. If the last frame number is 65535, the actual last frame number
    % is stored as a float in the parameter LONG_FRAMES in the group POINT. We
    % test for the presence of this parameter first, however, in case the C3D
    % was not produced by software that uses this system (such as BKIN's).
    idxPointGrp = strmatch('POINT', [c3d.ParameterGroup.name]);
    if EndFrame == 65535      
        if idxPointGrp > 0
            j = strmatch('LONG_FRAMES', [c3d.ParameterGroup(idxPointGrp).Parameter.name]);

            if j > 0
                EndFrame = c3d.ParameterGroup(idxPointGrp).Parameter(j).data;
            end
        end
    end

    % for some reason the start frame is normally 1, so we need to subtract
    % that off
    NvideoFrames = EndFrame - StartFrame;
    if StartFrame ~= 0
        NvideoFrames = NvideoFrames + 1;
    end

    % Get the number of analog signals (usually 22);
    % Find the ANALOG group, default group id is 2
    % HB 2011-11-28 determine number of channels to be read dynamically
    % Sometimes there is no hand trajectory data in c3d files.
    DEFAULT_IDX_ANA_GROUP = 2;
    NUM_CHANNELS_PER_TRAJECTORY = 4;

    idxAnaGrp = DEFAULT_IDX_ANA_GROUP;
    if isempty(strmatch('ANALOG', c3d.ParameterGroup(idxAnaGrp).name))
        % Look for the Analog group
        idxAnaGrp = strmatch('ANALOG', [c3d.ParameterGroup(:).name]);
    end
    if ~isempty(idxAnaGrp)  % ANALOG group is required, but just in case...
        idx = strmatch('USED', [c3d.ParameterGroup(idxAnaGrp).Parameter.name]);
        num_anasigs = c3d.ParameterGroup(idxAnaGrp).Parameter(idx).data; % Fixed by HBretzke 18-Nov-2008


        idx = strmatch('USED', [c3d.ParameterGroup(idxPointGrp).Parameter.name]);
        num_traj = c3d.ParameterGroup(idxPointGrp).Parameter(idx).data;
        num_matrix_columns = (num_traj * NUM_CHANNELS_PER_TRAJECTORY + num_anasigs);    

        % This part of the original code was far too slow as the binary freads were
        % being done in a loop up to 60000 times.  So I altered the code to read it
        % all in in one big chunk and after that, the data is reorganized.  The
        % speed difference was enormous.  --JS
        if Scale < 0
            % Read all signals in and then deal with them.
            % 1 - Right Hand X
            % 2 - Right Hand Y
            % 3 - Right Hand Z
            % 4 - Right Hand CAMERA INFO (NOT NEEDED)
            % 5 - Left Hand X
            % 6 - Left Hand Y
            % 7 - Left Hand Z
            % 8 - Left Hand CAMERA INFO (NOT NEEDED)
            % 9 - 30 - Analog Signals (listed in order in ParameterGroup(2))        

            if num_matrix_columns*NvideoFrames > 0
                rawRd = fread(fid, num_matrix_columns*NvideoFrames, 'float32');
            else
                rawRd = [];
            end
            try
                data_matrix = reshape(rawRd, num_matrix_columns, NvideoFrames)';
            catch ME
                print ME
            end
            % TODO If only one arm, figure out which one and store trajectory data into the
            % struct.
            if num_traj == 2
                c3d.Hand.RightX = data_matrix(:,1);
                c3d.Hand.RightY = data_matrix(:,2);
                c3d.Hand.RightZ = data_matrix(:,3);
                % 4th column is not needed
                c3d.Hand.LeftX = data_matrix(:,5);
                c3d.Hand.LeftY = data_matrix(:,6);
                c3d.Hand.LeftZ = data_matrix(:,7);
                % 8th column is not needed
            else
                c3d.Hand.RightX = [];
                c3d.Hand.RightY = [];
                c3d.Hand.RightZ = [];            
                c3d.Hand.LeftX = [];
                c3d.Hand.LeftY = [];
                c3d.Hand.LeftZ = [];           
            end       
            % Start after reading hand info (9) and get all analog signals.
            c3d.AnalogSignals = data_matrix(:, num_traj * NUM_CHANNELS_PER_TRAJECTORY + 1:num_matrix_columns);
        end
    else
        for i=1:NvideoFrames
            for j=1:Nmarkers
                c3d.Markers(i,j,1:3)=fread(fid,3,'float32')'.*Scale;
                c3d.ResidualError(i,j)=fread(fid,1,'int8');
                c3d.CameraInfo(i,j)=fread(fid,1,'int8');
            end
            if enable_waitbar
                waitbar(i/NvideoFrames)
            end
            for j=1:NanalogFramesPerVideoFrame
                c3d.AnalogSignals(j+NanalogFramesPerVideoFrame*(i-1),1:NanalogChannels)=...
                    fread(fid,NanalogChannels,'float32')';
            end
        end
    end


    % ###############################################
    % ##                                           ##
    % ##    read new event and range section       ##
    % ##                                           ##
    % ###############################################

    fseek(fid,294,'bof');
    EventAndRangeIndicator=fread(fid,1,'int16');

    if EventAndRangeIndicator == 12345
        StartrecordNERBlock=fread(fid,1,'uint16');			%starting record number for new event and range data
        fseek(fid,(StartrecordNERBlock-1) * BLOCK_SIZE, 'bof');

        key=fread(fid,1,'int8');                           % key = 90;
        if key~=90
            h=errordlg(['File: ',FileName,' does not comply to the C3D format'],'application error');
            uiwait(h)
            fclose(fid);
            return
        end

        NNERRecords=fread(fid,1,'uint16');				%Total number of records in the NER section
        NEvents=fread(fid,1,'uint16');					%Number of EVENT units in the NER section
        NCharEventLabel=fread(fid,1,'uint16');			%Number of characters to store each EVENT label
        NRanges=fread(fid,1,'uint16');					%Number of Range units in the NER section
        NCharRangeLabel=fread(fid,1,'uint16');			%Number of characters to store each Range label

        % In Dexterit-E 2.3.0 and 2.3.1, the events in this section not related to video latency were 
        % written with times in milliseconds, not seconds. Here we check for such a possibility, and 
        % flag that we must correct the event times if necessary.
        fixEventTimes = 0;

        for j = 1:length(c3d.ParameterGroup)
            if strcmp(c3d.ParameterGroup(j).name, 'MANUFACTURER')
                for k = 1:length(c3d.ParameterGroup(j).Parameter)
                    if strcmp(c3d.ParameterGroup(j).Parameter(k).name, 'SOFTWARE_VERSION')
                        if strcmp(c3d.ParameterGroup(j).Parameter(k).data, 'BKIN Dexterit-E 2.3.0') || ...
                            strcmp(c3d.ParameterGroup(j).Parameter(k).data, 'BKIN Dexterit-E 2.3.1')
                            fixEventTimes = 1;
                        end
                    end
                end
            end
        end

        for i = 1:NEvents
            c3d.NEREvents.LABELS{i} = strtrim(fread(fid,NCharEventLabel,'*char'));
            c3d.NEREvents.TIMES(i) = fread(fid,1,'float32');

            if fixEventTimes
                c3d.NEREvents.TIMES(i) = c3d.NEREvents.TIMES(i) / 1000.0;
            end

            c3d.NEREvents.DISPLAYSWITCH(i) = fread(fid,1,'int8');
            fseek(fid,4,'cof'); % skip 4 bytes
        end
        c3d.NEREVENTS.USED = NEvents;

    % %		This older code was very slow. The replacement code below is ~10x
    % %		faster
    % %
    % % 	positionNERStart = ftell(fid);
    % % 	for i = 1:NRanges
    % % 		c3d.NERRangesOLD.LABELS{i} = strtrim(char(fread(fid,NCharRangeLabel,'char')'));
    % % 		c3d.NERRangesOLD.STARTEVENTFLAG(i) = fread(fid,1,'int8');
    % % 		c3d.NERRangesOLD.STARTEVENTLABEL{i} = strtrim(char(fread(fid,NCharEventLabel,'char')'));
    % % 		c3d.NERRangesOLD.STARTEVENTINSTANCE(i) = fread(fid,1,'uint16');
    % % 		c3d.NERRangesOLD.STARTTIMES(i) = fread(fid,1,'*float');
    % % 		c3d.NERRangesOLD.STOPEVENTFLAG(i) = fread(fid,1,'int8');
    % % 		c3d.NERRangesOLD.STOPEVENTLABEL{i} = strtrim(char(fread(fid,NCharEventLabel,'char')'));
    % % 		c3d.NERRangesOLD.STOPEVENTINSTANCE(i) = fread(fid,1,'uint16');
    % % 		c3d.NERRangesOLD.STOPTIMES(i) = fread(fid,1,'float');
    % % 		c3d.NERRangesOLD.USED = NRanges;
    % % 	end
    % % 
    % % 	fseek(fid, positionNERStart, 'bof');

        % To speed up this process, read all data at once and then parse it in
        % memory
        if NRanges > 0
            bytesPerRange = 14 + NCharRangeLabel + 2 * NCharEventLabel;
            NERdata = fread(fid, [bytesPerRange NRanges], '*uint8')';

            temp = NERdata( :, 1:NCharRangeLabel );
            c3d.NERRanges.LABELS				= strtrim( cellstr( char( temp) ) )';

            temp = NERdata( :, NCharRangeLabel + 1 );
            c3d.NERRanges.STARTEVENTFLAG		= typecast( temp, 'INT8');

            temp = NERdata( :, NCharRangeLabel + 1 + (1:NCharEventLabel) );
            c3d.NERRanges.STARTEVENTLABEL		= strtrim( cellstr( char( temp ) ) )';

            temp = NERdata( :, NCharRangeLabel + 1 + NCharEventLabel + (1:2) );
            c3d.NERRanges.STARTEVENTINSTANCE	= typecast( reshape(temp', 1, []), 'UINT16');

            temp = NERdata( :, NCharRangeLabel + 3 + NCharEventLabel + (1:4) );
            c3d.NERRanges.STARTTIMES	= typecast( reshape(temp', 1, []), 'SINGLE');

            temp = NERdata( :, NCharRangeLabel + 8 + NCharEventLabel );
            c3d.NERRanges.STOPEVENTFLAG	= typecast( temp, 'INT8');

            temp = NERdata( :, NCharRangeLabel + 8 + NCharEventLabel + (1:NCharEventLabel) );
            c3d.NERRanges.STOPEVENTLABEL		= strtrim( cellstr( char( temp ) ) )';

            temp = NERdata( :, NCharRangeLabel + 8 + 2 * NCharEventLabel + (1:2) );
            c3d.NERRanges.STOPEVENTINSTANCE	= typecast( reshape(temp', 1, []), 'UINT16');

            temp = NERdata( :, NCharRangeLabel + 10 + 2 * NCharEventLabel + (1:4) );
            c3d.NERRanges.STOPTIMES	= typecast( reshape(temp', 1, []), 'SINGLE');

            c3d.NERRanges.USED = NRanges;
        end


    end


    if enable_waitbar
        close(h); % waitbar
    end

%     fclose(fid);

%     fprintf('.');
    % disp(['Done reading ', FullFileName]);

end


function data_out = correctXTorque(data_in)
% correctXTorque - Correct the TorqueX data from the Force/Torque sensors in 
% an EP robot.  
%
% From the time Force/Torque sensors were introduced until Dexterit-E 3.4.2
% there was a bug in the calculation of TorqueX data of the Force/Torque 
% sensor. This code corrects those errors. If TorqueX data are not 
% found in the given data file then nothing is done. If the build TDK for the 
% given data file is >=3.4.2 then nothing is done.
%
% NOTE: The TorqueY and TorqueZ data and all of the Force data from the 
% Force/Torque sensors are correct, only TorqueX needs correction. 
    data_out = data_in;
    
    if isfield(data_in, 'c3d')
        for n=1:length(data_in.c3d)
            [bCorrected, R,L] = correctTorqueInTrial(data_in.c3d(n));
            if bCorrected
                data_out.c3d(n).Right_FS_TorqueX = R;
                data_out.c3d(n).Left_FS_TorqueX = L;
                data_out.c3d(n).torqueCorrected = 1;
            end
        end
    else
        [bCorrected, R,L] = correctTorqueInTrial(data_in);
        if bCorrected
            data_out.Right_FS_TorqueX = R;
            data_out.Left_FS_TorqueX = L;
            data_out.torqueCorrected = 1;
        end
    end
    
    function [bRequiresFix, R_TX, L_TX] = correctTorqueInTrial(trial)
        bRequiresFix = (isfield(trial, 'Right_FS_TorqueX') || isfield(trial, 'Left_FS_TorqueX')) && ~isfield(trial, 'torqueCorrected');
        
        %The bug was fixed in 3.4.2. Any version before 3.4.0 does not have
        %the build tdk, so we can assume it's wrong. 
        if bRequiresFix && isfield(trial.EXPERIMENT, 'TASK_PROGRAM_BUILD_TDK')
            parts = sscanf(trial.EXPERIMENT.TASK_PROGRAM_BUILD_TDK, '%d.%d.%d');
            
            if parts(1) > 3
                bRequiresFix = 0;
            elseif parts(1) == 3
                if parts(2) > 4
                    bRequiresFix = 0;
                elseif parts(2) == 4
                    if parts(3) >= 2
                        bRequiresFix = 0;
                    end
                end
            end
        end
        
        R_TX = [];
        L_TX = [];
        
        if bRequiresFix == 0
			% if no correction required, then use the original, uncorrected data
            if isfield(trial, 'Right_FS_TorqueX')
                R_TX = trial.Right_FS_TorqueX;
            end
            
            if isfield(trial, 'Left_FS_TorqueX')
                L_TX = trial.Left_FS_TorqueX;
            end
            return
        end
        
        if isfield(trial, 'Right_FS_TorqueX') && ~isempty(trial.Right_FS_TorqueX)
            R_TX = correctTorque(trial.Right_L2Ang, trial.Right_FS_TorqueX, trial.Right_FS_TorqueY);
        end

        if isfield(trial, 'Left_FS_TorqueX') && ~isempty(trial.Left_FS_TorqueX)
            L_TX = correctTorque(trial.Left_L2Ang, trial.Left_FS_TorqueX, trial.Left_FS_TorqueY);
        end

    end

    function torqueX = correctTorque(L2_angle, FS_TorqueX, FS_TorqueY)
        force_sensor_angle_offset = 29.0 * pi/180.0;  %angle offset is always 29 degrees in this older data
        
		% calculate the angles of F/T sensor local coordinate system relative the global coordinate frame
        sensor_u_angle = L2_angle - force_sensor_angle_offset;
        sensor_v_angle = L2_angle - force_sensor_angle_offset - pi/2;
        
		% calculate what the torques were in the original F/T sensor local coordinate frame
        force_sensor_torque_u = (FS_TorqueY.*cos(sensor_v_angle) - FS_TorqueX.*sin(sensor_v_angle)) ./ (sin(sensor_u_angle).*(cos(sensor_v_angle)-sin(sensor_v_angle)));
        force_sensor_torque_v = (FS_TorqueX - FS_TorqueY) ./ (cos(sensor_v_angle)-sin(sensor_v_angle));
        
		% re-calculate what TorqueX is in the global coordinate frame
        torqueX = force_sensor_torque_u .* cos(sensor_u_angle) + force_sensor_torque_v .* cos(sensor_v_angle);
    end
end
