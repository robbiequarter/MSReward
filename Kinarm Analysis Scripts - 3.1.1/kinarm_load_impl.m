% KINARM_LOAD_IMPL An internal function for reading .kinarm files.
function c3dstruct = kinarm_load_impl(varargin)
    x = 1;
    num_files = 0;
    newArgs = {};
    startFolder = pwd;

    while x <= length(varargin)
        % See if the user included a directory to look in
        if strncmpi(varargin{x}, 'dir', 3)
            x = x + 1;
            cd(varargin{x});
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
            fprintf('\nWARNING!!!  No zip files found.\n');
            c3dstruct = [];
            return;
        end
    else
        % Get all c3d files
        zipfiles = dir('*.kinarm');
        if isempty(zipfiles)
            fprintf('\nWARNING!!!  No zip files found in: %s\n', pwd);
            c3dstruct = [];
            return;
        end
        zipfiles = strcat(pwd, '\', {zipfiles.name});
    end

    cd(startFolder);
    c3dstruct = [];
    ME = [];

    for x = 1:length(zipfiles)
        
        if exist(zipfiles{x}, 'file') ~= 2
           error(['File not found: ' zipfiles{x}]);
        end
        
        fprintf( 'Loading %s', zipfiles{x});
        start_time = tic;

        zipReader = [];
        try
            zipReader = ZipReader(zipfiles{x});

            if ~zipReader.containsFolder('raw/common')
                fprintf('\nWARNING!!!  Not a Dexterit-E zip file: %s\n', zipfiles{x});
                continue
            end

            % get the common file so we can use it to place parameters in 
            % all other c3d structs.
            common_data = c3d_load(zipReader, 'raw/common');

            % bulk load the c3d files
            zip_data.c3d = c3d_load(zipReader, newArgs{:});   
            [~, fname, ext] = fileparts(zipfiles{x});
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


%         zip_data = correctXTorque(zip_data);
        c3dstruct = catStructs(c3dstruct, zip_data);
        end_time = toc(start_time);

        display(sprintf( ' %.2fs', end_time) );
    end
    % auto sort the trials into the order in which they were run
    c3dstruct = sort_trials(c3dstruct);

%     display(sprintf( 'Finished loading all exam files.') );
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
    to_ignore = [];
    to_keep = [];
    c3dfiles = {};

    while x <= length(varargin)
        if strncmpi(varargin{x}, 'ignore', 6)
            x = x + 1;
            to_ignore = varargin{x};   
        elseif strncmpi(varargin{x}, 'keep', 4)
            x = x + 1;
            to_keep = varargin{x};   
        else            
            c3dfiles{end+1} = varargin{x};
        end
        x = x + 1;
    end

    if ~isempty(to_ignore) && ~isempty(to_keep)
        disp('You can only specify the params/signals to keep OR to ignore, not both.');
        return;
    end
    
    if isempty(c3dfiles)
        c3dfiles = zipReader.listTrials();
        if isempty(c3dfiles)
            fprintf('\nWARNING!!!  No trials found in: %s\n', pwd);
            c3dstruct = [];
            return;
        end
    end

    c3dstruct = [];
    
    % Read in each c3d file and organize into structure
    for x = 1:length(c3dfiles)

        c3d = load_trial_folder(zipReader, c3dfiles{x});

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

        if ~isempty(c3d.AnalogSignals)            
            for fn=fieldnames(c3d.AnalogSignals)'
                data = c3d.AnalogSignals.(fn{1});
                if strcmp(fn{1}, 'Right_FS_Status') || strcmp(fn, 'Left_FS_Status')
                    data = typecast(single(data), 'uint32');
                elseif strcmp(fn, 'StatusBits')
                    data = uint32( data );
                end            
                c3dstruct(x).(fn{1}) = data;
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
    end
end

function c3d = load_trial_folder(zipReader, FullFolderName, varargin)

    %LOAD_TRIAL_FOLDER Load single trial.
    % 
    % 	Input:	FullFolderName - file (including path) to be read
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
    % 	This code is loosely based on c3d reading code only because we
    % 	wanted to maintain compatability with older Kinarm exam files.

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


    % ###############################################
    % ##                                           ##
    % ##    open the file                          ##
    % ##                                           ##
    % ###############################################

    ind=strfind(FullFolderName,'/');
    if ind>0
        c3d.FileName=FullFolderName(ind(length(ind))+1:length(FullFolderName));
    else
        c3d.FileName=FullFolderName;
    end

    fileList = zipReader.readFolder(FullFolderName);
    
    % assume that data was saved on x86 processor (using IEEE little endian format)
    dis=fileList([FullFolderName '/header.bin']);
    fver = readInt(dis);
    if fver < 2 || fver > 3
        fprintf('\nInvalid header file format!! %d\n', fver);
        return
    end
    
    frameCount = readInt(dis);
    channelCount = readInt(dis);
    positionCount = readInt(dis);
    eventCount = readInt(dis);
    ackCount = readInt(dis);
    
    c3d.VideoFrameRate = readFloats(dis, 1);
    name = readString(dis);
    
    if channelCount == 0
        kinematicsNames = {};
    else
        kinematicsNames{channelCount} = [];
        for i=1:channelCount
            kinematicsNames{i} = readString(dis);
        end
    end
    
    if positionCount == 0
        positionNames = {};
    else
        positionNames{positionCount} = [];
        for i=1:positionCount
            positionNames{i} = readString(dis);
        end
    end    


    % ###############################################
    % ##                                           ##
    % ##    read 1st parameter block               ##
    % ##                                           ##
    % ###############################################

    dis=fileList([FullFolderName '/parameters.bin']);
    fver = readInt(dis);
    if fver < 2 || fver > 3
        fprintf('\nInvalid parameter file format!! %d\n', fver);
        return
    end
    
    while available(dis) > 0
        type = readInt(dis);
        
        if type == 1
            GroupName = readString(dis);
            GroupDescription = readString(dis);
            c3d.ParameterGroup(end+1).name={GroupName};
            c3d.ParameterGroup(end).description={GroupDescription};
            c3d.ParameterGroup(end).Parameter = [];
        else
            ParamName = readString(dis);
            ParamDescription = readString(dis);
            dataType = readInt(dis);
            valueCount = readInt(dis);
            
            c3d.ParameterGroup(end).Parameter(end+1).name={ParamName};
            c3d.ParameterGroup(end).Parameter(end).description={ParamDescription};
            
            if dataType == 0 % String
                if valueCount == 1
                    c3d.ParameterGroup(end).Parameter(end).data = readString(dis);
                else
                    c3d.ParameterGroup(end).Parameter(end).data{valueCount} = [];
                    for i=1:valueCount
                        c3d.ParameterGroup(end).Parameter(end).data{i} = readString(dis);
                    end
                end
            elseif dataType == 1 % byte
                fprintf('byte data');
            elseif dataType == 2 % integer
                if valueCount == 1
                    c3d.ParameterGroup(end).Parameter(end).data = readInt(dis);
                else
                    c3d.ParameterGroup(end).Parameter(end).data = readInts(dis, valueCount)';
                end
            elseif dataType == 3 % float
                if valueCount == 1
                    c3d.ParameterGroup(end).Parameter(end).data = readFloats(dis, 1);
                else
                    c3d.ParameterGroup(end).Parameter(end).data = readFloats(dis, valueCount)';
                end
            end
        end
    end
    
    
    % ###############################################
    % ##                                           ##
    % ##    read positions and kinematics          ##
    % ##                                           ##
    % ###############################################
    %  Get the coordinate and analog data
    
    if positionCount == 0
        c3d.Hand.RightX = [];
        c3d.Hand.RightY = [];
        c3d.Hand.RightZ = [];            
        c3d.Hand.LeftX = [];
        c3d.Hand.LeftY = [];
        c3d.Hand.LeftZ = [];
    else
        for i=1:positionCount
            fid = fileList([FullFolderName '/' positionNames{i} '.position']);
            fver = readInt(fid);
            if fver ~= 1
                sprintf('Unknown file version for position data.\n');
                continue;
            end
            
            label = readString(fid);
            desc = readString(fid);
            posCount = readInt(fid);
            data = readFloats(fid, posCount * 2);
            data = reshape(data, [2, posCount]);
            if strfind(positionNames{i}, 'Right') == 1
                c3d.Hand.RightX = data(1,:)';
                c3d.Hand.RightY = data(2,:)';
                c3d.Hand.RightZ = zeros(posCount, 1);            
            elseif strfind(positionNames{i}, 'Left') == 1
                c3d.Hand.LeftX = data(1,:)';
                c3d.Hand.LeftY = data(2,:)';
                c3d.Hand.LeftZ = zeros(posCount, 1);
            else
                sprintf('Unknown position data: %s.\n', positionNames{i});
            end
        end
    end

    if channelCount == 0
        c3d.AnalogSignals = [];
    else
        for i=1:channelCount
            fid = fileList([FullFolderName '/' kinematicsNames{i} '.kinematics']);
            fver = readInt(fid);
            if fver ~= 1
                sprintf('Unknown file version for position data.\n');
                continue;
            end
            
            label = readString(fid);
            desc = readString(fid);
            unit = readString(fid);
            dataCount = readInt(fid);
            
            data = readFloats(fid, dataCount)';            
            fieldName = makeValidFieldName({label}, false);
            c3d.AnalogSignals.(fieldName{1}) = data;
        end
    end
                    

    % ###############################################
    % ##                                           ##
    % ##    read events                            ##
    % ##                                           ##
    % ###############################################

    dis=fileList([FullFolderName '/examevents.bin']);
    fver = readInt(dis);
    if fver < 2 || fver > 3
        fprintf('Invalid event file format!! %d\n', fver);
        return
    end
    
    eventCount = readInt(dis);
    if eventCount > 0
        c3d.NEREvents.LABELS{eventCount} = [];
        c3d.NEREvents.TIMES = zeros(1, eventCount);

        for i=1:eventCount
            c3d.NEREvents.LABELS{i} = readString(dis);
            desc = readString(dis);
            c3d.NEREvents.TIMES(i) = readFloats(dis, 1);
        end
        c3d.NEREVENTS.USED = eventCount;
    else
        c3d.NEREvents = [];
    end
    
    % ###############################################
    % ##                                           ##
    % ##    read video ACKs                        ##
    % ##                                           ##
    % ###############################################
    
    dis=fileList([FullFolderName '/videoack.bin']);
    fver = readInt(dis);
    if fver < 2 || fver > 3
        fprintf('Invalid ack file format!! %d\n', fver);
        return
    end
    
    ackCount = readInt(dis);
    if ackCount > 0
        c3d.NERRanges.LABELS{ackCount} = [];
        c3d.NERRanges.STARTTIMES = zeros(1, ackCount);
        c3d.NERRanges.STOPTIMES = zeros(1, ackCount);
        
        times = readFloats(dis, ackCount*2);
        times = reshape(times, [2 ackCount]);
        c3d.NERRanges.STARTTIMES = times(1,:);
        c3d.NERRanges.STOPTIMES = times(2,:);
        for i=1:ackCount
            c3d.NERRanges.LABELS{i} = sprintf('Video Frame %d', i);
        end
    else
        c3d.NERRanges.LABELS = {};
        c3d.NERRanges.STARTTIMES = [];
        c3d.NERRanges.STOPTIMES = [];
    end
    c3d.NERRanges.USED = ackCount; 
    fprintf('.');
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
		namesNoSpaces = regexprep(namesNoSpaces, '-', '_');
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

function [ ret ] = available( fstruct )
    ret = ftell(fstruct.fid) < flength(fstruct.fid);
end

function [ ret ] = readFloats( fstruct, count )
    if count == 0
        ret = [];
    else                
        if fstruct.requiresSwap
            % KAS-51 - The "*float32" forces the function to return singles.
            % this makes it possible for the byte reordering to
            % work properly without accidentally finding values
            % that are NaN
            ret = fread(fstruct.fid, count, '*float32');
            ret = double(typecast(swapbytes(typecast(ret, 'int32')), 'single'));
        else
            ret = fread(fstruct.fid, count, 'float32');
        end
    end
end

function i = readInt( fstruct )
    i = fread_int(fstruct.fid, 1, fstruct.requiresSwap);
end

function [ ret ] = readInts( fstruct, count )
    if count == 0
        ret = [];
    else
        ret = fread_int(fstruct.fid, count, fstruct.requiresSwap);
    end
end

function [ ret ] = readString( fstruct )
    % A string is a length and then a UTF-16 set of characters
    slen = fread_int(fstruct.fid, 1, fstruct.requiresSwap);
    if slen == 0
        ret = char([]);
    else
        bytes = fread(fstruct.fid, slen * 2, '*int8');
        ret = native2unicode(bytes, 'UTF-16LE');
    end

end

