% EXAM_LOAD Load and format exam files created with Dexterit-E 3.0 and higher.
%
%   DATA = EXAM_LOAD opens all data files (.ZIP & .KINARM files) created by Dexterit-E
%   3.0 and higher that are in the currect directory and outputs the data
%   into the structure DATA.  Each element of DATA corresponds to a single
%   data file.    
%
%   DATA contains at least two fields:
%		.c3d - this field contains all of the trial data stored in the 
%       exam file. The trials are sorted into the order in which they were 
%       collected.
%		.filename - the filename of the exam file
%
%   The format of the .c3d field is a structured array, each element of
%   which corresponds to a single trial.  The sub-fields of the .c3d
%   structure are of two types: time-varying data or a c3d Parameter Group.
%   Time-varying data are vectors of data corresponding to the field name.
%   Details of the time varying data can be found in the related Parameter
%   Group.  The Parameter Group fields have sub-fields of their own, most
%   of which are specific to each Parameter Group.
% 
%   DATA = EXAM_LOAD(EXAM_FILENAME) only opens EXAM_FILENAME.
%   EXAM_FILENAME can contain the '*' wildcard.
%   
%   DATA = EXAM_LOAD(EXAM_FILENAME1, EXAM_FILENAME2) opens EXAM_FILENAME1
%   and EXAM_FILENAME2 and outputs the data into the DATA structure.
%   EXAM_FILENAME1 and EXAM_FILENAME2 can both contain the % '*' wildcard.
%   Any number of filenames can be listed.  
%   
%   DATA = EXAM_LOAD('dir', DIRECTORY) looks for all exam files in
%   DIRECTORY.
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

function c3dstruct = exam_load(varargin)

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
			zipfiles = [zipfiles strcat(pwd, '/', {temp.name})];
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
	zipfiles = [dir('*.zip'); dir('*.kinarm')];
	if isempty(zipfiles)
		disp(strvcat(' ','WARNING!!!  No exam files found in:', pwd));
		c3dstruct = [];
		return;
	end
    zipfiles = strcat(pwd, '/', {zipfiles.name});
end

cd(startFolder);
c3dstruct = [];
ME = [];

for x = 1:length(zipfiles)
    
    if exist(zipfiles{x}, 'file') ~= 2 % Must be found as a file
        if exist([zipfiles{x} '.zip'], 'file')
            zipfiles{x} = [zipfiles{x} '.zip'];
        elseif exist([zipfiles{x} 'zip'], 'file')
            zipfiles{x} = [zipfiles{x} 'zip'];
        elseif exist([zipfiles{x} '.kinarm'], 'file')
            zipfiles{x} = [zipfiles{x} '.kinarm'];
        elseif exist([zipfiles{x} 'kinarm'], 'file')
            zipfiles{x} = [zipfiles{x} 'kinarm'];
        end
    end
    
    if strfind(zipfiles{x}, '.zip')
        data = zip_load_impl(zipfiles{x});
    else
        data = kinarm_load_impl(zipfiles{x});
    end
    
    c3dstruct = catStructs(c3dstruct, data);
end

    c3dstruct = mergeSplitTrials(c3dstruct);
    display(sprintf( 'Finished loading all exam files.') );
    display(sprintf('Size of Structure: [%d, %d]',size(c3dstruct)));
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

function c3dstruct=mergeSplitTrials(c3dstruct)
    % This function will concatenate trials that were split as part of the
    % recording process. Trials longer than 15 minutes are automatically
    % split in order to reduce the chance of data loss.
    for i=1:length(c3dstruct)
        % if this parameter is not present then the data is older than the
        % feature of splitting long trials
        if isempty(c3dstruct(i).c3d) || ~isfield(c3dstruct(i).c3d(1).TRIAL, 'BREAK_ID')
            continue;
        end
        
        j = 1;
        while j < length(c3dstruct(i).c3d)
            % a break id of 0 means that the trial was never split
            if c3dstruct(i).c3d(j).TRIAL.BREAK_ID == 0
                j = j+1;
                continue;
            end
            
            % if the next trial has a larger break ID then we should merge.
            if c3dstruct(i).c3d(j).TRIAL.BREAK_ID + 1 <= c3dstruct(i).c3d(j+1).TRIAL.BREAK_ID
                names = fieldnames(c3dstruct(i).c3d(j));
                
                % concatentate all kinematics
                for k=1:length(names)
                    if ischar(c3dstruct(i).c3d(j).(names{k})) || isstruct(c3dstruct(i).c3d(j).(names{k}))
                        continue;
                    end                    
                    frame_count = length(c3dstruct(i).c3d(j).(names{k}));
                    c3dstruct(i).c3d(j).(names{k}) = [c3dstruct(i).c3d(j).(names{k}); c3dstruct(i).c3d(j+1).(names{k})];
                end
                
                rate = c3dstruct(i).c3d(j).ANALOG.RATE;
                trial_duration = frame_count /rate;
                
                % concatenate exam events
                hasEvents1 = isfield(c3dstruct(i).c3d(j), 'EVENTS');
                hasEvents2 = isfield(c3dstruct(i).c3d(j+1), 'EVENTS');
                if hasEvents1 && hasEvents2 && ~isempty(c3dstruct(i).c3d(j).EVENTS) && ~isempty(c3dstruct(i).c3d(j+1).EVENTS)
                    c3dstruct(i).c3d(j).EVENTS.LABELS = [c3dstruct(i).c3d(j).EVENTS.LABELS, c3dstruct(i).c3d(j+1).EVENTS.LABELS];
                    c3dstruct(i).c3d(j).EVENTS.TIMES = [c3dstruct(i).c3d(j).EVENTS.TIMES, c3dstruct(i).c3d(j+1).EVENTS.TIMES + trial_duration];
                elseif (~hasEvents1 || isempty(c3dstruct(i).c3d(j).EVENTS)) && (hasEvents2 && ~isempty(c3dstruct(i).c3d(j+1).EVENTS))
                    c3dstruct(i).c3d(j).EVENTS = c3dstruct(i).c3d(j+1).EVENTS;
                end
                
                % concatenate Video information
                if ~isempty(c3dstruct(i).c3d(j).VIDEO_LATENCY) && ~isempty(c3dstruct(i).c3d(j+1).VIDEO_LATENCY)
                    c3dstruct(i).c3d(j).VIDEO_LATENCY.LABELS = [c3dstruct(i).c3d(j).VIDEO_LATENCY.LABELS, c3dstruct(i).c3d(j+1).VIDEO_LATENCY.LABELS];               
                    c3dstruct(i).c3d(j).VIDEO_LATENCY.SEND_TIMES = [c3dstruct(i).c3d(j).VIDEO_LATENCY.SEND_TIMES, c3dstruct(i).c3d(j+1).VIDEO_LATENCY.SEND_TIMES + trial_duration];
                    c3dstruct(i).c3d(j).VIDEO_LATENCY.ACK_TIMES = [c3dstruct(i).c3d(j).VIDEO_LATENCY.ACK_TIMES, c3dstruct(i).c3d(j+1).VIDEO_LATENCY.ACK_TIMES + trial_duration];
                elseif isempty(c3dstruct(i).c3d(j).VIDEO_LATENCY) && ~isempty(c3dstruct(i).c3d(j+1).VIDEO_LATENCY)
                    c3dstruct(i).c3d(j).VIDEO_LATENCY = c3dstruct(i).c3d(j+1).VIDEO_LATENCY;               
                end
                
                c3dstruct(i).c3d(j + 1) = [];
            else
                j = j+1;
            end            
        end
    end
end