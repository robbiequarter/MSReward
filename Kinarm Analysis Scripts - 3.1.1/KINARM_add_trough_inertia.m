function dataOut = KINARM_add_trough_inertia(dataIn, varargin)

%KINARM_ADD_TROUGH_INERTIA Add trough inertia to KINARM robot inertia.
% 	DATA_OUT = KINARM_ADD_TROUGH_INERTIA(DATA_IN...) adds two new fields
%	(.RIGHT_KINARM_TROUGHS and .LEFT_KINARM_TROUGHS) to the DATA_IN
%	structure.  These two new fields contain an estimate of the inertial
%	properties of the arm troughs  (or anything else added to the KINARM
%	robot, such as a handle).  These fields are used by	KINARM_ADD_TORQUES
%	to estimate applied and intramuscular torques.   
% 
% 	DATA_OUT = KINARM_ADD_TROUGH_INERTIA(DATA_IN..., 'trough_db', TROUGH_DB, ...)
% 	Use TROUGH_DB as a database containing the inertial parameters of the arm
% 	troughs, such as mass, CofM.  TROUGH_DB should be created by calling
% 	KINARM_CREATE_TROUGH_DATABASE, which can be modified to create custom
% 	databases (e.g. if a custom handle was used with the KINARM robot).  
% 
% 	DATA_OUT = KINARM_ADD_TROUGH_INERTIA(DATA_IN..., 'trough_size', TROUGH_SIZE, ...)
% 	TROUGH_SIZE should either the string 'estimate' or a structure
% 	whose fields are identical to the fields of TROUGH_DB and whose values
% 	are equal to one of the sub-fields for each of TROUGH_DB's fields.  For
% 	example, if TROUGH_DB had the fields: .UA, .FA and .H, each of which had
% 	subfield .SML and .LRG, then TROUGH_SIZE must have the fields .UA, .FA and
% 	.H. and values 'SML' or 'LRG', such as:
% 
% 	TROUGH_SIZE = 
% 			UA: 'SML'
% 			FA: 'LRG'
% 			 H: 'SML'
% 
% 	If TROUGH_SIZE == 'estimate', then an estimate of trough size is made
% 	based on information in TROUGH_DB and the subject's mass and height.  See
% 	KINARM_CREATE_TROUGH_DATABASE for the required information in TROUGH_DB.
% 	Subject mass and height are either extracted on a per-trial basis from
% 	DATA_IN, or are provided via optional 'mass' and 'height' inputs (see
% 	below). 
% 
% 	DATA_OUT = KINARM_ADD_TROUGH_INERTIA(DATA_IN..., 'trough_location', TROUGH_LOCATION, ...)
% 	TROUGH_LOCATION should either the string 'estimate' or a structure whose
% 	fields are identical to the fields of TROUGH_DB and whose values are
% 	equal to location of the trough-index mark relative to the proximal joint
% 	(units of meters).  For example, if TROUGH_DB had the fields: .UA, .FA and .H, then
% 	TROUGH_LOCATION must have the fields .UA, .FA and .H. as well, containing
% 	numeric values, such as:   
% 
% 	TROUGH_LOCATION = 
% 			UA: 0.01
% 			FA: 0.15
% 			 H: 0.25
% 
% 	If TROUGH_LOCATION == 'estimate', then the input ESTIMATE_L2 can be
% 	provided, as per: 
% 	DATA_OUT = KINARM_ADD_TROUGH_INERTIA(DATA_IN..., 'trough_location', 
%   TROUGH_LOCATION, 'estimate_L2', ESTIMATE_L2 ...)  If ESTIMATE_L2 == true, 
%   then the length of L2 is estimated from the	length of L1, which is 
%   extracted on a per-trial basis from DATA_IN. If ESTIMATE_L2 == false or
% 	is not provided then L2 is extracted from DATA_IN.  L1 and L2 are then 
%   used with information in TROUGH_DB to estimate trough location.  See
% 	KINARM_CREATE_TROUGH_DATABASE for the required information in TROUGH_DB.
% 
% 	Differences between the anatomical length of L2 and the calibrated length
% 	can arise, for example,  if a subject chose to use their knuckle as
% 	feedback cursor position rather than their fingertip, or if a handle was
% 	grasped rather than arm troughs. 
% 
% 	DATA_OUT = KINARM_ADD_TROUGH_INERTIA(DATA_IN..., 'mass', MASS, ...)
% 	If TROUGH_SIZE == 'estimate', then an optional 'mass' input can be
% 	provided such that MASS is used to estimate trough_size rather than the
% 	mass stored in DATA_IN.  MASS should be in kg.
% 
% 	DATA_OUT = KINARM_ADD_TROUGH_INERTIA(DATA_IN..., 'height', HEIGHT, ...)
% 	If TROUGH_SIZE == 'estimate', then the optional 'height' input can be
% 	provided such that HEIGHT is used to estimate trough_size rather than the
% 	height stored in DATA_IN.  HEIGHT should be meters.
%
%	The input structure DATA_IN	can in one of two forms, based on zip_load:
%   data = zip_load;
%   dataNew = KINARM_add_friction(data)
%     OR
%   dataNew.c3d = KINARM_add_friction(data(ii).c3d)
%
%   Copyright 2009-2021 BKIN Technologies Ltd

dataOut = dataIn;			%default

if isempty(dataIn)
	return
end

%check out varargin to ensure that they are valid
x = 1;
input.mass = [];			%default, indicating the input_mass 
input.height = [];			%default
input.estimateL2 = false;	%default

while x <= length(varargin)
	if strncmpi(varargin{x}, 'trough_db', 9)
		x = x + 1;
		if length(varargin) >= x && isstruct(varargin{x})
			troughDatabase = varargin{x};
		else
			error('---> The value of trough_db was either not provided or is not a structure.');
		end
	elseif strncmpi(varargin{x}, 'trough_size', 11)
		x = x + 1;
		if length(varargin) >= x
			troughSize = varargin{x};
		else
			error('---> The value of trough_size was not provided.');
		end
	elseif strncmpi(varargin{x}, 'trough_location', 15)
		x = x + 1;
		if length(varargin) >= x
			troughLocation = varargin{x};
		else
			error('---> The value of trough_location was not provided.');
		end
	elseif strncmpi(varargin{x}, 'mass', 4)
		x = x + 1;
		if length(varargin) >= x && isnumeric(varargin{x})
			input.mass = varargin{x};
		else
			error('---> The value of the mass was either not provided or was not numeric.');
		end
	elseif strncmpi(varargin{x}, 'height', 6)
		x = x + 1;
		if length(varargin) >= x && isnumeric(varargin{x})
			input.height = varargin{x};
		else
			error('---> The value of the height was either not provided or was not numeric.');
		end
	elseif strncmpi(varargin{x}, 'estimate_L2', 11)
		x = x + 1;
		if length(varargin) >= x && islogical(varargin{x})
			input.estimateL2 = varargin{x};
		else
			error('---> The value of estimate_L2 was either not provided or was not logical (i.e. do not use quotes around true or false).');
		end
	end
	x = x + 1;
end

if ~exist('troughDatabase', 'var')
	error('---> The required input trough_db was not provided');
end
if ~exist('troughSize', 'var')
	error('---> The required input trough_size was not provided');
end
if ~exist('troughLocation', 'var')
	error('---> The required input trough_location was not provided');
end

% Once all of the varargin are collected, ensure that trough_size and
% trough_location are valid.  
% get the trough types listed in the database.
troughTypes = fieldnames(troughDatabase);

% determine if trough_size was provided or is to be guessed
estimate.troughSize = false;		%default value
if ischar(troughSize) && strcmp(troughSize, 'estimate')
	estimate.troughSize = true;		%Trough size is to be guessed
elseif isstruct(troughSize)
	%if trough_size is provided, ensure that it is a structure containing
	%the required fields (i.e. the trough_types)
	sizeFields = fieldnames(troughSize);
	for ii = 1:length(troughTypes)
		if isempty(strmatch(troughTypes{ii}, sizeFields, 'exact'))
			error(['---> TROUGH_SIZE input structure is missing trough_type field ''.' troughTypes{ii} '''.']);
		end
	end
else 
		error('---> TROUGH_SIZE input must be the string ''estimate'' or a structure.');
end



% determine if trough_location was provided or is to be estimated
estimate.location = false;		%default value
if ischar(troughLocation) && strcmp(troughLocation, 'estimate')
	estimate.location = true;		%trough location is to be estimated
elseif isstruct(troughLocation)
	% if a structure was passed for trough_location, check to see that it
	% has the correct fields and valid values for those fields.	location_fields = fieldnames(trough_location);
	locationFields = fieldnames(troughLocation);
	for ii = 1:length(troughTypes)
		if isempty(strmatch(troughTypes{ii}, locationFields, 'exact'))
			error(['---> TROUGH_LOCATION input structure is missing trough_type field ''.' troughTypes{ii} '''.']);
		end
	end
else 
	error('---> TROUGH_LOCATION input must be the string ''estimate'' or a structure.');
end


% check troughDatabase to ensure that each trough_type has the
% necessary fields (i.e, .segment).  Then check each
% trough_type for the necessary sub-fields (.M, .I, etc).
for ii = 1:length(troughTypes)
	if ~isfield(troughDatabase.(troughTypes{ii}), 'segment')  
		error(['---> Error in input parameter TROUGH_DB.  Trough type .' troughTypes{ii} ' is missing subfield .segment.  Trough inertias cannot be added. ']);
	end
	segment = troughDatabase.(troughTypes{ii}).segment;
	if isempty(segment) || isempty(strmatch(segment, {'L1', 'L2', 'L3', 'L4'}, 'exact'))
		error(['---> Error in input parameter TROUGH_DB.  Value of trough_db.' troughTypes{ii} '.segment is not valid.  Must be ''L1'', ''L2'', ''L3'' or ''L4''.  Trough inertias cannot be added.']);
	end

	% get the names of all trough sizes, which is assumed to be all fields
	% other than .segment and .location_est
	troughSizes = fieldnames(troughDatabase.(troughTypes{ii}));
	troughSizes(strmatch('segment', troughSizes, 'exact')) = [];
	troughSizes(strmatch('location_est', troughSizes, 'exact')) = [];
	
	% make sure that all trough sizes have valid .M, .I, .C_AXIAL and
	% .C_ANTERIOR subfields
	for jj = 1:length(troughSizes)
		if sum( ~isfield(troughDatabase.(troughTypes{ii}).(troughSizes{jj}),{'M', 'I', 'C_AXIAL', 'C_ANTERIOR'}) )
			error(['---> Error in input parameter TROUGH_DB.  troughdb.' troughTypes{ii} '.' troughSizes{jj} ' is missing one of the required subfields: .M, .I, .C_AXIAL and/or .C_ANTERIOR.  Trough inertias cannot be added.']);
		end
		M = troughDatabase.(troughTypes{ii}).(troughSizes{jj}).M;
		I = troughDatabase.(troughTypes{ii}).(troughSizes{jj}).I;
		C_AXIAL = troughDatabase.(troughTypes{ii}).(troughSizes{jj}).C_AXIAL;
		C_ANTERIOR = troughDatabase.(troughTypes{ii}).(troughSizes{jj}).C_ANTERIOR;
		if isempty(M) || ~isnumeric(M) || length(M) > 1
			error(['---> Error in input parameter TROUGH_DB.  Value of trough_db.' troughTypes{ii} '.' troughSizes{jj} '.M is not valid.  Trough inertias cannot be added.']);
		end
		if isempty(I) || ~isnumeric(I) || length(I) > 1
			error(['---> Error in input parameter TROUGH_DB.  Value of trough_db.' troughTypes{ii} '.' troughSizes{jj} '.I is not valid.  Trough inertias cannot be added.']);
		end
		if isempty(C_AXIAL) || ~isnumeric(C_AXIAL) || length(C_AXIAL) > 1
			error(['---> Error in input parameter TROUGH_DB.  Value of trough_db.' troughTypes{ii} '.' troughSizes{jj} '.C_AXIAL is not valid.  Trough inertias cannot be added.']);
		end
		if isempty(C_ANTERIOR) || ~isnumeric(C_ANTERIOR) || length(C_ANTERIOR) > 1
			error(['---> Error in input parameter TROUGH_DB.  Value of trough_db.' troughTypes{ii} '.' troughSizes{jj} '.C_ANTERIOR is not valid.  Trough inertias cannot be  added.']);
		end
	end
end					%end for loop 

% MAIN PART OF CODE TO ADD TROUGH INERTIA
if isfield(dataIn, 'c3d')
	% if the data passed in are in the form of exam files (i.e. from
	% zip_load), then add torques to each exam file, one at a time.
	for jj = 1:length(dataIn)
		dataOut(jj).c3d = AddTroughInertiasToAllTrials(dataIn(jj).c3d, dataOut(jj).c3d, troughDatabase, input, estimate, troughSize, troughLocation, dataOut(jj).filename{:} );
    	dataOut(jj).c3d = ReorderFieldNames(dataIn(jj).c3d, dataOut(jj).c3d);
	end
else
	% legacy functionality, assuming that dataIn = examFile(ii).c3d
	dataOut = AddTroughInertiasToAllTrials(dataIn, dataOut, troughDatabase, input, estimate, troughSize, troughLocation, 'legacy');
	dataOut = ReorderFieldNames(dataIn, dataOut);
end	




disp('Finished adding KINARM robot arm trough inertias');

end

%%
function dataOut = AddTroughInertiasToAllTrials(dataIn, dataOut, troughDatabase, input, estimate, troughSize, troughLocation, examFilename)
	%for each trial, add the trough_inertia
	
	troughInertiaAdded = true;

	for ii = 1:length(dataIn)
		if estimate.troughSize
			troughSize = GuessTroughSize(dataIn(ii), troughDatabase, input);
		end
		
		
		%add inertias for right KINARM robot
		if isfield(dataIn(ii), 'RIGHT_KINARM');
			if estimate.location && ~isfield(dataIn(ii).CALIBRATION, 'RIGHT_L1');
				% Only KINARM Exoskeletons have the 'RIGHT_L1' field, and it is necessary for estimating trough location
				if strcmp(examFilename, 'legacy')
					disp( '   WARNING - Estimate trough location is only valid for KINARM Exoskeleton. Trough inertia not added.' );
				else
					troughInertiaAdded = false;
				end
			else
				if estimate.location
					troughLocation = EstimateTroughLocation(dataIn(ii), troughDatabase, 'RIGHT', input.estimateL2);
				end
				inertias = EstimateTroughInertias(troughDatabase, troughSize, troughLocation);
				dataOut(ii).RIGHT_KINARM_TROUGHS = inertias;
			end
		end
		
		%add inertias for left KINARM  robot
		if isfield(dataIn(ii), 'LEFT_KINARM');
			if estimate.location && ~isfield(dataIn(ii).CALIBRATION, 'LEFT_L1');
				% Only KINARM Exoskeletons have the 'LEFT_L1' field, and it is necessary for estimating trough location
				if strcmp(examFilename, 'legacy')
					disp( '   WARNING - Estimate trough location is only valid for KINARM Exoskeleton. Trough inertia not added.' );
				else
					troughInertiaAdded = false;
				end
			else
				if estimate.location
					troughLocation = EstimateTroughLocation(dataIn(ii), troughDatabase, 'LEFT', input.estimateL2);
				end
				inertias = EstimateTroughInertias(troughDatabase, troughSize, troughLocation);
				dataOut(ii).LEFT_KINARM_TROUGHS = inertias;
			end
		end
	end
	if ~strcmp(examFilename, 'legacy') && ~troughInertiaAdded
		disp( ['   WARNING - Estimate trough location is only valid for KINARM Exoskeleton. Trough inertia not added to ' examFilename '.'] );
	end
end


function dataOut = ReorderFieldNames(dataIn, dataOut)
	%re-order the fieldnames
	origNames = fieldnames(dataIn);
	rightNames = {'RIGHT_KINARM_TROUGHS'};
	leftNames = {'LEFT_KINARM_TROUGHS'};
	%Before re-arranging them, check to see if they existed in the original
	%data structure, in which case do NOT re-arrange
	if ~isempty( find(strcmp('RIGHT_KINARM', origNames), 1) ) && isempty( find( strcmp(rightNames{1}, origNames), 1) ) && isfield(dataOut, 'RIGHT_KINARM_TROUGHS')
		index = find(strcmp('RIGHT_KINARM', origNames), 1);
		newNames = cat(1, origNames(1:index), rightNames, origNames(index+1:length(origNames)));
	else
		newNames = origNames;
	end
	if ~isempty( find( strcmp('LEFT_KINARM', origNames), 1) ) && isempty( find( strcmp(leftNames{1}, origNames), 1) ) && isfield(dataOut, 'LEFT_KINARM_TROUGHS')
		index = find(strcmp('LEFT_KINARM', newNames), 1);
		newNames = cat(1, newNames(1:index), leftNames, newNames(index+1:length(newNames)));
	end
	dataOut = orderfields(dataOut, newNames);
end


function troughSizeOut = GuessTroughSize(data_trial_in, troughDatabase, input)
% Guess which trough size to use, based on subject  mass and height
	% Guess which trough Add the trough inertia 
	if isempty(input.mass)
		subjectMass = data_trial_in.EXPERIMENT.WEIGHT;			%kg
	else
		subjectMass = input.mass;							%kg
	end
	if subjectMass <= 0
		error('---> Subject mass was <=0, therefore cannot guess which arm trough was used from body index.  Trough inertia not estimated.');
	end
	if isempty(input.height) 
		subjectHeight = data_trial_in.EXPERIMENT.HEIGHT;			%m
        if subjectHeight > 100
            subjectHeight = subjectHeight / 100; %convert to m
        end
	else
		subjectHeight = input.height;							%kg
	end
	if subjectHeight <= 0
		error('---> Subject_height was <=0, therefore cannot guess which arm trough was used from body index.  Trough inertia not estimated.');
	end
	bodyIndex = subjectMass / subjectHeight;
	troughTypes = fieldnames(troughDatabase);
	% for each trough_type, guess the size and store it in the output structure
	for ii = 1:length(troughTypes)
		% get the names of all trough sizes, which is assumed to be all fields
		% other than .segment and .location_est
		troughSizes = fieldnames(troughDatabase.(troughTypes{ii}));
		troughSizes(strmatch('segment', troughSizes, 'exact')) = [];
		troughSizes(strmatch('location_est', troughSizes, 'exact')) = [];

		%the following guesses which size to use based on the 'body_index'
		minBodyIndices = zeros(size(troughSizes));
		for jj = 1:length(troughSizes)
			minBodyIndices(jj) = troughDatabase.(troughTypes{ii}).(troughSizes{jj}).body_index_min;
		end
		possibleSizes = minBodyIndices( bodyIndex >= minBodyIndices );
		if ~isempty(possibleSizes)
			troughSizeOut.(troughTypes{ii}) = troughSizes{ max(possibleSizes) == minBodyIndices  };
		else
			error(['No trough sizes are specified for the body_index from subject mass ' num2str(subjectMass) 'kg and subject height ' num2str(subjectHeight) 'm.'])
		end		%end testing for which size
	end
	
end


function troughLocationOut = EstimateTroughLocation(dataTrialIn, troughDatabase, side, estimateL2)
% Estimate trough location based on L2 and L1.
	% estimate trough locations
	L1 = dataTrialIn.CALIBRATION.([side '_L1']);
	if estimateL2
		L2 = 1.37 * L1;				%Estimate forearm+hand length (L2) from upper arm length (L1).  From Winters, page 48
	else
		L2 = dataTrialIn.CALIBRATION.([side '_L2']);
	end
	% troughDatabase must have a .location_est that contains a 1x4 vector for
	% estimating trough location based on L1 and L2 lengths
	troughTypes = fieldnames(troughDatabase);
	for ii = 1:length(troughTypes)
		if ~isfield(troughDatabase.(troughTypes{ii}), 'location_est')  
			error(['---> Error in input parameter TROUGH_DB.  Trough type .' troughTypes{ii} ' is missing subfields .segment and/or .location_est.  Trough inertias cannot be added. ']);
		end
		locationEst = troughDatabase.(troughTypes{ii}).location_est;
		if isempty(locationEst) || ~isnumeric(locationEst) || length(locationEst)~= 4
			error(['---> Error in input parameter TROUGH_DB.  Value of trough_db.' troughTypes{ii} '.location_est is not valid.  Must be a 1x4 numeric vector.  Trough inertias cannot be added.']);
		else
			troughLocationOut.(troughTypes{ii}) = dot(locationEst, [L1 1 L2 1]);
		end
	end

end


function totalInertia = EstimateTroughInertias(troughDatabase, troughSize, troughLocation)
	% Estimate the trough inertia (total for all trough types)
	troughTypes = fieldnames(troughDatabase);
	totalInertia.L1_M = 0;				%for KINARM_combine_inertias to function correctly, total_inertia cannot be empty
	for ii = 1:length(troughTypes)
		clear inertia;					%clear inertia so that it starts out empty
		inertia.L1_M = 0;				%for KINARM_combine_inertias to function correctly, inertia cannot be empty
		segment = troughDatabase.(troughTypes{ii}).segment;
		size = troughSize.(troughTypes{ii});  
		if ~isfield(troughDatabase.(troughTypes{ii}), size)
			error(['---> Error.  The specified trough size ''' size ''' was not found in the trough database for ''trough_db.' troughTypes{ii} '''.']);
		end
		inertia.([segment '_M']) = troughDatabase.(troughTypes{ii}).(size).M;
		inertia.([segment '_I']) = troughDatabase.(troughTypes{ii}).(size).I;
		inertia.([segment '_C_AXIAL']) = troughDatabase.(troughTypes{ii}).(size).C_AXIAL + troughLocation.(troughTypes{ii});
		inertia.([segment '_C_ANTERIOR']) = troughDatabase.(troughTypes{ii}).(size).C_ANTERIOR;
		totalInertia = KINARM_combine_inertias(totalInertia, inertia);
	end					%end for loop 
	% Add fields for each trough_type, and put the size as its value
	for ii = 1:length(troughTypes)
		totalInertia.(troughTypes{ii}) = troughSize.(troughTypes{ii});
	end					%end for loop 

end
