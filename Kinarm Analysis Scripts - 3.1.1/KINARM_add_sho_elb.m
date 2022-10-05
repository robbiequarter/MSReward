function [ dataOut ] = KINARM_add_sho_elb( dataIn )
%KINARM_ADD_SHO_ELB Kinarm exam data stores joint locations in global
%   coordinates, this method converts those to a local frame of reference. The
%   local frame of refernece is an exterior shoulder and exterior elbow angle.
%
%                        Elbow angles
%                         v          v           (* - robot linkage)
%                                                (\/-  - angle baseline)
%                       \   *     *    /         ( ==O== Subject)
%                        \ *       *  /
%                         *         *
% Shoulder angle ->        *       *     <- Shoulder angle
%                       ----*==O==*----   
%
%   Right shoulder and elbow angles are measured counter clock-wise. Left
%   sholder and elbow angles are measured clock-wise. Identical right
%   and left angles indicates a mirrored position.
%
%   This method only works on Kinarm Exoskeleton and Exoskeleton Classic
%   data.
%
%	The input structure dataIn	can in one of two forms, based on zip_load:
%   data = zip_load;
%   dataNew = KINARM_add_sho_elb(data)
%     OR
%   dataNew.c3d = KINARM_add_sho_elb(data(ii).c3d)
%
% The new fields have units of: rad, rad/s, rad/s^2. The fields are:
%   .Right_ShoAng
%   .Right_ShoVel
%   .Right_ShoAcc
%   .Right_ElbAng
%   .Right_ElbVel
%   .Right_ElbAcc
%   .Left_ShoAng
%   .Left_ShoVel
%   .Left_ShoAcc
%   .Left_ElbAng
%   .Left_ElbVel
%   .Left_ElbAcc

    dataOut = dataIn;

    if isempty(dataIn)
        return
    end
    
    if isfield(dataIn, 'c3d')
        % if the data passed in are in the form of exam files (i.e. from
        % zip_load), then add torques to each exam file, one at a time.
        for jj = 1:length(dataIn)            
            if ~isempty(strfind(dataIn(jj).c3d(1).RIGHT_KINARM.VERSION, 'EP'))
                fprintf('End-Point exam cannot calculate shoulder and elbow');
            else            
                fprintf('Adding shoulder and elbow values\n');
                dataOut(jj).c3d = add_sho_elb(dataIn(jj).c3d, dataOut(jj).c3d );
                disp( ['Finished adding shoulder and elbow values to ' dataOut(jj).filename{:}] );
                dataOut(jj).c3d = ReorderFieldNames(dataIn(jj).c3d, dataOut(jj).c3d);
            end
        end
    else
        if ~isempty(strfind(dataIn.RIGHT_KINARM.VERSION, 'EP'))
            fprintf('End-Point exam cannot calculate shoulder and elbow');
        else                    
            % legacy functionality, assuming that data_in = examFile(ii).c3d
            dataOut = add_sho_elb(dataIn, dataOut, 'legacy');
            dataOut = ReorderFieldNames(dataIn, dataOut);
            disp('Finished adding shoulder and elbow values');
        end
    end	
end

function dataOut = add_sho_elb(dataIn, dataOut)

    types = {'Ang', 'Vel', 'Acc'};

	for ii = 1:length(dataIn)
        
        for jj = 1:length(types)
            type = types{jj};
            if isfield(dataIn(ii), ['Right_L1' type]) && isfield(dataIn(ii), ['Right_L2' type])
                L1 = dataIn(ii).(['Right_L1' type]);
                L2 = dataIn(ii).(['Right_L2' type]);        
                dataOut(ii).(['Right_Sho' type]) = L1;
                dataOut(ii).(['Right_Elb' type]) = L2 - L1;
            end
        end
        

        if isfield(dataIn(ii), 'Left_L1Ang') && isfield(dataIn(ii), 'Left_L2Ang')
            L1 = dataIn(ii).Left_L1Ang;
            L2 = dataIn(ii).Left_L2Ang;        
            dataOut(ii).Left_ShoAng = pi - L1;
            dataOut(ii).Left_ElbAng = L1 - L2;
        end

        for jj = 2:length(types)
            type = types{jj};
            if isfield(dataIn(ii), ['Left_L1' type]) && isfield(dataIn(ii), ['Left_L2' type])
                L1 = dataIn(ii).(['Left_L1' type]);
                L2 = dataIn(ii).(['Left_L2' type]);        
                dataOut(ii).(['Left_Sho' type]) = -L1;
                dataOut(ii).(['Left_Elb' type]) = L1 - L2;
            end
        end
    end
end

function dataOut = ReorderFieldNames(dataIn, dataOut)
	%re-order the fieldnames so that the hand velocity, acceleration and
	%commanded forces are with the hand position at the beginning of the field
	%list 
	origNames = fieldnames(dataIn);
	tempNames = fieldnames(dataOut);
	rightNames = {'Right_ShoAng'; 'Right_ShoVel'; 'Right_ShoAcc'; 'Right_ElbAng'; 'Right_ElbVel'; 'Right_ElbAcc'};
	leftNames = {'Left_ShoAng'; 'Left_ShoVel'; 'Left_ShoAcc'; 'Left_ElbAng'; 'Left_ElbVel'; 'Left_ElbAcc'};

	%check to see if any right-handed or left-handed fields were added to the
	%output data structure
	addedRightToOutput = false;
	addedLeftToOutput = false;
	for ii = 1:length(rightNames)
		if isempty( strmatch(rightNames{ii}, origNames, 'exact') ) && ~isempty( strmatch(rightNames{ii}, tempNames, 'exact') )
			addedRightToOutput = true;
		end
		if isempty( strmatch(leftNames{ii}, origNames, 'exact') ) && ~isempty( strmatch(leftNames{ii}, tempNames, 'exact') )
			addedLeftToOutput = true;
		end
	end

	if addedRightToOutput
		% remove all of the new fields from the original list
		for ii = 1:length(rightNames)
			index = strmatch(rightNames{ii}, origNames, 'exact');
			if ~isempty(index)
				origNames(index) = [];
			end
		end
		% place the new fields right after the HandY field
		index = strmatch('Right_HandY', origNames, 'exact');
		newNames = cat(1, origNames(1:index), rightNames, origNames(index+1:length(origNames)));
	else
		newNames = origNames;
	end

	if addedLeftToOutput
		% remove all of the new fields from the original list
		for ii = 1:length(leftNames)
			index = strmatch(leftNames{ii}, origNames, 'exact');
			if ~isempty(index)
				origNames(index) = [];
			end
		end
		% place the new fields right after the HandY field
		index = strmatch('Left_HandY', newNames, 'exact');
		newNames = cat(1, newNames(1:index), leftNames, newNames(index+1:length(newNames)));
	end
	dataOut = orderfields(dataOut, newNames);
end
