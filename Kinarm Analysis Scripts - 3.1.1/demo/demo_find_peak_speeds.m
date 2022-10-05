function mean_peak_speeds=demo_find_peak_speeds( zip_name )
    %%%
    % This method is intended to be a simple example of analyzing an exam
    % file - in this case an 8 target Visually Guided Reaching exam. This
    % code will load the exam, prepare the data, find the peak speed during
    % each reach, then find the mean peak speed for each target. The code
    % then plots the mean peak speeds.
    %%%
    
    data = zip_load(zip_name);
    data = KINARM_add_hand_kinematics(data);				% Add hand velocity, acceleration and commanded forces to the data structure
    data = filter_double_pass(data, 'enhanced', 'fc', 10);	% Double-pass filter the data at 3db cutoff frequency of 10 Hz.  Use an enhanced method for reducing transients at ends.
    data = sort_trials(data, 'tp');                         % sort all of the trials into Trial Protocol order. i.e. all TP 1, then TP 2...    
    trials = data.c3d;    
    
    % based on the arm that was working during the exam determine the
    % fields we must pull out for our analysis.
    if strcmp(trials(1).EXPERIMENT.ACTIVE_ARM, 'RIGHT')
        x_vel_field = 'Right_HandXVel';
        y_vel_field = 'Right_HandYVel';
    else
        x_vel_field = 'Left_HandXVel';
        y_vel_field = 'Left_HandYVel';
    end
    
    % since the trials are sorted, the last trial has the highest Trial Protocol (TP) number
    max_tp_row = trials(end).TRIAL.TP_ROW;
    mean_peak_speeds = zeros(max_tp_row, 1);
    target_locations = zeros(max_tp_row, 2);
    
    % look at each TP that was run
	% Note: this code assumes that all TPs between 1 and max_tp_row were used.
    for cur_TP = 1:max_tp_row
        % pull out all trials for the current trial protocol
        trials_for_tp = get_trials(trials, cur_TP);
        max_speeds = zeros(length(trials_for_tp), 1);
        
        % look at each trial that used the current trial protocol
        for m = 1:length(trials_for_tp)
            trial = trials_for_tp(m);
            
            % pull out the parts of the structure we need to calculate hand
            % speed
            x_vel = trial.(x_vel_field);
            y_vel = trial.(y_vel_field);
            hand_vel = sqrt(x_vel.^2 + y_vel.^2);
            
            if cur_TP == 1
                % We know that for Visually Guided Reaching, TP 1 is a catch trial without a second target, so we need to
                % treat this TP differently. We can just look at the 
                % peak speed from the time they LAST entered the target, which was recorded online using the STAY_CENTRE
                % event, to the end of the trial.
                stay_centre_frames = get_event_frame(trial, 'STAY_CENTRE');
				last_stay_centre_frame = stay_centre_frames(end);
                max_speeds(m) = max(hand_vel(last_stay_centre_frame:end));
            else
                % All other TP's are reaches. We only want to look at hand
                % speed during the reach. This means from the time the
                % target goes on to the time we reach the destination.
                target_on_frame = get_event_frame(trial, 'TARGET_ON');
                movement_end_frame = get_event_frame(trial, 'HOLD_AT_TARGET');
                max_speeds(m) = max(hand_vel(target_on_frame:movement_end_frame));
			end
        end
        
		% calculate the mean across all trials for this TP, and also record the destination target location for this TP
        mean_peak_speeds(cur_TP) = mean(max_speeds);        
        target_locations(cur_TP,:) = get_destination_target_location(trials_for_tp(1));
    end
    
    build_graph(mean_peak_speeds, target_locations);
end

function trials_by_tp = get_trials(trials, tp)
    % Given the entire list of trials this will pull out all for the
    % specified trial protocol.
    % This is a universal function, it will work on any exam type. 
	% NOTE: for data collected before Dexterit-E 3.8 you will need to find TP not TP_ROW.
    trial_info = [trials.TRIAL];
    trials_by_tp = trials([trial_info.TP_ROW] == tp);        
end

function evt_frames = get_event_frame(trial, name)
    % Given an event name to look for, this will find all times that the
    % event occured and turn those into kinematic frame indexes.
    % This is a universal function, it will work on any exam type.
	event_list_idx = strncmpi(trial.EVENTS.LABELS, name, length(name));
    evt_times = trial.EVENTS.TIMES(event_list_idx);
	% evt_frames is cast to an int32, to enable its use as an index to vectors/matrices
    evt_frames = int32(evt_times / (1.0 / trial.ANALOG.RATE));
end

function location = get_destination_target_location(trial)
    % Given a trial, this will find the location of the destination target
    % in global coordinates, in cm (because the Target Table X_GLOBAL and Y_GLOBAL are in (cm).
    % This is NOT a univeral function - you would need to modify for the
    % fields to match those that you have named in the TP table (e.g. "End_Target").
    target_idx = trial.TP_TABLE.End_Target(trial.TRIAL.TP_ROW);
    location = [trial.TARGET_TABLE.X_GLOBAL(target_idx) trial.TARGET_TABLE.Y_GLOBAL(target_idx)];
end

function build_graph(mean_peak_speeds, target_locations)
    figure(1);
    clf;
    
    % I know that target_location from TP 1 (which is the catch trial), is also the center target for all other trials. This
    % shifts all target locations to center on (0,0)
    target_locations(:,1) = target_locations(:,1) - target_locations(1,1);		% shift x coordinates
    target_locations(:,2) = target_locations(:,2) - target_locations(1,2);		% shift y coordinates
    
    for n = 1:length(mean_peak_speeds)
		% we will draw 1 cm radius targets
		radius = 1;
		
        % define the outer bounds of a rectangle, that would contain the circle we want to draw
		center = target_locations(n,:);
		pos = [center - radius, radius * [2 2] ];
        
        % draws a circle, using the built-in "rectangle" function
        rectangle('Position',pos,'Curvature',[1 1], 'FaceColor', [1, 0, 0])
                
        msg = sprintf('%.1f cm/s', mean_peak_speeds(n) * 100);
        text_loc_x = target_locations(n,1) - 2; % shift the text a bit to center under each target
        text_loc_y = target_locations(n,2) - 1.75;
        text(text_loc_x, text_loc_y, msg, 'FontSize', 10);
    end
    
    % find the maximal X and Y to help set up the axes (+4 to give a
    % boundry buffer)
    max_pos = max(max(abs(target_locations))) + 4;
    xlim([-max_pos max_pos]);
    ylim([-max_pos max_pos]);
    pbaspect([1 1 1])
    
    title('Mean Peak Reaching Hand Speed by Target', 'FontSize', 14);
    xlabel('Target position (cm)', 'FontSize', 12);
    ylabel('Target position (cm)', 'FontSize', 12);
end
