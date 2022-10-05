% The purpose of demo_exam_load.m is to demonstrate the basic functionality
% of the library of m-files associated with exam_load.
% 
% Please make sure that the path to the c3d_load m-files is in the Matlab
% path.  Then change to the directory containing the c3d_files of interest
% and type demo_c3d_load at the command prompt

%   Copyright 2005-2021 BKIN Technologies Ltd

data = exam_load('*.zip');									% Loads the named file into a new structure called 'data'.
data = KINARM_add_hand_kinematics(data);						% Add hand velocity, acceleration and commanded forces to the data structure
data_filt = filter_double_pass(data, 'enhanced', 'fc', 10);		% Double-pass filter the data at 3db cutoff frequency of 10 Hz.  Use an enhanced method for reducing transients at ends.

% Message
disp(' ')
disp('**************************************************************')
disp('Demonstration of exam_load functionality.')
disp(' ')
% Display contents of final data structure
disp('Contents of first trial of first file:');
disp(data_filt(1).c3d(1));

% Plot example hand paths
figure
for ii = 1:length(data.c3d)
	hold on
	plot(data_filt(1).c3d(ii).Right_HandX,  data_filt(1).c3d(ii).Right_HandY);
end
ylabel('Y (m)');
xlabel('X (m)');
title ('Example - Hand Paths (all trials)');



