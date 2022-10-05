function demo_calc_visual_stimulus_time()
% The purpose of this function is to provide sample code that calculates the
% time at which a visual stimulus would have been presented to a subject, taking
% into account all of the various delays, including the location of the visual
% stimulus on the subject display screen 

%   Copyright 2016-2021 BKIN Technologies Ltd

% Load data into MATLAB
filename = 'sampleDataForLatencyCalc.zip';
singleExamData = zip_load(filename);

% Calculate the period of a single frame of video (seconds)
tVideoFramePeriod = 1 / singleExamData.c3d(1).VIDEO_SETTINGS.REFRESH_RATE;

% Specify the subject display’s properties  
tDisplayDelays = 0.010;			% e.g. display delay (including response time and other internal delays)
numBufferedFrames = 1;			% e.g. the number of frames that are buffered by the subject display

% Alternatively, one can estimate tDisplayDelays from the feed_forward value
% that is stored/used in Dexterit-E. The feed-forward delay is optimized for a
% location ~33% from the top of the subject display (i.e. closest to the
% subject). Additionally, there is typically about ~1 frame of delay due to
% DexDisplay.exe for a total of 1.33 frames. 
% As of 2016, all displays shipped with KINARM Labs have a single frame buffer.
tFeedForward = singleExamData.c3d(1).VIDEO_SETTINGS.FEED_FORWARD;
numBufferedFrames = 1;			% assume that the display buffers a single frame
tDisplayDelays = tFeedForward - (1.33 + numBufferedFrames) * tVideoFramePeriod;  

numTrials = length(singleExamData.c3d);
for trial = 1:numTrials
	% Retrieve the data from a single trial
	trialData = singleExamData.c3d(trial);

	% Estimate the display time of all video frames (i.e. upper-left pixel):
	trialData = add_video_latency(trialData, tDisplayDelays, numBufferedFrames);

	% Identify the time at which the visual stimulus of interest was commanded
	% NOTE: This section of code will be unique for each Task Program
	eventIndex = find(strncmp('TARGET_ON', trialData.EVENTS.LABELS, 9), 1);
	tVisStimRequest = trialData.EVENTS.TIMES(eventIndex);

	% Identify the video frame that displayed the visual stimulus of interest
	videoFrameNum = find(trialData.VIDEO_LATENCY.SEND_TIMES >= tVisStimRequest, 1); 

	% Retrieve the time at which the frame showing the visual stimulus of
	% interest was displayed to the subject. 
	% Note: DISPLAY_MAX_TIMES under normal operating conditions contains the
	% time at which the top-left pixel would be have been displayed for each
	% frame. If there are irregularities in these values, then it is possible
	% that the frame was displayed earlier.
	tVisStimFrameDisplayed  = trialData.VIDEO_LATENCY.DISPLAY_MAX_TIMES(videoFrameNum);

	% Correct for the impact of location of the visual stimulus on the screen location 
	% NOTE: This section of code will be unique for each Task Program
	yBottomOfDisplay = trialData.VIDEO_SETTINGS.DISPLAY_SIZE_M(2); % units of m
	tpRow = trialData.TRIAL.TP;										
	targetRow = trialData.TP_TABLE.Start_Target(tpRow);
	yVisStim = trialData.TARGET_TABLE.Y_GLOBAL(targetRow) / 100; % target table has units of cm
	tLatencyLocation = tVideoFramePeriod * (yVisStim / yBottomOfDisplay);
	tVisStimDisplayed = tVisStimFrameDisplayed + tLatencyLocation; % time that visual stimulus was presented to subject

	% If desired, calculate and display the contributions to the overall latency
	% at which the visual stimulus was  displayed vs requested.
	tVisStimFrameSend  = trialData.VIDEO_LATENCY.SEND_TIMES(videoFrameNum);
	latencyWaitingForVsync = (tVisStimFrameSend - tVisStimRequest) * 1000;	% ms
	latencyDisplayBuffer = numBufferedFrames * tVideoFramePeriod * 1000;	% ms
	latencyLocation = tLatencyLocation * 1000;	% ms
	latencyResponseTime = tDisplayDelays * 1000;	% ms
	latencyTotal = (tVisStimDisplayed - tVisStimRequest) * 1000;		% ms
	latencyDexDisplay = latencyTotal - (latencyWaitingForVsync + latencyDisplayBuffer + latencyLocation + latencyResponseTime); % ms
	if trial == 1
		fprintf('\r');
		display( 'Latency contributions from:' );
		display( '    waitVsync + dexDisp + dispBuffer + location + displayDelay = total' );
	end
	fprintf('Trial %d: %0.1f + %0.1f + %0.1f + %0.1f + %0.1f = %0.1f ms\n', trial, ...
		latencyWaitingForVsync, latencyDexDisplay, latencyDisplayBuffer, latencyLocation, latencyResponseTime, latencyTotal);
end
