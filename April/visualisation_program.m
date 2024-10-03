% Created by Jeroen Klein Brinke
% Pervasive Systems @ University of Twente
% 2018-2019

% This code works out of the box. However, slight changes can be made to
% load different activities.

clf('reset');
%% Load the file
% For this purpose, the activities are listed in a folder called
% "activities". If files in the same folder, put a dot (.) for folder
folder = 'mat'; % Can only be used for the 'mat'-folder
day = '1' % day in {1, 2, 3, 6, 7, 8}
participant = '1' % participant in {1,2,3} for day={1,2,3}; participant in {1,2} for day={6,7,8}}
activity = 'waving'; % activity in {walking, clapping, jumping, falling, nothing, waving}
trial = '1' % For trial in {1..50}
csitrace = getfield(load(strcat(folder,'/day_',day,'/',participant, '_', activity, '_', trial ,'.mat')),'csi_trace');

%% Extract all useful information from the csitrace
% Extract information regarding size, rate, CSI, RSSI, etc.
% Note that for CSI, usually just applying the abs is not enough [3].
% However, for demonstration purposes, it is.
frames = [csitrace{:}];
temp_frames = [];
for c_f = frames
    if c_f.Ntx==3
        temp_frames = [temp_frames c_f]
    end
end
frames = temp_frames
frame_len = size(frames,2);
Nrx = [frames.Nrx];
Ntx = [frames.Ntx];
csi  = abs(reshape([frames.csi],[max(Ntx), max(Nrx), frame_len, size([frames.csi],3)]));
rssi_a = [frames.rssi_a];
rssi_b = [frames.rssi_b];
rssi_c = [frames.rssi_c];
rate = [frames.rate];
noise = [frames.noise];
agc = [frames.agc];
perm = reshape([frames.perm], [3, frame_len]).';

%% Visualise all the information
% For the visualisation of CSI, two options are possible: Plot everything
% based on the receiving antennas - or plot individual subcarriers. For
% demonstration, both are shown here. To smooth things out, an easy smooth
% function is used.
sp_col = 3;
sp_row = 3;
m_window = 1;
for tx=1:max(Ntx)
    for rx=1:max(Nrx)
        for sc=1:30
            ax(1) = subplot(sp_row, sp_col,1:3);
            hold on;
            plot(movmean(squeeze(csi(tx,rx,1:frame_len,sc)),m_window));
            ax(2) = subplot(sp_row, sp_col,4:6);
            hold on;
            plot(movmean(squeeze(csi(tx,rx,1:frame_len,sc)),m_window), get_plot_color(rx));
        end
    end
end
ax(3) = subplot(sp_row, sp_col,7);
hold on;
plot(movmean(rssi_a,m_window));
plot(movmean(rssi_b,m_window));
plot(movmean(rssi_c,m_window));
ax(4) = subplot(sp_row, sp_col,8);
hold on;
plot(noise);

linkaxes(ax(:), 'x')
xlim([0 frame_len])

text(120,-65,strcat('Activity: ',activity));
text(120,-75,strcat('Frames: ', num2str(frame_len)));
text(120,-85,strcat('Receivers: ', num2str(min(Ntx))));
text(120,-95,strcat('Transmitters: ', num2str(min(Nrx))));
text(120,-105,strcat('Rate: ', num2str(min(rate)), '-', num2str(max(rate))));


%% Define functions
function c = get_plot_color(r)
    switch r
        case 1
            c = 'b';
        case 2
            c = 'r';
        case 3
            c = 'g';
        otherwise
            c = 'k';
    end
end