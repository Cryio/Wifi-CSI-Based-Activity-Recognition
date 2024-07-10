clear
clc
close all
flist = dir('*.csv');
B = [];
for k = 1:length(flist)
    fid = fopen(flist(k).name);
    A = textscan(fid,'%d, %s\n');
    B = [B;A{2}];
    fclose(fid);
end
B = unique(B);
count = zeros(length(B),1);
for k = 1:length(flist)
    fid = fopen(flist(k).name);
    A = textscan(fid,'%d, %s\n');
    fclose(fid);
    for l = 1:length(A{2})
        count = count + double(ismember(B,A{2}{l}));
    end
    k
end