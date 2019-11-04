clear;
clc;
data = load('euro_lang.mat');
body_data = data.body_parts;
body_features = body_data.vectorOut;
body_labels = body_data.labels;
mapdim = 2;
units = 4;
b1 = 1;
b2 = 0.2;
itn = 80;
[mapout, qtest, topmap] = som(body_features, body_features, mapdim, units, b1, b2, itn);
[m,n]=meshgrid(0:100);

% draw figures to show the result
[X,Y]=meshgrid(1:20:100,1:20:100);
figure; hold on;
plot(X,Y,'k');
plot(Y,X,'k');axis off

% get grid position from topmap
count_list = zeros(4,4);
for num=1:17
    cur_x = topmap(num,1);
    cur_y = topmap(num,2);
    cur_language = body_labels(num);
    cur_val = count_list(cur_x, cur_y);
    start_pos_x = (cur_x-1)*20+2;
    start_pos_y = (cur_y-1)*20+4+cur_val*4;
    %text(start_pos_x, start_pos_y, cur_language,'FontWeight','bold');
    count_list(cur_x, cur_y) = count_list(cur_x, cur_y) + 1;
end

% get grid position from topmap
count_list = zeros(4,4);
for num=1:17
    cur_x = topmap(num,1);
    cur_y = topmap(num,2);
    cur_language = body_labels(num);
    cur_val = count_list(cur_x, cur_y);
    start_pos_x = (cur_x-1)*20+2;
    start_pos_y = (cur_y-1)*20+4+cur_val*4;
    text(start_pos_x, start_pos_y, cur_language,'FontWeight','bold');
    count_list(cur_x, cur_y) = count_list(cur_x, cur_y) + 1;
end













