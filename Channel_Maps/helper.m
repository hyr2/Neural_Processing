% This code is only for FLEX PCB backend with 4 shank NETs:
flag = '1x32';
% Using AutoCAD, generate a table Untitled1 of the shape of the BGA

% Hard coded Flex PCB channel map
load('flexpcb.mat');


% Loading table from excel file:
bga_to_device = table2array(Untitled1);
% Splitting into 
bga_to_deviceR = bga_to_device(:,[5:8]);
bga_to_deviceL = bga_to_device(:,[1:4]);

% mapping from electrodes to flex pcb contacts (connector side):
electrodeID_on_flexPCB_backend_R = zeros(size(flexpcbR_connectorside));
electrodeID_on_flexPCB_backend_L = zeros(size(flexpcbL_connectorside));
for iter_l = [1:64]
    tmp_replace = bga_to_deviceR(iter_l);
    electrodeID_on_flexPCB_backend_R(find(flexpcbR_connectorside == flexpcbR_bgaside(iter_l))) = tmp_replace;
    tmp_replace = bga_to_deviceL(iter_l);
    electrodeID_on_flexPCB_backend_L(find(flexpcbL_connectorside == flexpcbL_bgaside(iter_l))) = tmp_replace;
end


% Going to Intan
load('FlexadapterANDIntan.mat')         % Right side means U2 % Left side means U1
U3_flipped = fliplr(electrodeID_on_flexPCB_backend_L);  % This will connect to U2 on the rigid PCB adapter
U4_flipped = fliplr(electrodeID_on_flexPCB_backend_R);  % This will connect to U1 on the rigid PCB adapter

U1 = Intan_to_adapter_and_flexconnector_L;
U2 = Intan_to_adapter_and_flexconnector_R;
if flag == '1x32'
    final_channel_map = zeros(32,4);
elseif flag == '2x16'
    final_channel_map = zeros(16,8);
end
for iter_l = [1:64]

    [shankID,y_depth] = findShankID(U3_flipped(iter_l),flag);   % U3 connects with U2
    final_channel_map(round(y_depth),round(shankID)) = U2(iter_l);

    [shankID,y_depth] = findShankID(U4_flipped(iter_l),flag);   % U4 connects with U1
    final_channel_map(round(y_depth),round(shankID)) = U1(iter_l);

end
filename_local = string(timeofday(datetime('now')));
save(filename_local,'final_channel_map');



