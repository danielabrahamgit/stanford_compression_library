% using poly on sens est

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Author: C
%% This version requires large RAM memory



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc

addpath(genpath('/local_mount/space/tiger/1/users/xc/share/spi_mrf_pack_sim/spiMRF_pack'))


%% 
n_window_for_signal_generation=1;
%% things you may wanna adjust to your needs


kloc_dcf_name='./kloc_DCF_r1_int3fov2211_18448_19035_g4s100_T2us.mat';% the path of kspace location and DCF

effMtx    = 220;    
glo_para.n_window=1;
%% parameters please do not change
numThread = 1;  
%% set environment
addpath('/usr/local/app/bart/bart-0.6.00/matlab')
setenv('TOOLBOX_PATH','/usr/local/app/bart/bart-0.6.00')

%% kloc/dcf prep 
load (kloc_dcf_name) 
n_interleaves=size(k_3d,3);
n_rot=size(k_3d,4);
n_point=size(k_3d,1); %% need modification
n_compcoil=1;



crds = permute(k_3d,[2,1,3,4]);
DCF=reshape(DCF,[n_point,n_interleaves,n_rot]);   


clear k_3d

spiral_select=1:n_interleaves;
DCF=single(DCF(1:n_point,spiral_select,:));
crds=single(crds(:,1:n_point,spiral_select,:));

n_frame=size(crds,4);



%% load dictionary

 load /local_mount/space/tiger/1/users/xc/share/spi_mrf_pack_sim/spiMRF_pack/dic/GE/mrf3d_s5/dic_cen_830_hg23_TR12p5TE1p75R1200_phase_iab_dense_TI15_fa75_inv180
 I=(I_a+I_b*47)/48;
 clear I_a I_b
 n_window=glo_para.n_window;
for ii=1:n_window:n_frame
     I_temp(:,:,:,ii)=sum((I(:,:,:,ii:ii+n_window-1)),4);
end
I=(I_temp(:,:,:,1:n_window:n_frame));
 clear I_a I_b I_temp
 
 I_dic=matrix_norm(I);
 
%% generate signal
slc_sel=110;

load /local_mount/space/tiger/1/users/xc/share/spi_mrf_pack_sim/data/T1T2P_tgasfs_r1_i500s5e2l1e5
T1_find_all=T1_find_all(:,:,slc_sel);
T2_find_all=T2_find_all(:,:,slc_sel);
p_all=p_all(:,:,slc_sel);
% change the T1/T2
% T1_find_all(:,:,101:120)=T1_find_all(:,:,101:120)*1.1;
% T2_find_all(:,:,101:120)=T2_find_all(:,:,101:120)*1.2;
%
for ii_x=1:size(T1_find_all,1)
    tic
    for ii_y=1:size(T1_find_all,2)
        for ii_z=1:size(T1_find_all,3)
         
temp_T1=abs(T1_find_all(ii_x,ii_y,ii_z)-T1);
T1_index=find(temp_T1==min(temp_T1(:)));
temp_T2=abs(T2_find_all(ii_x,ii_y,ii_z)-T2);
T2_index=find(temp_T2==min(temp_T2(:)));
img_signal(ii_x,ii_y,ii_z,1,:)=I_dic(T1_index(1),T2_index(1),1,:).*repmat(p_all(ii_x,ii_y,ii_z),[1,1,1,size(I_dic,4)]);


        end 
    end 
    toc
    
ii_x
end


clear Rec CurRec
%% load sensitivity map

load /local_mount/space/tiger/1/users/xc/share/spi_mrf_pack_sim/data/sens_self_mat220_r24c08_coil12b
sens=sens(:,:,slc_sel,:);
n_coil=size(sens,4);


%% generate signal
n_window=glo_para.n_window;
ii_count=1;


for ii_frame=1:n_frame
 tic   
     clear DCF_ii crds_ii raw_ii
     point_sel=1:n_point;
     
    for ii_temp=1:n_window_for_signal_generation
    frame_sel=ii_temp+ii_frame-1;
    
    index_help=mod(frame_sel-1,n_rot)+1;
    
    acc=1;
    spiral_sel=(1+mod(ii_temp-1,acc)):acc:n_interleaves*3/3; %under sample

    DCF_ii(:,:,ii_temp)=DCF(point_sel,spiral_sel,index_help);
    crds_ii(:,:,:,ii_temp)=crds(:,point_sel,spiral_sel,index_help);
   
    end

     DCF_ii=reshape(DCF_ii,[1,length(point_sel),length(spiral_sel)*n_window_for_signal_generation]);
     crds_ii=reshape(crds_ii,[3,length(point_sel),length(spiral_sel)*n_window_for_signal_generation])*effMtx;
 
image_signal_frame=repmat(img_signal(:,:,:,:,ii_frame),[1,1,1,n_coil]).*sens;

img_ref=sos(image_signal_frame,4);

kspace_signal_frame=bart(['nufft -t '],crds_ii,image_signal_frame);

 img_temp=bart(['nufft -i -d ' num2str(effMtx) ':' num2str(effMtx) ':' num2str(length(slc_sel)) ' -t'],crds_ii,kspace_signal_frame);

 % img_temp_pics=bart(['pics -d 5 -i 200 -s 1e-2 -R L:7:7:1e-5 -t'],crds_ii,kspace_signal_frame,sens);


kspace_signal_frame=reshape(permute(kspace_signal_frame,[4,2,3,1]),[n_coil,n_point,length(spiral_sel),n_window_for_signal_generation]);

kspace_signal(:,:,:,ii_frame)=kspace_signal_frame(:,:,:,1);
toc
ii_frame
end


