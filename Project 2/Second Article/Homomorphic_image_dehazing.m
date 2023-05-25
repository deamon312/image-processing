 %%
clear all;
close all;
clc;
[file ,path] = uigetfile({'*.jpg;*.jpeg;*.png;*.gif;*.tif';'*.*'},'File Selector');
selectedfile = fullfile(path,file);
I=imread(selectedfile);
figure,imshow(I),title('original');
%% 进行同态滤波
% 取三个通道的平均灰度作为参照
I_mean = mean(I,3);
I_mean = im2double(I_mean);
%% Filter
gL = 0.3;
gH = 1.8;
C = 10;
D0 = 8;
tic
H= gaushp(I_mean, gL, gH, D0, C);
toc
figure,imshow(H,[]),title('H');

%%
D0=2;
n=100;
tic
H=butterhp(I_mean, D0,n);
toc
figure,imshow(H,[]),title('H');
%% 
D0=1;
tic
H=idealhp(I_mean,D0);
toc
figure,imshow(H,[]),title('H');
%%
I_gray_defog = homomorphic_filter(I_mean,H);
% Rescale[0,1]
I_gray_defog = rescale(I_gray_defog);
%% 利用同态滤波后的平均灰度来映射
I_defog = zeros(size(I));
for i = 1:3
    % 用去雾的平均灰度来映射
    I_defog(:,:,i) = (double(I(:,:,i)).*I_gray_defog);
end
I_defog = rescale(I_defog);
figure;
imshow(I_defog);


function [out]=butterhp(im, d,n)
 
h= size(im,1);
w= size(im,2);
[x y]= meshgrid(-floor(w/2): floor(w-1)/2, -floor(h/2):floor(h-1)/2);
out=1./(1.+(d./(x.^2+y.^2).^0.5).^(2*n));

end