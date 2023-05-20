%%
clear all;
close all;
clc;
[file ,path] = uigetfile({'*.jpg;*.jpeg;*.png;*.gif;*.tif';'*.*'},'File Selector');
selectedfile = fullfile(path,file);
image=imread(selectedfile);
figure,imshow(image),title('original');
[height, width, ~] = size(image);
%%
sigma_s=0.03*min(height,width);
sigma_r=20;
sigma_t=20;
p=0.95;
w=0.95;
t0=0.25;
beta=0.01;
radius = 3;
omega(1:2) = 15;

%%
W = double(min(image,[],3));
figure,imshow(uint8(W)),title('W');

%%
B = medfilt2(W,omega,'symmetric');
C=B-medfilt2(abs(W-B),omega,'symmetric');
V=max(min(p.*C,W),0);
figure,imshow(uint8(V)),title('V');
fprintf('V finished');
%%
R=bilat_filter(W,radius,sigma_s,sigma_r);
figure,imshow(uint8(R)),title('R');
fprintf('R finished');

%%
V_R=bilat_filter_joint(V,R,radius,sigma_s,sigma_r,sigma_t);
figure,imshow(uint8(V_R)),title('V_r');
fprintf('V_R finished');
%%
A  = min([estimateAtmosphericLight(W), max(max(255-W))]);
% t=zeros(height,width);
t=ones(height,width)-w*V_R/A;
t_depth=-log(t)./beta;
figure,imshow(uint8(t_depth)),title('depth image t');

image_double=double(image);
J=zeros(size(image));
J(:,:,1)=(image_double(:,:,1)-A)./max(t,t0)+A;
J(:,:,2)=(image_double(:,:,2)-A)./max(t,t0)+A;
J(:,:,3)=(image_double(:,:,3)-A)./max(t,t0)+A;
figure,imshow(uint8(J)),title('J');
% imwrite(J,'fog2_out.jpg','jpg');