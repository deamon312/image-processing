%%
clear all;
close all;
clc;
[file ,path] = uigetfile({'*.jpg;*.png;*.gif;*.tif';'*.*'},'File Selector');
selectedfile = fullfile(path,file);
image=imread(selectedfile);
figure,imshow(image),title('original');
[height, width, ~] = size(image);
%%
sigma_s=0.03*min(height,width);
sigma_r=15;
sigma_t=0.1;
p=0.95;
w=0.95;
t0=0.25;
minAtmosLight=220;
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
R=bilat_filter_grayscale(W/255,radius,sigma_s,sigma_r);
figure,imshow(R),title('R');
fprintf('R finished');

%%
ALight = min([minAtmosLight, max(max(255-W))]);

V_R=bilat_filter_joint(V/255,R,radius,sigma_t,sigma_r)*255;
figure,imshow(uint8(V_R)),title('V_r');
fprintf('V_R finished');
%%
d=-log(1-V_R/ALight)./beta;
figure,imshow(uint8(d)),title('depth from V_R');

% t=zeros(height,width);
t=ones(height,width)-w/ALight*V_R;
t_depth=-log(t)./beta;
figure,imshow(uint8(t_depth)),title('depth from t');

image_double=double(image);
J=zeros(size(image));
J(:,:,1)=(image_double(:,:,1)-ALight)./max(t,t0)+ALight;
J(:,:,2)=(image_double(:,:,2)-ALight)./max(t,t0)+ALight;
J(:,:,3)=(image_double(:,:,3)-ALight)./max(t,t0)+ALight;
figure,imshow(uint8(J)),title('J');
% imwrite(J,'fog2_out.jpg','jpg');