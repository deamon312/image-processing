 %%
clear all;
close all;
clc;
[file ,path] = uigetfile({'*.jpg;*.jpeg;*.png;*.gif;*.tif';'*.*'},'File Selector');
selectedfile = fullfile(path,file);
I=imread(selectedfile);
figure,imshow(I),title('Original');

%% 
I = im2double(I);
figure;imhist(I);title('Histogram_Org');
I_gray = rgb2gray(I);
%% Filter
gL = 0.3;
gH = 1.8;
C = 8;
D0 = 16;
H = gaushp(I_gray, gL, gH, D0, C);
plotTFSurface(H)
%%
D0 = 24;
n = 1;
tic
H = butterhp(I_gray, D0,n);
toc
plotTFSurface(H)
%% 
D0 = 4;
tic
H = idealhp(I_mean,D0);
toc
plotTFSurface(H)

%%
[I_gray_defog ,If,G] = homomorphic_filter(I_gray,H);
figure,imshow(If),title('I_f');
figure,imshow(G),title('H*I_f');
figure,imshow(I_gray_defog,[]),title('Defog');
%% 
I_defog = zeros(size(I));
for i = 1:3
    % 用去雾的平均灰度来映射
    I_defog(:,:,i) = I(:,:,i).*I_gray_defog;
end
I_defog = rescale(I_defog);

figure;imshow(I_defog);
figure;imhist(I_defog);
%%
LAB = rgb2lab(uint8(I_defog*255)); 
L = LAB(:,:,1)/100;
L = adapthisteq(L,'NumTiles',[8 8],'ClipLimit',0.005,'Distribution','rayleigh');
LAB(:,:,1) = L*100;
J_adapt = lab2rgb(LAB);
figure,imshow(J_adapt),title('RJ_adaptEQ');
figure;imhist(I);