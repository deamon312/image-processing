function [Iout] = homomorphic_filter(I,H)
tic
I = im2double(I);
I_log = log(1 + I);
If = fft2(I_log);
If = fftshift(If);
figure,imshow(real(If)),title('I_fft_log');

Iout = abs(ifft2(ifftshift(H.*If)));
Iout = exp(Iout) - 1;
toc
end

