import tkinter as tk
from tkinter import filedialog
import numpy as np  
import matplotlib.pyplot as plt
import cv2 
from scipy import ndimage
from skimage import exposure ,metrics
from tkinter import ttk

def disp_img(img , title = 'img' ,text = {'text' : [None],'loc':[(165,500)]}):
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONUP:
            cv2.destroyAllWindows()
    
    I = img.copy()
    avg = np.mean(I)
    for  i , val  in  enumerate(text['text']):
        if avg> 100:
            cv2.putText(I, val, text['loc'][i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        else:  
            cv2.putText(I, val, text['loc'][i], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) 

    cv2.imshow(title ,I)
    cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)

    # Associate the callback function with the named window
    cv2.setMouseCallback(title, mouse_callback)


    ########################################### Convert Color Spaces #####################################
def BGRtoHSV(BGR):
    hsv = cv2.cvtColor(BGR, cv2.COLOR_BGR2HSV)
    return cv2.split(hsv)

def HSVtoBGR(H,S,V):
    hsv = np.stack([H,S,V],axis=2)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

################################################ Denoise TV  #########################################
def denoise_tv(image, weight=1/90, eps=1.e-6, max_num_iter=200):
    """Perform total-variation denoising on n-dimensional images.
    
    Parameters
    ----------
    image : ndarray
        n-D input data to be denoised.
    weight : float, optional
        Denoising weight It is equal to 1/lambda . The greater `weight`, the more denoising .
        
    eps : float, optional
        Relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:
            (E_(n-1) - E_n) < eps * E_0
        where E_0 is the initial value of the cost function.    

    max_num_iter : int, optional
        Maximal number of iterations used for the optimization.

    Returns
    -------
    out : ndarray
        Denoised array  uint8 [0 - 255].
    """
    image = image.astype(np.float64) # convert image to float
    ndim = image.ndim
    p = np.zeros((image.ndim, ) + image.shape, dtype=image.dtype)
    g = np.zeros_like(p)
    d = np.zeros_like(image)
    i = 0
    while i < max_num_iter:
        if i > 0:
            # d will be the (negative) divergence of p
            d = -p.sum(0)
            slices_d = [slice(None), ] * ndim
            slices_p = [slice(None), ] * (ndim + 1)
            for ax in range(ndim):
                slices_d[ax] = slice(1, None)
                slices_p[ax+1] = slice(0, -1)
                slices_p[0] = ax
                d[tuple(slices_d)] += p[tuple(slices_p)]
                slices_d[ax] = slice(None)
                slices_p[ax+1] = slice(None)
            out = image + d
        else:
            out = image
        E = (d ** 2).sum()

        # g stores the gradients of out along each axis
        # e.g. g[0] is the first order finite difference along axis 0
        slices_g = [slice(None), ] * (ndim + 1)
        for ax in range(ndim):
            slices_g[ax+1] = slice(0, -1)
            slices_g[0] = ax
            g[tuple(slices_g)] = np.diff(out, axis=ax)
            slices_g[ax+1] = slice(None)

        norm = np.sqrt((g ** 2).sum(axis=0))[np.newaxis, ...] # calculate magnitude
        E += weight * norm.sum() # Update cost function
        tau = 1. / (2.*ndim) # calc step 
        norm *= tau / weight
        norm += 1.
        p -= tau * g
        p /= norm
        E /= float(image.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if np.abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    print(i,tau)    
    return out.astype(np.uint8)

######################################## Adaptive Gamma Correction ###################################
def adaptive_gamma_transform(img, n,m):
    """
    Applies adaptive gamma transform on a given image.

    Args:
        img: A grayscale image to be processed.
        m: Size of the local area (height).
        n: Size of the local area (width).

    Returns:
        A gamma corrected image.
    """
    rows, cols = img.shape
    gamma_corrected = np.zeros((rows, cols))

    # Add small value to ignore zero division error and convert to float
    img = (img+1.)/255. 

    for i in range(rows):
        for j in range(cols):
            rmin = max(0, i - m//2)
            rmax = min(rows, i + m//2 + 1)
            cmin = max(0, j - n//2)
            cmax = min(cols, j + n//2 + 1)
            local_area = img[rmin:rmax, cmin:cmax]

            N = np.mean(local_area) # calculate mean on local area n x m
            b = np.var(local_area)  # calculate var on local area n x m

            # Calculate the gamma value.
            gamma = N/img[i,j] + b

            # Gamma correct the pixel value.
            gamma_corrected[i,j] = np.power(img[i,j], gamma)

    return (gamma_corrected*255).astype(np.uint8) # transform back to uint8 [0 - 255]

#################################################### MSR #############################################
def get_ksize(sigma):
    # Opencv calculates ksize from sigma as
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    # then ksize from sigma is
    # ksize = ((sigma - 0.8)/0.15) + 2.0

    return int(((sigma - 0.8)/0.15) + 2.0)

def get_gaussian_blur(img, ksize=0, sigma=5):
    # Perform convolution I(i,j)*G(i,j)
    # if ksize == 0, then compute ksize from sigma
    if ksize == 0:
        ksize = get_ksize(sigma)
    
    # Gaussian 2D-kernel can be seperable into 2-orthogonal vectors
    # then compute full kernel by taking outer product or simply mul(V, V.T)
    sep_k = cv2.getGaussianKernel(ksize, sigma)

    return cv2.filter2D(img, -1, np.outer(sep_k, sep_k))

def ssr(img, sigma):
    # Single-scale retinex of an image
    # SSR(x, y) = log(I(x, y)) - log(I(x, y)*G(x, y))
    # G = surrounding function,( Gaussian )
    
    return np.log10(img) - np.log10(get_gaussian_blur(img, ksize=0, sigma=sigma) + 1.0)

def msr(img, sigma_scales=[15, 80, 250],apply_normalization=True):
    # Multi-scale retinex of an image
    # MSR(x,y) = sum(weight[i]*SSR(x,y, scale[i])), i = {1..n} scales
    img = img + 1.0 # add small value to ignore log(0)
    msr = np.zeros(img.shape)
    # for each sigma scale compute SSR
    for sigma in sigma_scales:
        msr += ssr(img, sigma)
    
    # divide MSR by weights of each scale
    # here we use equal weights 
    msr = msr / len(sigma_scales)
    
    # computed MSR could be in range [-k, +l], k and l could be any real value
    # so normalize the MSR image values in range [0, 255]
    if apply_normalization: 
       return cv2.normalize(msr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    else: 
       return msr


############################### Multi-scale Hyperbolic Tangent Enhancement ###########################
def tanh(img, sigma):
    # Single-scale Hyperbolic Tangent Enhancement
    # tanh(I(x,y) / (I(x, y)*G(x, y)))
    # G = surrounding function,( Gaussian )
    return np.tanh(img/get_gaussian_blur(img, ksize=0, sigma=sigma))

def mtanh(img, sigma_scales=[15, 80, 250]):
    # Multi-scale Hyperbolic Tangent Enhancement
    img = img + 1.0 # add small value to ignore zero division
    i_t = np.zeros(img.shape)
    # for each sigma scale compute tanh
    for sigma in sigma_scales:
        i_t += tanh(img, sigma)
    
    # divide tanh by weights of each scale
    # here we use equal weights 1/3
    i_t = i_t / len(sigma_scales)
    
    return (i_t*255).astype(np.uint8) # transform back to uint8 [0 - 255]


################################### Double-Function Image Enhancement ################################
def DFIE(img , sigma=[10,40,300],n = 3,m= 3):
    i_l = msr(img,sigma).astype(np.float64) # calculate  weighted MSR
    i_t = mtanh(img,sigma).astype(np.float64) # calculate  weighted tanh
    
    # Using gausian to estimate mean , much faster than do in in loop
    i_l_mean = cv2.blur(i_l, (n, m))
    i_t_mean = cv2.blur(i_t, (n, m))
    a= i_t_mean/i_l_mean
    alpha = cv2.normalize(a, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    
    balanced = alpha*i_l + (1-alpha)*i_t
   
    # # Same as above but take much longer to calculate
    # rows, cols = img.shape
    # balanced = np.zeros((rows, cols))

    # for i in range(rows):
    #     for j in range(cols):
    #         rmin = max(0, i - m//2)
    #         rmax = min(rows, i + m//2 + 1)
    #         cmin = max(0, j - n//2)
    #         cmax = min(cols, j + n//2 + 1)
    #         # Calculate the indices for the local area.
    #         local_i_l = i_l[rmin:rmax, cmin:cmax]
    #         local_i_t = i_t[rmin:rmax, cmin:cmax]
    #         alpha = np.mean(local_i_t)/np.mean(local_i_l)
    #         balanced[i,j] = alpha*i_l[i,j]+(1-alpha)*i_t[i,j]

    return balanced.astype(np.uint8) # transform back to uint8 [0 - 255]


################################## Three-Dimensional Gamma Correction ################################
def three_dim_gamma_correction(image, weights=[0.05,0.05,0.2], n=3, m=3):
    # add some small value to ignore zerro division and convert to float [0.0 - 1.0]
    image = (image+1.)/255. 
    # Initialize output image
    output_image = np.zeros_like(image)
    rows, cols = image.shape
    
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    Gr =np.hypot(gx,gy) # calculate magnitude, same as sqrt(gx**2 +gy**2)
    # Iterate over each pixel in the input image
    for i in range(rows):
        for j in range(cols):
            rmin = max(0, i - m//2)
            rmax = min(rows, i + m//2 + 1)
            cmin = max(0, j - n//2)
            cmax = min(cols, j + n//2 + 1)
            # Extract local region of size n x m around the pixel
            local_region = image[rmin:rmax, cmin:cmax]
            # Compute local maximum, mean gradient, and variance
            local_max = np.max(local_region)
            local_mean_gradient = np.mean(Gr[i:i+n, j:j+m])
            local_variance = np.var(local_region)

            # Compute gamma correction factor based on local statistics and weights
            gamma = weights[0]*np.exp(image[i,j]/local_max) + weights[1]*np.exp(local_mean_gradient) + weights[2]*np.exp(local_variance)

            # Apply gamma correction to pixel value
            output_image[i,j] = np.power(image[i,j],gamma)*255

    return output_image.astype(np.uint8) # transform back to uint8 [0 - 255]

#################################### Adaptive Saturation  Correction #################################

def adaptive_saturation_adjustment(s_channel,n,m):
    # add some small value to ignore zerro division and convert to float [0.0 - 1.0]
    s_channel= (s_channel+1.)/255
    rows, cols = s_channel.shape
    saturation_corrected = np.zeros((rows, cols))

    # Get x-gradient in "sx"
    sx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=3)
    # Get y-gradient in "sy"
    sy = cv2.Sobel(s_channel, cv2.CV_64F, 0, 1, ksize=3)
    # Get square root of sum of squares

    Sg=np.hypot(sx,sy)
    
    # # Compute the global mean value of the S channel
    S_mean = np.mean(s_channel)

    for i in range(rows):
        for j in range(cols):
            rmin = max(0, i - m//2)
            rmax = min(rows, i + m//2 + 1)
            cmin = max(0, j - n//2)
            cmax = min(cols, j + n//2 + 1)
            # Extract local region of size n x m around the pixel
            local_region = s_channel[rmin:rmax, cmin:cmax]
            # Calculate the average  of the local area.
            Sm = np.mean(local_region)

            # Apply regulation
            if s_channel[i,j] <= S_mean+Sg[i,j]:
                saturation_corrected[i,j] = 1+0.8*np.log10(Sm/(s_channel[i,j]+0.5*Sg[i,j]))   
            else:
                saturation_corrected[i,j] = np.exp((Sm-s_channel[i,j])/2)
    
    return  cv2.normalize(saturation_corrected, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

############################### Multi Scale Retinex with Color Restoration ###########################
def color_balance(img, low_per, high_per):
    '''Contrast stretch img by histogram equilization with black and white cap'''
    
    tot_pix = img.shape[1] * img.shape[0]
    # no.of pixels to black-out and white-out
    low_count = tot_pix * low_per / 100
    high_count = tot_pix * (100 - high_per) / 100

    # channels of image
    ch_list = []
    if len(img.shape) == 2:
        ch_list = [img]
    else:
        ch_list = cv2.split(img)
    
    cs_img = []
    # for each channel, apply contrast-stretch
    for i in range(len(ch_list)):
        ch = ch_list[i]
        # cummulative histogram sum of channel
        cum_hist_sum = np.cumsum(cv2.calcHist([ch], [0], None, [256], (0, 256)))

        # find indices for blacking and whiting out pixels
        li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
        if (li == hi):
            cs_img.append(ch)
            continue
        # lut with min-max normalization for [0-255] bins
        lut = np.array([0 if i < li 
                        else (255 if i > hi else round((i - li) / (hi - li) * 255)) 
                        for i in np.arange(0, 256)], dtype = 'uint8')
        # constrast-stretch channel
        cs_ch = cv2.LUT(ch, lut)
        cs_img.append(cs_ch)
    
    if len(cs_img) == 1:
        return np.squeeze(cs_img)
    elif len(cs_img) > 1:
        return cv2.merge(cs_img)
    return None

def msrcr(img, sigma_scales=[15, 80, 250], alpha=125, beta=46, G=192, b=-30, low_per=1, high_per=1):
    # Multi-scale retinex with Color Restoration
    # MSRCR(x,y) = G * [MSR(x,y)*CRF(x,y) - b], G=gain and b=offset
    # CRF(x,y) = beta*[log(alpha*I(x,y) - log(I'(x,y))]
    # I'(x,y) = sum(Ic(x,y)), c={0...k-1}, k=no.of channels
    
    img = img + 1.0
    # Multi-scale retinex and don't normalize the output
    msr_img = msr(img, sigma_scales, apply_normalization=False)
    # Color-restoration function
    crf = beta * (np.log10(alpha * img) - np.log10(np.sum(img, axis=2, keepdims=True)))
    # MSRCR
    msrcr = G * (msr_img*crf - b)
    # normalize MSRCR
    msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    # color balance the final MSRCR to flat the histogram distribution with tails on both sides
    msrcr = color_balance(msrcr, low_per, high_per)
    
    return msrcr


################################################# CLAHE ##############################################
def CLAHE(Img):
    # Convert image to LAB color space
    lab = cv2.cvtColor(Img, cv2.COLOR_BGR2LAB)

    # Split LAB image into separate channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L channel with the other LAB channels
    lab_cl = cv2.merge((cl,a,b))

    # Convert back to RGB color space
    final = cv2.cvtColor(lab_cl, cv2.COLOR_LAB2BGR)
    return final 

################################### Adaptive histogram equalization ##################################
def AHE(Img):
    I = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    eq = exposure.equalize_adapthist(Img)
    return cv2.normalize(eq, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    
    
################################################ Metrics #############################################
def PSNR(I_r, I_f):
    mse = np.mean((I_r - I_f) ** 2)
    max_pixel = 255
    psnr = 10 * np.log10(max_pixel ** 2 / mse)
    # psnr = 20 * np.log10(max_pixel/np.sqrt(mse))
    return round(psnr, 4)


def SD(I_f):
    # Compute the histogram
    hist, bins = np.histogram(I_f.flatten(), bins=256)
    # Compute the mean of the histogram
    mean = np.sum(hist * bins[:-1]) / np.sum(hist)

    # Compute the variance of the histogram
    variance = np.sum((bins[:-1] - mean) ** 2 * hist) / np.sum(hist)

    # Compute the standard deviation of the histogram
    return round(np.sqrt(variance), 4)



def SSIM(I_r, I_f, L=255):
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    # INITS
    I2_2 = I_f ** 2  # I2^2
    I1_2 = I_r ** 2  # I1^2
    I1_I2 = I_r * I_f  # I1 * I2
    # END INITS
    # PRELIMINARY COMPUTING
    mu1 = cv2.GaussianBlur(I_r, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I_f, (11, 11), 1.5)
    mu1_2 = mu1 ** 2
    mu2_2 = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2  # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2  # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    ssim_map = t3 / t1
    mssim = np.mean(ssim_map)  # mssim = average of ssim map
    return round(mssim, 4)


def IE(I_f):
    # Add epsilon to avoid division by zero errors
    epsilon = 2**(-32) 
    # Compute the histogram of the image
    hist, _ = np.histogram(I_f.flatten(), bins=256)

    # Calculate the total number of pixels in the image
    num_pixels = np.sum(hist) # same as N*M

    # Calculate the PMF by dividing each bin in the histogram by the total number of pixels
    hist_p = hist / num_pixels

    hist_p = np.clip(hist_p, epsilon, 1)
    E = -np.sum(hist_p * np.log2(hist_p))
    return round(E, 4)


def metric(I_ref ,I_enc):
    I_ref_gray = cv2.cvtColor(I_ref, cv2.COLOR_BGR2GRAY).astype(np.float64)
    I_enc_gray = cv2.cvtColor(I_enc, cv2.COLOR_BGR2GRAY).astype(np.float64)
    info_ref = {'PSNR': PSNR(I_ref_gray, I_enc_gray),'SSIM': SSIM(I_ref_gray, I_enc_gray, L=255) ,'SD': SD(I_enc_gray) ,'IE':IE(I_enc_gray)}
    return info_ref


def Model(Img ,model = 'DFE' ,disp_selector = [False,False,False,False,False ,False,False ,False, False ,False]
           ,sigma = [10,40,400],weights=[0.05,0.05,0.2], kernel = [9,9],lam = 40):
    
    if model == 'DFE':
         # disp_selector = [Original , Original & I_o,  HSV ,I_d,I_p,I_out,I_img,I_u,S_tag ]
         h,s,v = BGRtoHSV(Img) 
         n , m = kernel
         ######################### V - Channel #########################
         I_v= denoise_tv(v, weight =1/lam, eps=1e-6, max_num_iter=100)
         I_d = adaptive_gamma_transform(I_v,n=3,m=3)
         I_p = DFIE(I_d , sigma,n ,m )
         I_out = three_dim_gamma_correction(I_d,weights,n,m)
         I_img = ((I_out/255.*I_p/255.)*255).astype(np.uint8)

         ######################### S - Channel #########################
         I_u= denoise_tv(s, weight=1/40, eps=1e-6, max_num_iter=100)
         S_tag = exposure.equalize_adapthist(I_u/255.)
         S_tag = cv2.normalize(S_tag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
         
         ############################# I_o #############################
         I_o = HSVtoBGR(h,S_tag,I_img)
         I_o = HSVtoBGR(h,I_u,I_img)
         I_o_cb = color_balance(I_o,1,1)
         performance = metric(Img,I_o)
         if disp_selector[0]:
            disp_img(I_o, title = 'I_o' ,text = {'text' : ['I_o'],'loc':[(280,460)]})
         if disp_selector[1]:
            disp_img(np.block([[Img],[I_o],[I_o_cb]]) , title = 'Enhancement' ,text = {'text' : ['Original','Enhancement','Enhancement + Color Balance'],'loc':[(280,460),(640+280,460),(640*2+230,460)]})     
         if disp_selector[2]:
            disp_img(np.block([h,s,v]) , title = 'HSV' ,text = {'text' : ['h-channel','s-channel','v-channel'],'loc':[(280,460),(640+280,460),(640*2+280,460)]})  
         if disp_selector[3]:
            disp_img(I_d, title = 'I_d' ,text = {'text' : ['I_d'],'loc':[(280,460)]})
         if disp_selector[4]:
            disp_img(I_p, title = 'I_p' ,text = {'text' : ['I_p'],'loc':[(280,460)]})
         if disp_selector[5]:
            disp_img(I_out, title = 'I_out' ,text = {'text' : ['I_out'],'loc':[(280,460)]})
         if disp_selector[6]:
            disp_img(I_img, title = 'I_img' ,text = {'text' : ['I_img'],'loc':[(280,460)]})
         if disp_selector[7]:
            disp_img(I_u, title = 'I_u' ,text = {'text' : ['I_u'],'loc':[(280,460)]})
         if disp_selector[8]:
            disp_img(I_v, title = 'I_v' ,text = {'text' : ['I_v'],'loc':[(280,460)]})
         if disp_selector[9]:
            disp_img(S_tag, title = 'S_tag' ,text = {'text' : ['S_tag'],'loc':[(280,460)]})
         return  performance
    if model == 'MSRCR':
         I_o = msrcr(Img,sigma_scales=sigma) 
         performance = metric(Img,I_o)
         if disp_selector[0]:
            disp_img(I_o, title = 'I_o' ,text = {'text' : ['I_o'],'loc':[(280,460)]})
         if disp_selector[1]:
            disp_img(np.block([[Img],[I_o]]) , title = 'Enhancement' ,text = {'text' : ['Original','Enhancement'],'loc':[(280,460),(640+280,460)]})

         return  performance    
    if model == 'CLAHE':
         I_o = CLAHE(Img) 
         performance = metric(Img,I_o)
         if disp_selector[0]:
            disp_img(I_o, title = 'I_o' ,text = {'text' : ['I_o'],'loc':[(280,460)]})
         if disp_selector[1]:
            disp_img(np.block([[Img],[I_o]]) , title = 'Enhancement' ,text = {'text' : ['Original','Enhancement'],'loc':[(280,460),(640+280,460)]})  

         return  performance            
    if model == 'AHE':
         I_o = AHE(Img)
         performance = metric(Img,I_o) 
         if disp_selector[0]:
            disp_img(I_o, title = 'I_o' ,text = {'text' : ['I_o'],'loc':[(280,460)]})
         if disp_selector[1]:
            disp_img(np.block([[Img],[I_o]]) , title = 'Enhancement' ,text = {'text' : ['Original','Enhancement'],'loc':[(280,460),(640+280,460)]}) 
 
         return  performance   
             

class ImageProcessorGUI:

    def __init__(self, master):
        self.master = master
        master.title("Image Processor")
        w = 380
        h = 350
        # open window in the center of screen
        screen_width = master.winfo_screenwidth()  # get the screen width
        screen_height = master.winfo_screenheight()  # get the screen height
        x = int((screen_width / 2) - (w / 2))
        y = int((screen_height / 2) - (h / 2))
        master.geometry('{}x{}+{}+{}'.format(w, h, x, y))  # window.geometry('wxh+x+y')
            
        # Select image button
        self.select_image_button = tk.Button(master, text="Select Image", command=self.select_image)
        self.select_image_button.grid(row=0, column=0, padx=10, pady=10)

        # Display Image button
        self.select_image_button = tk.Button(master, text="Display Selected",state='disabled', command= self.display_selected)
        self.select_image_button.grid(row=0, column=1, padx=10, pady=10)
        
        # Model combobox
        self.model_label = tk.Label(master, text="Select Model:")
        self.model_label.grid(row=1, column=0, padx=10, pady=5)
        
        self.model_combobox = ttk.Combobox(master, values=['MSRCR', 'CLAHE', 'AHE', 'DFE'],textvariable= 'Select',state='disabled')
        self.model_combobox.bind("<<ComboboxSelected>>", self.en)
        self.model_combobox.grid(row=1, column=1, padx=10, pady=5)
        
        # Value entries
        self.values_label_sigma = tk.Label(master, text="Enter Sigma values :")
        self.values_label_sigma.grid(row=2, column=0, padx=10, pady=5)
        
        self.values_entry_sigma = tk.Entry(master,state='disabled')
        self.values_entry_sigma.grid(row=2, column=1, padx=10, pady=5 )

        self.values_label_weight = tk.Label(master, text="Enter Weight values :")
        self.values_label_weight.grid(row=3, column=0, padx=10, pady=5)
        
        self.values_entry_weight = tk.Entry(master,state='disabled')
        self.values_entry_weight.grid(row=3, column=1, padx=10, pady=5)

        
        self.values_label_kernel = tk.Label(master, text="Enter Kernel size (n,m) :")
        self.values_label_kernel.grid(row=4, column=0, padx=10, pady=5)
        
        self.values_entry_kernel= tk.Entry(master,state='disabled')
        self.values_entry_kernel.grid(row=4, column=1, padx=10, pady=5)

        
        self.values_label_lambda = tk.Label(master, text="Enter Lambda value :")
        self.values_label_lambda.grid(row=5, column=0, padx=10, pady=5)
        
        self.values_entry_lambda = tk.Entry(master,state='disabled')
        self.values_entry_lambda.grid(row=5, column=1, padx=10, pady=5)
        
        # Checkboxes
        self.checkbox_frame = tk.Frame(master)
        self.checkbox_frame.grid(row=6, column=1, padx=10)
       
        self.checkbox_labels = ['I_o','I_i&o','HSV', 'I_d', 'I_p', 'I_out', 'I_img', 'I_u','I_v', "S'"]
        self.checkbox_vars = [tk.BooleanVar() for i in range(len(self.checkbox_labels))]
        self.checkbox_buttons = []
        
        for i in range(len(self.checkbox_labels)):
            self.checkbox_buttons.append(tk.Checkbutton(self.checkbox_frame, text=self.checkbox_labels[i], variable=self.checkbox_vars[i],state='disabled'))
            if i < len(self.checkbox_labels) // 2:
                self.checkbox_buttons[i].grid(row=i, column=0, sticky='w')
            else:
                self.checkbox_buttons[i].grid(row=i-len(self.checkbox_labels) // 2, column=1, sticky='w')
        
        self.metric_table = ttk.Frame(master)
        self.metric_table.grid(row=6, column=0, padx=5)
        self.metric_table_label = ttk.Label(self.metric_table, text="Metric Table")
        self.metric_table_label.grid(row=7, column=0, pady=(0, 1))

        self.metric_table_treeview = ttk.Treeview(self.metric_table, height=5)
        self.metric_table_treeview.grid(row=8, column=0)
        self.metric_table_treeview['columns'] = ("Metric-Name", "Metric-Value")

        # format columns
        self.metric_table_treeview.column("#0", width=0, stretch=False)
        self.metric_table_treeview.column("Metric-Name", width=100, minwidth=100, anchor="center")
        self.metric_table_treeview.column("Metric-Value", width=100, minwidth=100, anchor="center")

        # create headings
        self.metric_table_treeview.heading("#0", text="", anchor="w")
        self.metric_table_treeview.heading("Metric-Name", text="Metric-Name", anchor="center")
        self.metric_table_treeview.heading("Metric-Value", text="Metric-Value", anchor="center")
        

        # Run button
        self.run_button = tk.Button(self.checkbox_frame, text="Run",state='disabled', command=self.run)
        self.run_button.grid(row=7, column=0, padx=10, pady=10 )

          
    
    def en(self, event):
        self.run_button.config(state='normal')
        if self.model_combobox.get() == 'MSRCR':
           self.values_entry_sigma.configure(state='normal')
           self.values_entry_sigma.delete(0,'end')
           self.values_entry_sigma.insert(1,'30,100,300')
           self.values_entry_weight.configure(state='disable')
           self.values_entry_kernel.configure(state='disable')
           self.values_entry_lambda.configure(state='disable')
           for i in range(len(self.checkbox_buttons)):
               self.checkbox_buttons[i].configure(state='disable')
               self.checkbox_buttons[i].deselect()
           self.checkbox_buttons[0].configure(state='normal')
           self.checkbox_buttons[1].configure(state='normal')

        elif  self.model_combobox.get() == 'DFE': 
           self.values_entry_sigma.configure(state='normal')
           self.values_entry_sigma.delete(0,'end')
           self.values_entry_sigma.insert(1,'30,100,300')
           self.values_entry_weight.configure(state='normal')
           self.values_entry_weight.delete(0,'end')
           self.values_entry_weight.insert(1,'0.01,0.01,0.25')
           self.values_entry_kernel.configure(state='normal')
           self.values_entry_kernel.delete(0,'end')
           self.values_entry_kernel.insert(1,'9,9')
           self.values_entry_lambda.configure(state='normal')
           self.values_entry_lambda.delete(0,'end')
           self.values_entry_lambda.insert(1,'40')
           for i in range(len(self.checkbox_buttons)):
               self.checkbox_buttons[i].configure(state='normal') 

        else:
           self.values_entry_sigma.configure(state='disable')
           self.values_entry_weight.configure(state='disable')
           self.values_entry_kernel.configure(state='disable')
           self.values_entry_lambda.configure(state='disable')
           for i in range(len(self.checkbox_buttons)):
               self.checkbox_buttons[i].configure(state='disable')
               self.checkbox_buttons[i].deselect()
           self.checkbox_buttons[0].configure(state='normal')
           self.checkbox_buttons[1].configure(state='normal')

        for i in self.metric_table_treeview.get_children():
              self.metric_table_treeview.delete(i)         
        
    def select_image(self):
        file_paths = filedialog.askopenfilenames(title="Select Image Files", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.tif;*.bmp;")])
        if file_paths:
            image = cv2.imread(file_paths[0])
            self.image = cv2.resize(image, (640, 480))
            self.select_image_button.config(state='normal')
            self.model_combobox.config(state='readonly')
        else:
            self.image = None    

    def display_selected(self):
        disp_img(self.image , title = 'Reference' ,text = {'text' : ['Original'],'loc':[(280,460)]})
        cv2.waitKey(0)
        cv2.destroyAllWindows()  

    def insert_data_to_metric_table(self):
        for i in self.metric_table_treeview.get_children():
            self.metric_table_treeview.delete(i)

        for idx, (key, value) in enumerate(self.info.items()):
            self.metric_table_treeview.insert(parent='', index='end', iid=str(idx), values=(key, value))      
    
    def run(self):
        # Get values from GUI elements
        sigma =  None
        weights = None
        kernel = None 
        lam = None
        Img = self.image
        model = self.model_combobox.get()
 
        if model == 'DFE':
           sigma = [int(num)for num in self.values_entry_sigma.get().split(',')]
           weights = [float(num)for num in self.values_entry_weight.get().split(',')]
           kernel = [int(num)for num in self.values_entry_kernel.get().split(',')]
           lam = int(self.values_entry_lambda.get())
        elif model == 'MSRCR': 
           sigma = [int(num)for num in self.values_entry_sigma.get().split(',')] 
        
        
        checkboxes = [var.get() for var in self.checkbox_vars]
        print(checkboxes)

        # Display processed image
        self.info = Model(Img ,model =model ,disp_selector = checkboxes,sigma = sigma,weights=weights ,kernel =kernel ,lam = lam)
        self.insert_data_to_metric_table()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
if __name__ == '__main__':
    root = tk.Tk()
    
    gui = ImageProcessorGUI(root)
    root.mainloop()