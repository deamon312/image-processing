function [ out ] = bilat_filter_joint( in,D,radius,sigma_s,sigma_r )

[X,Y]=meshgrid(-radius:radius,-radius:radius);
G = exp(-(X.^2+Y.^2)/(2*sigma_s^2));  
sz=size(in);
out=zeros(sz);
for i=1:sz(1)
    for j=1:sz(2)
        imin=max(i-radius,1);
        imax=min(i+radius,sz(1));
        jmin=max(j-radius,1);
        jmax=min(j+radius,sz(2));
        
        D_1=D(imin:imax,jmin:jmax);
        in_1=in(imin:imax,jmin:jmax);
        
        H=exp(-(D_1-D(i,j)).^2/2/sigma_r^2);
        
        F = H.*G((imin:imax)-i+radius+1,(jmin:jmax)-j+radius+1);  
        out(i,j) = sum(F(:).*in_1(:))/sum(F(:)); 
    end
end
end

