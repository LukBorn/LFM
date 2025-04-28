clear all;
close all;

%% Input parameters:
load('PSFs.mat');
ImgMultiView=imread('test.tif');
RatioAB=0.93085;
ItN=30;
ROISize=300;

%% Main code
NxyExt=201;
Nxy=size(PSF1,1)+NxyExt*2;
Nz=size(PSF1,3);

OTF_A=complex(single(zeros(Nxy,Nxy,Nz)));
OTF_B=complex(single(zeros(Nxy,Nxy,Nz)));
for ii=1:Nz
    OTF_A(:,:,ii)=fft2(ifftshift(single(padarray(PSF1(:,:,ii),[NxyExt NxyExt],0,'both'))));
    OTF_B(:,:,ii)=fft2(ifftshift(single(padarray(PSF2(:,:,ii),[NxyExt NxyExt],0,'both'))));
end

NxyAdd=round((Nxy/RatioAB-Nxy)/2);
NxySub=round(Nxy*(1-RatioAB)/2)+NxyAdd;

Tmp1=complex(single(zeros(Nxy+NxyAdd*2,Nxy+NxyAdd*2)));
Tmp2=single(Tmp1);
Tmp3=zeros(Nxy,Nxy,'single');

ImgExp=padarray(single(ImgMultiView),[NxyExt NxyExt],0,'both');

ObjReconTmp=zeros(Nxy,Nxy,'single');
ObjRecon=ones(2*ROISize,2*ROISize,Nz,'single');
ImgEst=zeros(Nxy,Nxy,'single');
Ratio=zeros(Nxy,Nxy,'single');
for ii=1:ItN
    display(['iteration: ' num2str(ii)]);
    tic;
    ImgEst=ImgEst*0;
    for jj=1:Nz
        ObjReconTmp(Nxy/2-ROISize+1:Nxy/2+ROISize,Nxy/2-ROISize+1:Nxy/2+ROISize)=ObjRecon(:,:,jj);
        Tmp1(NxyAdd+1:end-NxyAdd,NxyAdd+1:end-NxyAdd)=fftshift(fft2(ObjReconTmp));
        Tmp2=abs(ifft2(ifftshift(Tmp1)));
        ImgEst=ImgEst+max(real(ifft2(OTF_A(:,:,jj).*fft2(ObjReconTmp))),0)...
            +max(real(ifft2(OTF_B(:,:,jj).*fft2(Tmp2(NxyAdd+1:end-NxyAdd,NxyAdd+1:end-NxyAdd)))),0);
    end
    Tmp4=ImgExp(NxyExt+1:end-NxyExt,NxyExt+1:end-NxyExt)./(ImgEst(NxyExt+1:end-NxyExt,NxyExt+1:end-NxyExt)+eps);
    Ratio=Ratio*0+single(mean(Tmp4(:))*(ImgEst>(max(ImgEst(:))/200)));
    Ratio(NxyExt+1:end-NxyExt,NxyExt+1:end-NxyExt)=ImgExp(NxyExt+1:end-NxyExt,NxyExt+1:end-NxyExt)./(ImgEst(NxyExt+1:end-NxyExt,NxyExt+1:end-NxyExt)+eps);
    Tmp2=Tmp2*0;
    for jj=1:Nz
        Tmp1(NxyAdd+1:end-NxyAdd,NxyAdd+1:end-NxyAdd)=fftshift(fft2(Ratio).*conj(OTF_B(:,:,jj)));
        Tmp2(NxySub+1:end-NxySub,NxySub+1:end-NxySub)=abs(ifft2(ifftshift(Tmp1(NxySub+1:end-NxySub,NxySub+1:end-NxySub))));
        ObjReconTmp(Nxy/2-ROISize+1:Nxy/2+ROISize,Nxy/2-ROISize+1:Nxy/2+ROISize)=ObjRecon(:,:,jj);
        Tmp3=ObjReconTmp.*(max(real(ifft2(fft2(Ratio).*conj(OTF_A(:,:,jj)))),0)+Tmp2(NxyAdd+1:end-NxyAdd,NxyAdd+1:end-NxyAdd))/2;
        ObjRecon(:,:,jj)=Tmp3(Nxy/2-ROISize+1:Nxy/2+ROISize,Nxy/2-ROISize+1:Nxy/2+ROISize);
    end
    toc
    % draw max projection views of restored 3d volume  
    figure(1);
    subplot(1,3,1);
    imagesc(squeeze(max(ObjRecon,[],3)));
    title(['iteration ' num2str(ii) ' xy max projection']);
    xlabel('x');
    ylabel('y');
    axis equal;

    subplot(1,3,2);
    imagesc(squeeze(max(ObjRecon,[],2)));
    title(['iteration ' num2str(ii) ' yz max projection']);
    xlabel('z');
    ylabel('y');
    axis equal;

    subplot(1,3,3);
    imagesc(squeeze(max(ObjRecon,[],1)));
    title(['iteration ' num2str(ii) ' xz max projection']);
    xlabel('z');
    ylabel('x');
    axis equal;
    drawnow
end
save('ObjRecon.mat','ObjRecon');
MIPs=[max(ObjRecon,[],3) squeeze(max(ObjRecon,[],2));squeeze(max(ObjRecon,[],1))' zeros(size(ObjRecon,3),size(ObjRecon,3))];
figure(2);imagesc(MIPs);axis image;