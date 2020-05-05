%% HARMONIC Inpainting 
% part of "MATLAB Codes for the Image Inpainting Problem"
%
% Usage:
% harmonic_inpainting(damaged_file,mask_file,output_file,lambda,tol,maxiter,dt)
%
% Author:
% Taras Rumezhak            (email: rumezhak@ucu.edu.ua)
% Date:
% May, 2020
%
% Refference to Carola-Bibiane Sch?nlieb book "Partial Differential Equation Methods for Image Inpainting"
% and Simone Parisotto (code example)
%
function harmonic_inpainting(damaged_file,mask_file,output_file,lambda,tol,maxiter,dt)
damaged = double(mat2gray(im2double(imread(damaged_file)))==1);
mask = double(mat2gray(im2double(imread(mask_file)))==1);
[M,N,C] = size(damaged);
%% LAPLACIAN
% GRID INTERVAL FOR AXIS ij
h1 = 1; h2 = 1;
d2i = toeplitz(sparse([1,1],[1,2],[-2,1]/h1^2,1,M));
d2j = toeplitz(sparse([1,1],[1,2],[-2,1]/h2^2,1,N));
% NEUMANN BOUNDARY CONDITIONS
d2i(1,[1 2])         = [-1 1]/h1;
d2i(end,[end-1 end]) = [1 -1]/h1;
d2j(1,[1 2])         = [-1 1]/h2;
d2j(end,[end-1 end]) = [1 -1]/h2;
% 2D domain LAPLACIAN
L = kron(speye(N),d2i)+kron(d2j,speye(M));
%% ------------------------------------------------------------ FREE MEMORY
clear d2i d2j
%% INPAINTINH ALGORITHM
% INITIALIZATION
u = damaged;
f = damaged;
% FOR EACH COLOR CHANNEL
for c = 1:C
    
    for iter = 1:maxiter
        
        % COMPUTE NEW SOLUTION
        laplacian = reshape( L*reshape(u(:,:,c),[],1) ,M,N);
        unew      = u(:,:,c) + dt*( laplacian + lambda*mask(:,:,c).*(f(:,:,c)-u(:,:,c)) );
        
        % COMPUTE EXIT CONDITION
        diff = norm(unew(:)-reshape(u(:,:,c),[],1))/norm(unew(:));
        
        % UPDATE
        u(:,:,c) = unew;
        
        % TEST EXIT CONDITION
        if diff<tol
            break
        end
        
    end  
end
%% WRITE IMAGE OUTPUT
imwrite(u,output_file)
endfunction
