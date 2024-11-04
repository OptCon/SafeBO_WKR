% MIT License
% 
% Copyright (c) 2024 Oleksii Molodchyk, Johannes Teutsch, Timm Faulwasser
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.


classdef objKernel
    %class for kernel regression

    properties
        ker
        n
        N
        xdata
        ydata
        Gram
        Gram_inv
        rho
        mean_ker
        var_ker
        var_wie
    end

    properties(Access=private)
        flagPinv
    end
   
    
    
    methods
        
        function obj = objKernel(n,rho,type,params)
            %dimension of input data:
            obj.n = n;
            
            %initialize:
            obj.N = 0;
            obj.xdata = [];
            obj.ydata = [];
            obj.Gram = [];
            obj.Gram_inv = [];
            obj.rho = rho;
            if(abs(rho)<1e-10)
                obj.flagPinv = true;
            else
                obj.flagPinv = false;
            end

            %kernel type:
            switch type
                case "se"
                    %squared exponential kernel:
                    sigmaSE = params.sigmaSE;
                    lSE = params.lSE;
                    obj.ker = @(x1,x2) sigmaSE^2*exp(-(x1-x2)'*(x1-x2)/(2*lSE^2));

                case "poly"
                    %polynomial kernel:
                    c1 = params.c1;
                    c2 = params.c2;
                    c3 = params.c3;

                    obj.ker = @(x1,x2) c1*(x1'*x2+c2)^c3;

                case "custom"
                    %custom kernel:
                    obj.ker = @(x1,x2) params(x1,x2);

                otherwise
                    error("Kernel type not supported.")
            end
            
            %initialize mean predictor:
            obj.mean_ker = @(x) 0;

            %initialize GP posterior variance:
            obj.var_ker = @(x) obj.ker(x,x);
            
            %initialize Wiener kernel variance:
            obj.var_wie = @(x) 0;

        end
        
        function obj = addData(obj, xdata_new, ydata_new)
            
            %number of new data points:
            N_new = size(xdata_new,2);
            if(size(xdata_new,1) ~= obj.n)
                error("Dimension of data is incompatible")
            end
            
            %update Gram matrix:
            Gram_new = blkdiag(obj.Gram,zeros(N_new));
            for i = 1:obj.N
                for j = 1:N_new
                    Gram_new(i, obj.N+j) = obj.ker(obj.xdata(:,i), xdata_new(:,j));
                    Gram_new(obj.N+j, i) = Gram_new(i, obj.N+j); %due to symmetry
                end
            end
            for i = 1:N_new
                for j = 1:N_new
                    Gram_new(obj.N+i, obj.N+j) = obj.ker(xdata_new(:,i), xdata_new(:,j));
                end
            end
            obj.Gram = Gram_new;

            %append new data:
            obj.xdata = [obj.xdata, xdata_new];
            obj.ydata = [obj.ydata, ydata_new];
            obj.N = size(obj.xdata,2);

            %update inverse of Gram matrix + rho * identity matrix:
            if(obj.flagPinv)
                obj.Gram_inv = pinv(obj.Gram);
            else
                obj.Gram_inv = (obj.Gram + obj.rho^2*eye(obj.N))\eye(obj.N); %inv(obj.Gram + obj.rho*eye(obj.N));
            end

            %update mean predictor:
            obj.mean_ker = @(x) (obj.kerVec(x)'*( obj.Gram_inv )*reshape(obj.ydata,[],1))';

            %update GP posterior variance:
            obj.var_ker = @(x) diag(obj.ker(x,x) - obj.kerVec(x)'*( obj.Gram_inv )*obj.kerVec(x))';
            
            %update Wiener kernel variance:
            obj.var_wie = @(x) diag(obj.rho^2 * obj.kerVec(x)'*( obj.Gram_inv^2 )*obj.kerVec(x))';
      
            %%% NOTE: diag(...)' is used to allow evaluating the function
            %%%       in parallel for a vector of inputs

        end


        function vec = kerVec(obj, x)
            %input-dependent vector of kernel evaluated at data points

            if(size(x,1) ~= obj.n)
                error("Input dimension is incompatible")
            end

            %number of points for evaluation:
            np = size(x,2);

            vec = zeros(obj.N,np);
            
            for j = 1:np
                for i = 1:obj.N
                    vec(i,j) = obj.ker(x(:,j), obj.xdata(:,i));
                end
            end
        end

    end
    
end