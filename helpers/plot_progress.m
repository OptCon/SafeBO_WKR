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


function plot_progress(f,kernel,x_max,x_min,f_min)
    % Function for plotting the learning progress

    %retrieve kernel functions and data:
    mean_ker = kernel.mean_ker;
    var_ker = kernel.var_ker;
    var_wie = kernel.var_wie;
    xdata = kernel.xdata;
    ydata = kernel.ydata;

    gcf;

    %create graphs to plot:
    numgrid = 50;
    xgrid = linspace(x_min,x_max,numgrid);
    vargrid_ker = var_ker(xgrid);
    vargrid_wie = var_wie(xgrid);
    meangrid_ker = mean_ker(xgrid);
    truegrid = f(xgrid);
    
    %plot:
    plot(xgrid,truegrid,'k', 'LineWidth',1.1);
    hold on
    plot(xgrid,meangrid_ker,'--r', 'LineWidth',1.1);
    plot(xgrid,meangrid_ker + sqrt(vargrid_ker),'-.b', 'LineWidth',1.2)
    plot(xgrid,meangrid_ker + sqrt(vargrid_wie),':g', 'LineWidth',1.2)
    yline(f_min,':k', 'LineWidth',1.1)
    plot(xdata,ydata,'mx','MarkerSize',8)
    legend('True $f(x)$','Mean predictor $\mu(x)$', '$\mu(x) \pm \sigma_{GP}(x)$', '$\mu(x) \pm \sigma_{WK}(x)$', ...
        '$f_{\min}$', 'Data points', 'AutoUpdate','off','Location','southeast', 'Interpreter', 'latex')
    plot(xgrid,meangrid_ker - sqrt(vargrid_wie),':g', 'LineWidth',1.2)
    plot(xgrid,meangrid_ker - sqrt(vargrid_ker),'-.b', 'LineWidth',1.2)
    xlim([x_min,x_max])
    ylim([-8,2])
    xlabel('Input $x$', 'Interpreter','latex')
    ylabel('Output $y = f(x) + $ noise', 'Interpreter','latex')
    hold off
    grid on


    %adjust figure position:
    figpos = [0.1 0.1 0.8 0.8];
    set(gcf,'units','normalized','outerposition',figpos)


end

