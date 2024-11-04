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


function plot_results(t,funmean,confmax,confmin,leglab)
    % Function for plotting results on regret / size of safe region
    
    plot(t,funmean(:,1),'g','LineWidth',1.1)
    hold on
    plot(t,funmean(:,2),'b','LineWidth',1.1)
    plot(t,funmean(:,3),'r','LineWidth',1.1)
    
    legend(leglab, 'Interpreter','latex','Location','southeast', 'AutoUpdate','off')
    
    p1 = fill([t;t(end:-1:1)],[confmax(:,1); confmin(end:-1:1,1)],'g');
    p1.FaceAlpha = 0.2;      
    p1.EdgeColor = 'none'; 
    
    p2 = fill([t;t(end:-1:1)],[confmax(:,2); confmin(end:-1:1,2)],'b');
    p2.FaceAlpha = 0.2;      
    p2.EdgeColor = 'none'; 
    
    p3 = fill([t;t(end:-1:1)],[confmax(:,3); confmin(end:-1:1,3)],'r');
    p3.FaceAlpha = 0.2;      
    p3.EdgeColor = 'none'; 
    
    
    hold off
    grid on

    xlim([t(1),t(end)])


end

