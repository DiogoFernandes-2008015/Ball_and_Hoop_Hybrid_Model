function [dx,y] = bandh_lugre(t,x,u,J,K,B,fc,fs,vs,s0,s1,s2,dv,varargin)
    %Ball and hoop linear model with viscous friction
    %Number of states = 2
    %Number of inputs = 1
    %Number of outputs = 1
    %Number of free parameters = 10
    %State Derivative equation
    if x<0
        fac = -1;
    else
        fac = 1;
    end
    s = fc+(fs-fc)*exp(-(x(1)/vs)^dv);
    dx = [1/J*(K*u-B*x(1)-fac*fc-s0*x(2)-s1*(x(1)-s0*abs(x(1))/s*x(2))-s2*x(1));
          x(1)-s0*abs(x(1))*x(2)/s];
    %Output equation
    y=x(1);