function [dx,y] = bandh_dahl(t,x,u,J,K,B,fc,s0,dd,varargin)
    %Ball and hoop linear model with viscous friction
    %Number of states = 2
    %Number of inputs = 1
    %Number of outputs = 1
    %Number of free parameters = 6
    %State Derivative equation
    if x(1)<0
        fac = -1;
    else
        fac = 1;
    end
    if 1-fac*x(1)*s0*x(2)/fc<0
        fac2 = -1;
    else
        fac2 = 1;
    end
    dx = [1/J*(K*u-B*x(1)-fc*fac-s0*x(2));
          x(1)*fac2*(1-fac*x(1)*s0*x(2)/fc)*abs(1-fac*(x(1))*s0*x(2)/fc)^dd];
    %Output equation
    y=x(1);