function [dx,y] = bandh_linear(t,x,u,J,K,B,varargin)
    %Ball and hoop linear model with viscous friction
    %Number of states = 1
    %Number of inputs = 1
    %Number of outputs = 1
    %Number of free parameters = 3
    %State Derivative equation
    dx = 1/J*(K*u-B*x);
    %Output equation
    y=x;
