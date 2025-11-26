%%
%Construção do modelo híbrido com base no modelo narmax
dados_treinamento = load('pythondatatrain.mat')
dados_validacao = load('pythondataval.mat')
load('Resultados Finais\resultados_narmax.mat')

%%
%Construção dos resíduos para treinamento da rbf
e_train = dados_treinamento.y/10-y_hat_train;
x_train = resample(x_train,10,85);
e_train = resample(e_train,10,85); 
net_narmax = newrb(x_train',e_train',0,1,200,10)

%Free run simulation com a rede construída
e_hat_train = zeros(length(x_train),1);
for i=1:length(x_train)
    e_hat_train(i,1) = sim(net_narmax,x_train(i));
end
e_hat_valid = zeros(length(x_valid),1);
for i=1:length(x_valid)
    e_hat_valid(i,1) = sim(net_narmax,x_valid(i));
end
y_hathib_train=y_hat_train+e_hat_train;
y_hathib_valid=y_hat_valid+e_hat_valid;

%%
%Métricas e plotagens
%Resultados=result_maker(dados_treinamento.t,dados_validacao.t,x_train,x_valid,dados_treinamento.y/10,dados_validacao.y/10,y_hathib_train,y_hathib_valid,"NARMAX-Hibrido")

%save('Resultados Finais\resultados_narmax_hibrido.mat',"y_hathib_valid","y_hathib_train","net_narmax")