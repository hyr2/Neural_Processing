function t=get_tc_band(K,exp_time,beta)
  tc_tmp=logspace(-10,-1,50001); 
  K_tmp=sqrt(beta.*tc_tmp.*tc_tmp./(2*exp_time.*exp_time).*(exp(-2*exp_time./tc_tmp)-1+(2*exp_time./tc_tmp)));
  
  if(ndims(K)==2 & min(size(K))==1)
      t=interp1(K_tmp,tc_tmp,K);
  else
      Kfoo=reshape(K,[1 prod(size(K))]);
      t=interp1(K_tmp,tc_tmp,Kfoo);
      t=reshape(t,size(K));
  end
  