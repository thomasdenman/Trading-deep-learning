# Trading-deep-learning (in progress)
Portfolio optimization algorithm uses proximal policy optimisation to learn to optimally allocate capital in portfolio.The program can take daily stock and cryptocurrency Data  from yahoo finance. This Data can then be used to train network to output learn to optimise portfolio returns. once trained the program output optimal weighting of portfolio on current day. 

Implementation is inspired by papers: 

For details on the exact algorithm used see:
***
1.&emsp; Model the spread of stock or portfolios as a Ornstein Ohlenbeck process 
$$dX^\alpha_t =  -\theta(X^\alpha_t - \mu)dt+\sigma dW(t)$$
where $$X^\alpha_t = S^1_t + \alpha S^2_t $$

2.&emsp; Fit OU process to timeseries data of some time window using maximising loglikelood function to find parameters. $$LL(\theta,\mu,\sigma|x^\alpha_0,x^\alpha_1,...)= 0.5ln(2\pi)-ln(\tilde{\sigma}) - \frac{1}{2\pi \tilde{\sigma}^2}\sum_{i=1}^N [x^\alpha_i -x^\alpha_{i-1}e^{\mu\Delta t}-\theta(1-e^{-\mu\Delta t})]^2$$

where $$\tilde{\sigma}^2 = \sigma^2\frac{1-e^{-2\mu\Delta t}}{\mu} $$
  
3.&emsp;from standardised OU-process 
$$dZ(t) = -Z(t) - \sqrt{2}dW(t)$$
&emsp;The time the process is most likely to returnto value c is found by maximising prob densityfunction:
$$f_{o,c}^{t} = \sqrt{\frac{2}{\pi}}\frac{|c|e^{-t}}{(1-e^{-2t})^{\frac{3}{2}}}exp\left(\frac{-c^2e^{-2t}}{2(1-e^{-2t})}\right)$$
$$t^*(c) = \frac{1}{2}ln\left(1+0.5\sqrt{(c^2-3)^2+4c^2}+c^2-3\right)$$
trading signal w_t is given 
$$w_t = 
    \begin{cases}
            1, &         \text{if } X^\alpha_t > \mu +c(\sigma/\sqrt{2\theta})\\
            -1, &         \text{if }  X^\alpha_t < \mu +c(\sigma/\sqrt{2\theta})\\
            0,& else .
    \end{cases}$$
Dependant on inequality short and long postions are opened on the stocks when price exeeds thresholds. postions are then unwinded after time T given by 
$$T = \frac{t^*(c)}{\theta} $$
typically c is set to the mean of lookback window.
