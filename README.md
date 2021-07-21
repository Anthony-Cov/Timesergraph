# Timesergraph
<I>Time series analysis by means of graph network</I> </p>
<B>Libraries</B> - the folder that contains procedures for time series forecasting, graph construction and some other useful deeds. All of the forecasting procedures are unified for input and output parameters.</p>
<p>
<LI>Autoregr.py - autoregressive predicting model with research tool that helps to test the model on different parts of a series </LI>
<LI>ChooChoo.py - forecasting model based on maximal likeness invented by Irina Chuchueva </LI>
<LI>EmbDim.cpp, EmbDim.h, EmbDim.so - computation of embedding dimencion (C++ for Python) </LI>
<LI>fakeseries.py - initial functions for <I>generator.py</I></LI>
<LI>features.py - time series features calculation </LI>
<LI>generator.py - fake series generator that uses neural networl</LI>
<LI>graph.py - graph construction and manipulation </LI>
<LI>HurstExp.cpp, HurstExp.h, HurstExp.so - computation of Hurst's exponent (C++ for Python) </LI>
<LI>Localapp.py - forecasting model based onlocal approximation invented by prof. Yuriy Loskutov </LI>
<LI>NeurosV.py - predicting recurrent neural network with LSTM sells</LI>
<LI>Spectrum.py - forecasting model of singular spectrum analysis </LI>
<LI>Tests.py - causality tests</LI>
<LI>Util.py - some useful things </LI>
</p>
<B>RealWeekly</B> - the folder with 34 real-world time series for a 10-year period of weekly frequency from open sources.</p>
<B>ArtSerGenerator.py</B> - the script that generates arificial time series as the sums of periodical components, transition process, white noise and random walking taken in different proportions. It places them into the 'Art_series' folder which will be created if not exist or emptied if contains any files</p>
