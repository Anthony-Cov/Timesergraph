# Timesergraph
<I>Time series analysis by means of graph network</I> </p>

<H2>Content</H2>
<B>Libraries</B> - the folder that contains procedures for time series forecasting, graph construction and some other useful deeds. All of the forecasting procedures are unified for input and output parameters. </p>
<p>
<LI>Autoregr.py - autoregressive predicting model with research tool that helps to test the model on different parts of a series. (The same function is available for the other forecasting models)</LI>
<LI>ChooChoo.py - forecasting model based on maximal likeness invented by Irina Chuchueva. </LI>
<LI>EmbDim.cpp, EmbDim.h, EmbDim.so - computation of embedding dimencion (C++ for Python). </LI>
<LI>features.py - time series features calculation. </LI>
<LI>graph.py - graph construction and manipulation. </LI>
<LI>HurstExp.cpp, HurstExp.h, HurstExp.so - computation of Hurst's exponent (C++ for Python). </LI>
<LI>Localapp.py - forecasting model based onlocal approximation invented by prof. Yuriy Loskutov. </LI>
<LI>NeurosV.py - predicting recurrent neural network with LSTM sells.</LI>
<LI>Spectrum.py - forecasting model of singular spectrum analysis. </LI>
<LI>Util.py - some useful things. </LI>
</p>
<B>RealWeekly</B> - the folder with 34 real-world time series for a 10-year period of weekly frequency from open sources.</p>
<B>ArtSerGenerator.py</B> - the script that generates arificial time series as the sums of periodical components, transition process, white noise and random walking taken in different proportions. It places them into the 'Art_series' folder which will be created if not exist or emptied if contains any file.s</p>
<B>Classifier.ipynb</B> - experiments with classification models trained on artificial data applied to real-world series.</p>
<B>ForecastAndFeatures.py</B> - computation of mean MAPE for each of the forecasting models, graph characteristics and series features for each time series in data set.</p>
<B>Predictab.ipynb</B> - experiments on time series clustering by characteristics of corresponding graph.</p>
<H2>Requirements and dependences</H2>
<LI>numpy, pandas, matplotlib as usual</LI>
<LI>itertools - for artificial time series generator</LI>
<LI>networkx - for graph construction and its characteristics computation</LI>
<LI>scipy, sklearn - for time series features computation</LI>
<LI>keras - forecasting model and classificator</LI>
<LI>xgboost - for Classifier.ipynb</LI>
<LI>ctypes (C++ objects (EmbDim.so and HurstExp.so) were compiled with gcc for Windows'10 and wrapped to CEmbDim and CНurst functions to be used with Python. C++ sourse codes are available here, besides, there are 'pure' Python functions for the same purposes (DimEmb and HurstTraj) in <I>features.py</I> module) </LI>
</p>
