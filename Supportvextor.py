{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[*********************100%%**********************]  1 of 1 completed\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import yfinance as yf\n",
    "import datetime\n",
    "from datetime import date, timedelta\n",
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "# Input data files are available in the \"../input/\" directory.\n",
    "# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory\n",
    "\n",
    "\n",
    "import math,datetime\n",
    "import time\n",
    "import arrow\n",
    "from sklearn import preprocessing,model_selection,svm # Preprocessing for scaling data,Accuracy,Processing speed ,cross validation for training and testing\n",
    "from sklearn.linear_model import LinearRegression #\n",
    "import matplotlib.pyplot as plt\n",
    "from matplotlib import style\n",
    "style.use('ggplot')\n",
    "import pickle\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import yfinance as yf\n",
    "from datetime import datetime\n",
    "end = datetime.now()\n",
    "start = datetime(end.year-20, end.month, end.day)\n",
    "stock = \"KOTAKBANK.NS\"\n",
    "data = yf.download(stock, start, end)\n",
    "data[\"Date\"] = data.index\n",
    "data = data[[\"Date\", \"Open\", \"High\", \"Low\", \"Close\", \"Adj Close\", \"Volume\"]]\n",
    "data.reset_index(drop=True, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], dtype='object')"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "start_date = '2003-1-1'\n",
    "end_date = '2024-5-15'\n",
    "\n",
    "fill = (data ['Date']>=start_date) & (data ['Date']<=end_date)\n",
    "data = data .loc[fill]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "        Open       High        Low      Close  Adj Close    Volume\n",
      "0  17.500000  19.090000  16.600000  18.872499  18.389887   2057200\n",
      "1  19.250000  20.100000  18.775000  19.110001  18.621307   1573960\n",
      "2  19.600000  20.495001  17.549999  18.580000  18.104866  10822400\n",
      "3  19.177500  19.225000  18.309999  18.480000  18.007420   6484440\n",
      "4  18.549999  18.650000  17.400000  17.557501  17.108513   2814060\n"
     ]
    }
   ],
   "source": [
    "data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]\n",
    "print(data.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "x=data[['Open','High','Low','Volume']].values\n",
    "y=data['Adj Close'].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "regressor=LinearRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "regressor.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[-5.18149451e-01  8.54749111e-01  6.61333316e-01 -5.50127274e-08]\n",
      "-1.8299589180882094\n"
     ]
    }
   ],
   "source": [
    "print(regressor.coef_)\n",
    "print(regressor.intercept_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 140.45349491 1714.79532121 1034.80983095  617.97936064 1714.42473794\n",
      "   16.56613809  694.8523516   195.41261417   59.14956812 1675.1163037\n",
      "  979.73871793 1517.13601582  197.72474845 1861.17358477 1458.52776131\n",
      "  996.16776422  255.55187997  189.90076434  439.49507247   23.35061674\n",
      "  184.36101942  657.54601045  313.85486447  688.74520049  196.12636034\n",
      " 1672.97551567 1016.88512909 1801.30179083   75.90002524 1827.41892262\n",
      "  771.5903779  1880.4587812   493.19886802  377.28663625  480.92966679\n",
      "  217.2070487   225.11262447 1252.89888703  182.45379753  230.89507118\n",
      "   77.00861266  180.02658073  689.99421922  346.47108419  257.56960774\n",
      "  285.6994569   187.21207723 1012.71567891  300.09065718  362.96812598\n",
      "  385.3421967  1718.94091468   59.27736121  340.60972682   90.9032868\n",
      "   15.76388348  189.2019211   123.87627901  328.25220815  139.49691858\n",
      "  103.29774648   24.47158885 1929.48066942  163.37861781  322.85041022\n",
      " 1485.70390386  195.49267636 2032.31391228   56.19269267  227.36796104\n",
      "  144.5815229   521.64958188  360.02411348  182.80238821   92.247722\n",
      "  138.27110965 1854.83208219 1763.58484181  100.5971015  1179.62054344\n",
      " 1235.93936963 1799.30630392  635.92811038 1275.78782766  317.47465386\n",
      "  672.67044056  221.95521614  245.5270849   270.2385789   240.06199737\n",
      "  843.50560979   41.02582462   25.14568171   46.81986702 1873.6791125\n",
      "  885.06631782  197.7102955   291.89613312  913.79599811  202.0344269\n",
      " 1342.38698529   16.73276972 1916.72781037   48.23463793  320.53103756\n",
      "  193.53670704 1333.58092926   15.84348025   38.54016103 1883.99931846\n",
      "   93.78971694 1872.16874021  280.33243552  535.2006954  1495.46895626\n",
      "  239.30812091   55.89710757  104.53117896  786.98453773  283.6291225\n",
      "  263.90998026   19.47722032 1678.97932977   24.48790876  665.36772539\n",
      "  209.4821211   400.4817638    76.58373814  663.67602733  545.05095813\n",
      "  228.17264748  239.62993019  904.84493335 1500.32364114   80.98550417\n",
      "  352.80850942  369.01656851  618.83483836  234.14111054  960.18157331\n",
      " 1745.61303626 1814.35785297 1731.33872097  242.69691655   92.14267439\n",
      "   50.32366263  157.37877156 1107.0613407   775.8670898   383.3337972\n",
      "  158.86600006 1994.36547694  684.09588783  647.07124013 1494.36485489\n",
      "  139.75134429   16.10622971   16.87934189  197.52288631   34.91730206\n",
      " 1780.93101663   26.79137463  647.05580821 1853.85910091  203.34777844\n",
      "  742.5735646   301.97727855 1941.27443241 1254.21183175  987.34765904\n",
      " 1702.74947455  152.77338111  228.82561664 1780.28623666  774.41153607\n",
      "  169.98870409   94.19927435 1969.14340663  227.1619524  1452.73693762\n",
      " 1196.47366851  116.95540229   17.00093751  923.12131617  230.45137343\n",
      " 1456.62288965  204.95014075  171.17061992 1710.90573524  121.4982161\n",
      " 1753.99794525 1253.11057516  795.08813545 1272.09344624  153.22104935\n",
      "  132.10514472  312.11083124  709.81329506  177.54295123 1968.84700978\n",
      " 1333.12918653  344.72297766   73.96010735 1974.13793446 2124.48643077\n",
      "  622.03680089 1614.40977318   48.72625528 2030.08170355 1503.43974361\n",
      "  435.50725705  787.19188033  211.40574386 1268.85127816   88.30553834\n",
      "   59.00510726 1318.79447498  676.55485693 1796.65536198  225.020984\n",
      " 1856.64082828  197.50913924  382.03847073  105.17424072  109.69228085\n",
      " 1882.42748209   40.99747129  177.78740851 1417.99170435   26.66248995\n",
      "   15.55219838  753.08679071  114.7440323   573.53363922  357.66626061\n",
      " 1876.20717703 1155.76913556 1380.97806823  382.66231996 1136.64768539\n",
      "  226.16919434  942.70386532 1872.00453238  648.75837493 1850.44336339\n",
      "  731.09975212   28.56414291 1230.75940947  726.67626013  208.4445121\n",
      " 1236.74127405  366.9769863  1254.95528735 1811.84580876  277.48402233\n",
      "  355.53789025 2027.4215597  1745.08844819   59.77935348  241.24086591\n",
      " 1795.72105625   46.28830368  713.40170526  686.43655007 1708.30131971\n",
      "  428.60721497   69.56858293   53.85576103  384.54367864 1713.798115\n",
      " 1336.6543488  1771.45629023 1708.12458326  178.50088415  151.78562023\n",
      " 1756.69278903 1772.221421     82.27307778 1729.75581278  158.53450327\n",
      "  399.27452731  814.87316884 1817.22026858  676.88524604 2200.34165743\n",
      "  253.16279388   82.16771616  201.03775791  228.0231663  1761.9613603\n",
      " 1921.36432242 2018.35669538 1207.74950456  234.94668306  235.61026293\n",
      " 1086.23734843 1300.09439324 1317.08249747 2097.65165434 1276.5683746\n",
      "  198.47977858 1816.09412127  331.23119791  650.75710806  678.21201906\n",
      "  676.9227862  1672.17374313 1524.98032591  323.48640026 1790.88658122\n",
      "  152.13507211 1317.47308594 1908.70109629  745.78070908  179.57495433\n",
      "  377.94551943   75.38591816 1651.23710685 1022.66316316 1342.16887463\n",
      "   49.45159215 1337.59443396  381.23841507  185.81626844  219.57632703\n",
      "  986.24812106  709.64362669   90.89751303  620.70523842 1752.60471296\n",
      "  470.39074225   58.59124847  162.9630432  1940.14134461 1244.42059479\n",
      " 1165.06098151  185.87456441 2030.22779292 1306.23574719  204.97233021\n",
      " 1921.54025382   92.10157956 1832.38348511  807.26526352   31.92958159\n",
      "  182.37457865   16.67188015   76.37023323   46.34450395  193.81885934\n",
      "  239.18397156 2046.38465362 1794.8203365    15.9122178   238.05459593\n",
      "  744.24255215   90.36055831  682.81885539   75.01996262 1736.21810889\n",
      "  298.26865081   37.86212222 1340.79631503  467.54418113 1025.08744983\n",
      "  658.93622236  251.3763577    47.55924738  194.99796262  653.04860225\n",
      " 1861.50943033  334.77301785  140.47731377 1923.20474752 1356.69614392\n",
      "  189.74318967 1940.04356035  323.01011959  208.51213363  281.29518416\n",
      "  657.43894541 1834.41207644  796.45238634  309.933739     17.08604016\n",
      "  195.77468631  669.68837815   32.53672091   76.79416971   48.77398471\n",
      " 1909.83359084  225.4734231   188.65561945  167.0507628   779.63758657\n",
      "  290.50490179  262.97543572  700.47384061  173.68593394  716.62733151\n",
      "   57.50881622 1015.55964681   66.06283465  728.731378    108.06064134\n",
      "  792.59045633   58.11502576  229.83131526 1587.77459115  183.6570463\n",
      "  782.0310678  1909.17774536  687.14711084   35.09460301   59.02488156\n",
      "   75.59669824  325.38677931 1269.52243103  157.31609103  220.42798243\n",
      " 1568.38254942   83.49553911  778.17960192   79.96719953 1167.46238274\n",
      "  373.3944941   178.85091207   88.95505988 1912.97442629  945.06935348\n",
      " 1631.82548906 1720.98736154   48.09366744  187.05390029  276.44065625\n",
      " 1146.0555198   332.61964923  102.79993929  181.18636122   75.33134199\n",
      " 2027.23978175  939.57291206  851.65674921  213.70134953  704.50308713\n",
      "  181.37412853  199.33641583 1473.57539987   68.55424738  299.30969973\n",
      " 1317.06813659   39.37344291  204.15303306 1478.28034038  223.912172\n",
      "   77.70347961  347.92338468   63.31948911  697.89298515   14.88847901\n",
      "  352.11835458  322.52599444  157.70324206 1733.57127154  621.81636218\n",
      "   94.6475055  1761.19165708   16.63108918  984.85491522 1010.88453472\n",
      "  751.6639335   792.59379285  218.35166882  677.83336694  805.7954284\n",
      "  153.54028004  786.55641025 1591.51751343 1589.94988837  643.81299928\n",
      "   89.90934579   72.55937608 1645.43387366 1727.72782149 1787.94475386\n",
      " 1934.56209425  180.62438966 1711.1631512   692.51503671  266.90893954\n",
      "  911.2732793   191.00771007 1926.43437726   65.9869687    16.73462948\n",
      "   41.13116538   57.98297751 1324.16586521  199.25753751 1922.42793628\n",
      "  179.38278517 1704.77668094  645.35195924 1724.65339406 1107.68302896\n",
      " 1755.31682393 1912.22771617  359.64756104 1212.34238611   58.53954265\n",
      "   75.61393985   22.48762267 1235.40947675  275.38339373   31.68796544\n",
      "  209.05303055 1579.28665237 1105.67757996 1035.29325423  196.61750295\n",
      "  152.03501371  465.25209762 1040.73673686 1844.30477679  686.53600801\n",
      "   37.82570182   96.94053938  324.42831846  811.84770187 1783.34703376\n",
      "   69.47720663  236.25733343  220.24244146   16.11588454  160.02529554\n",
      " 1122.9116266   370.46241894 1496.03642942   24.9896223    50.68262943\n",
      "  137.09061121  780.55296703  750.56415617  186.20854445  551.26705285\n",
      "  718.45122586  709.6994905  1335.83486316  224.11278526 1584.8310296\n",
      "  617.39831046  185.08796159 1034.75370292  221.11488281  189.25346281\n",
      "  206.24163487 1866.19587437 1651.21331153  144.99952195  181.83761106\n",
      " 1338.7544672    31.88403565 1284.08685055 1039.97573442  173.6645128\n",
      " 1895.08270265  202.92581967   90.61502958  115.37808389  775.67418974\n",
      "   30.60420672   24.86542057 1253.57540879 1694.8493903  1950.75873152\n",
      "  239.79628642  314.21299882  963.25432574  343.65616118  664.98968853\n",
      " 1860.10894301  219.42770964 1838.70721686  607.44772449  374.77804239\n",
      " 1850.1110477    54.54683228 1284.50424275  727.92358235  247.84732975\n",
      "  335.05525051  677.21424702  351.59019351  667.5408031   717.47989435\n",
      "  349.68099561 1792.69427067  283.79898805 1695.5682822  1870.37662337\n",
      "  123.4252492  1816.04767786 1901.17792457   34.91178386  275.28436383\n",
      " 1771.39891193 1353.69649465 1238.14313636  195.22096702  146.54561131\n",
      "  349.1961369  1859.90760239   59.21541188 1333.30436397 1150.07411638\n",
      "  804.66831174 1766.27247932   16.62276493  696.64305505   48.90275365\n",
      "   60.28894747  163.21216404  625.12343482  222.39237939   70.34517364\n",
      "  204.5585606  1272.09115124  160.40389071  670.17532708 1838.94226511\n",
      " 1076.57555706   27.73668803  118.9593155    16.63385093  809.19878175\n",
      "  554.41828994  265.89705225 1481.07812735  782.69788046 1238.75127256\n",
      "  256.23434465 1869.93287886 2118.51049163 1886.07219979 1757.87006304\n",
      "  263.23454354  353.41653899 1181.37658461  159.60391131 1626.32784024\n",
      " 1662.64620475  675.28279466  165.5830372    65.07130302  316.66070689\n",
      " 1669.33883161  616.95887224  687.45802862  773.24335846 1240.69684934\n",
      "  697.28255209 1898.10178032 1456.5741042   723.39489678   91.38746025\n",
      "  968.85940605  289.78525371 1300.84063717 1386.48636344 1915.86185743\n",
      "  322.26522254 1219.25424255 1860.72660782  682.2723208   369.66067678\n",
      " 1402.71999578  319.31051867  438.82570191  698.85723592 1334.53916006\n",
      "   58.63485084  459.90941395 1776.59812758  265.34661891 1750.76651677\n",
      "  177.9167332    59.73348684 1482.99926434  656.95166306 1028.13783831\n",
      " 1886.45663666 1238.12661914  651.53418144  241.48925326 1992.65838392\n",
      "  841.90885027  226.61140104 1663.1800734   226.98720158  284.98081564\n",
      "  105.43063347  766.19414285  693.33029537  343.58987907 1319.56576935\n",
      " 1845.09294963 1861.22045253  309.3767491   222.90206495  983.92484259\n",
      " 1883.07668373 1759.64831115  467.80590213 1739.92928836   22.18529868\n",
      "  311.29073285   60.16096947  168.43776513 1129.2162931  1701.4185552\n",
      "   93.90066522  298.23976165 1910.89640392  237.93829371 1816.19736294\n",
      " 2057.73790503 1232.43851356  997.78853612  322.52744251  203.51994435\n",
      "  209.79662925  381.57737335  487.5995626   659.33843151 1050.08177567\n",
      "   15.73058912 1209.20491319  128.31895914   65.79790083  320.77933233\n",
      " 1163.73299851  213.72367302   68.92907116  218.88423577  103.92821912\n",
      "   16.22438396  236.09714463 1744.46835374  217.36919038   30.31850005\n",
      "   55.53398091 1402.64335087 1262.91657275  287.71758024   37.58320351\n",
      " 1792.16872249 1813.14715614   39.75000482  361.7201292  1903.98422832\n",
      " 1936.80741308 1751.86416212   29.36441595   67.51782922  716.1807451\n",
      " 1350.99084393  208.9848315  1681.39231082  196.10173327  310.90738695\n",
      "   54.78387991  179.38695329 1351.11539644  244.30940257 1755.41251788\n",
      " 1760.45588962  100.80512875  463.62700584 1161.32223106 1870.26656818\n",
      " 1495.27444142  174.67496664   57.90344184  725.81418713 1361.87197296\n",
      "   49.77198368  225.38928353  273.23807572  248.40479893 1242.49439539\n",
      " 1695.34972502  232.60651795  186.34798964  183.11427043   66.53731237\n",
      "  332.40242541  768.00005011 1004.71692154 1693.87627288  435.56821413\n",
      "  181.60170837 1879.75182098   32.41217836  216.83521247 1175.94132428\n",
      " 1002.57599538  180.86462516  117.32808808   17.39063379  242.39264154\n",
      " 1153.74419888   16.35425747 1446.7681195    57.79216068  695.21785675\n",
      "  247.79356401 1161.22598425   26.49236802 1165.83169818 1732.74376072\n",
      "   57.10624271  672.2187474   196.65007589 1597.48030694  153.46916721\n",
      " 1819.06854972   35.79872278  309.96121444 1672.4343611  1498.01292713\n",
      "  184.8598706  1947.49007014 1757.82444942 1472.71613392  189.78702998\n",
      "  159.25061206  166.03788065  185.67826914  126.23266479   55.43581844\n",
      "  641.68342661  328.8838791   766.56722642  522.72999523 1312.9668254\n",
      "  210.0055021  1157.35478596 1722.1834288   130.63325669 1923.84735351\n",
      " 1921.80886486   70.36989064 1608.32351793  655.37104142  312.11029527\n",
      "  241.6223466   175.07818374  294.40556962  163.19874431  190.03910324\n",
      "  176.36626133  663.35147292  146.5249705    31.35733434 1754.06035496\n",
      " 1640.39527491  338.16465038 1667.43572377   92.36994545  664.65579073\n",
      "  325.04209498   99.11073774   26.11274481 1743.54953718 1770.54363609\n",
      "  188.05996854 1357.68816338  230.73450327  507.47779385 1338.56530258\n",
      "  187.37338327 1037.42256157  228.90740281  183.20991999  601.21024827\n",
      "  243.20658711 1048.5232424  1801.28807696 1250.33028036  217.8242109\n",
      " 1657.68644947 1882.6822499   190.16034906 1221.87077576 1005.19400578\n",
      " 1812.62532665  183.46366869 1650.53287903  314.44924852  175.03696964\n",
      "  982.87520698 1154.45263125   94.55458481 1108.51666691  685.47322786\n",
      " 1935.90602431  186.02410101   14.66415806  162.63519785  246.64806563\n",
      "  661.82760457  443.66965426  245.96734348 1347.29286431 1284.3152611\n",
      " 1643.45230577   51.1082268    33.31450327  125.87258462 1051.64869477\n",
      "  229.34056959  286.73980791  232.16904999 1418.85777904  376.6000468\n",
      "  619.20275196   15.80248845  372.41754278  323.65612101   26.12440165\n",
      " 1759.81269618  232.48743671 1952.30403226   14.35495201   88.37823517\n",
      "  759.10364002  155.02238585 1167.55532504  781.81467184  288.0529541\n",
      "  180.59584008  210.88260811 1618.55424956  204.33951409  184.09290738\n",
      "  308.75330526 1727.29609817 1310.5822945  1762.20758416 1228.59340798\n",
      " 1719.11193962 1906.53442048 1788.55094689 1127.45638553   16.44193204\n",
      " 1407.82234416 1394.76485779 1808.08161037  228.06493183   77.58575203\n",
      "  128.62958636 1063.84911448  697.95525995  192.83426149  193.54735528\n",
      "  235.57060079  554.71865013 1837.9266064   199.68941759  361.95734745\n",
      " 1860.85342122  786.21455233  322.3227172   636.79900659  226.52580081\n",
      "  620.25383671  297.39162371 1777.26531901  214.02626681  117.49405312\n",
      "  108.33692971  184.43766979  210.21979773 1689.91057741  179.16888209\n",
      "  329.81529456  310.81385611 1967.11629277]\n"
     ]
    }
   ],
   "source": [
    "predicted=regressor.predict(X_test)\n",
    "print(predicted)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Actual</th>\n",
       "      <th>predicted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>143.447464</td>\n",
       "      <td>140.453495</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1724.530029</td>\n",
       "      <td>1714.795321</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1026.795166</td>\n",
       "      <td>1034.809831</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>620.255127</td>\n",
       "      <td>617.979361</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Actual    predicted\n",
       "0   143.447464   140.453495\n",
       "1  1724.530029  1714.795321\n",
       "2  1026.795166  1034.809831\n",
       "3   620.255127   617.979361"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataframe=pd.DataFrame({'Actual':y_test.flatten(),'predicted':predicted.flatten()})\n",
    "dataframe.head(4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error: 3.5190018937488303\n",
      "Mean squared Error: 33.123252323711746\n",
      "root mean squared Error: 5.755280386194207\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\n",
    "from sklearn import metrics\n",
    "# Printout relevant metrics\n",
    "#print(\"Model Coefficients:\", model.coef_)\n",
    "print(\"Mean Absolute Error:\",metrics.mean_absolute_error(y_test, predicted))\n",
    "print(\"Mean squared Error:\",metrics.mean_squared_error(y_test, predicted))\n",
    "print(\"root mean squared Error:\",np.sqrt(metrics.mean_squared_error(y_test, predicted)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjEAAAGhCAYAAACQ4eUqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAABC3ElEQVR4nO3de1xVZb7H8e+GvVG5CBgSaJKgonhBUFNTS9OazpijWWmZjRbisdHRnBnL5qiVpiWW2mR1xqy8UHnJwnK0yRq76VzMS2NliYrmDQ6XQETksmGfP3yxxi1gXPZG1vbzfr16xbrs5/c8a7nhy7PWXlgcDodDAAAAJuN1pTsAAABQF4QYAABgSoQYAABgSoQYAABgSoQYAABgSoQYAABgSoQYAABgSoQYAABgSoQYAABgSoQYAABgStYr3QF3y83Nld1ur/H+LVu2VFZWltv64+72qdG4anjCGKjReNqnRuOq4QljaIw1rFargoODa7ZvfTplBna7XaWlpTXa12KxGK9xx5+Ucnf71GhcNTxhDNRoPO1To3HV8IQxeEINLicBAABTIsQAAABTIsQAAABTIsQAAABT8vgbewEAnsNut6uwsLBG+54/f14lJSVu64u72/fkGr6+vrJa6x9BCDEAAFOw2+06d+6cAgIC5OX18xcSbDZbjT+dWhfubt9Ta5SXl+vs2bPy8/Ord5DhchIAwBQKCwtrHGDQeHl5eSkgIKDGM2qXbcsF/QEAoEEQYDyDq84j/xoAAIApEWIAAIApEWIAALhKtW7dWlu3br3S3agzPp0EADC1sonDq17vhlreKz6o82u/+uor3XXXXbr55pv11ltv1fh1ffr0UWJioiZOnFjn2p6KmRgAABrA+vXr9dBDD2nXrl06derUle6ORyDEAADgZoWFhdq8ebPGjRunW2+9VRs2bHDavm3bNv3yl79UVFSUunbtqsTEREnSnXfeqZMnT+qpp55S69at1bp1a0nS4sWLddtttzm1sWLFCvXp08dY/vrrr3Xfffepa9eu6tSpk+6++2598803bh5pwyLEAADgZh988IHatWun9u3b66677tL69evlcDgkSZ988okSExM1ZMgQffTRR1q/fr1iY2MlSStXrlR4eLhmzJihffv2ad++fTWuWVBQoFGjRiklJUWbN29WZGSkfv3rX6ugoMAtY7wSuCcGAK5S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RzR37lxjec2aNZKkgQMHasqUKRoxYoRKSkr02muv6dy5c2rfvr1mzZplfFJIksaPHy9vb28tXbrUeNjdzJkz5eX1n4mhadOm6Y033tCCBQskST179tSECRPqPFAAAOBZah1iunTpog0bNlS73WKxaPTo0Ro9enS1+/j4+CghIaHaB+ZJkr+/v6ZNm1bb7gEAgKsEf8UaAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYEiEGAACYktXVDU6ZMkVZWVmV1v/iF79QYmKiXn75ZX3++edO2zp06KAFCxYYy6WlpUpOTtbOnTtVUlKirl27KjExUddcc42ruwsAAEzK5SHm2WefVXl5ubF8/PhxzZ8/XzfeeKOxLi4uTpMnT/5PJ6zO3Vi1apX27NmjRx55RAEBAVqzZo0WLlyopKQkeXkxeQQAANxwOal58+YKCgoy/tu7d6+uvfZade7c2djHarU67ePv729sKyws1Pbt2zVu3DjFxsYqMjJSU6dO1fHjx7V//35XdxcAAJiUy2diLma32/Xll1/qjjvukMViMdYfOHBAiYmJ8vPzU0xMjMaMGaPAwEBJUlpamsrKyhQbG2vs36JFC0VERCg1NVVxcXHu7DIAADAJt4aYXbt26dy5cxo0aJCxLj4+XjfeeKNCQkKUmZmp9evXa968eVq4cKFsNpvy8vJktVqdZmckKTAwUHl5edXWKi0tVWlpqbFssVjUrFkz4+uaqNivpvu7mrvruqp9TzlO7h5HQxwnT6lRk/rU4Fw0phqe8t4zew23hphPP/1UcXFxatGihbGuX79+xtcRERFq166dJk+erL1796pPnz7VtuVwOC5bKyUlRRs3bjSWIyMjlZSUpJYtW9a632FhYbV+zaVO1OE14eHhbq1R2/Z/jiuOk+Q547hS7ZuthrvPd2N879WlxuWY5Vw0RA1PORfUqBu3hZisrCzt379fM2bMuOx+wcHBatmypdLT0yVJQUFBstvtKigocJqNyc/PV8eOHattZ+TIkRo2bJixXJH4srKyZLfba9Rni8WisLAwZWRk/GxocoeKY9DY2/eU4+TucTTEcfKUGpfj7veFmWpwLhpPDU957zXGGlartcYTEG4LMZ9++qkCAwPVo0ePy+539uxZ5eTkKDg4WJIUFRUlb29v7d+/35i1yc3N1fHjxzV27Nhq27HZbLLZbFVuq+2JcTgcV+QbhLtrurp9TzlO7h5HQxwnT6lRXV1qVG6Lc9E4anjKe8+sNdwSYsrLy/XZZ59p4MCB8vb2NtYXFRVpw4YN6tu3r4KCgpSVlaW1a9cqICBAvXv3liT5+vpq8ODBSk5OVkBAgPz9/ZWcnKyIiAinm30BAMDVzS0h5ptvvlF2drZuueUWp/VeXl46ceKEvvjiC507d07BwcHq0qWLpk+fbtyEK0njx4+Xt7e3li5dajzsbubMmTwjBgAAGNwSYrp3764NGzZUWu/j46NZs2b97Ot9fHyUkJCghIQEd3QPAAB4AKY2AACAKRFiAACAKRFiAACAKbn1YXfA1aJs4vCqN2zZ3bAdAYCrCCEGcKMbntte7bb3x3ZqwJ4AgOfhchIAADAlQgwAADAlQgwAADAlQgwAADAlQgwAADAlQgwAADAlQgwAADAlQgwAADAlQgwAADAlQgwAADAlQgwAADAlQgwAADAlQgwAADAl/oo1Lqts4vCqN2zZ3bAdAQDgEoQY1MkNz22vcv37Yzs1cE8AAFcrLicBAABTIsQAAABTIsQAAABTIsQAAABTIsQAAABTIsQAAABTIsQAAABTIsQAAABTIsQAAABTIsQAAABTIsQAAABTIsQAAABTIsQAAABTIsQAAABTIsQAAABTIsQAAABTIsQAAABTsrq6wQ0bNmjjxo1O6wIDA7VixQpJksPh0DvvvKO//e1vKigoUIcOHTRhwgS1adPG2L+0tFTJycnauXOnSkpK1LVrVyUmJuqaa65xdXcBAIBJuTzESFKbNm00Z84cY9nL6z8TPu+//762bNmiyZMnKzw8XO+9957mz5+vF154Qc2aNZMkrVq1Snv27NEjjzyigIAArVmzRgsXLlRSUpJTWwAA4OrllkTg5eWloKAg47/mzZtLujALs3XrVo0cOVJ9+vRRRESEpkyZouLiYu3YsUOSVFhYqO3bt2vcuHGKjY1VZGSkpk6dquPHj2v//v3u6C4AADAht8zEZGRkaNKkSbJarerQoYPGjBmja6+9VpmZmcrLy1P37t2NfW02mzp37qyDBw/qtttuU1pamsrKyhQbG2vs06JFC0VERCg1NVVxcXFV1iwtLVVpaamxbLFYjJkdi8VSo35X7FfT/V3N3XUbYlxmquEJ57shxuAJx8lTanAuGk8NT3nvmb2Gy0NMhw4dNGXKFLVq1Up5eXl67733NHv2bC1ZskR5eXmSLtwjc7HAwEBlZ2dLkvLy8mS1WuXv719pn4rXVyUlJcXpXpzIyEglJSWpZcuWtR5DWFhYrV9zqRN1eE14eLhba9S2fU+qcTlmOd+X44oxNFQNd5/vxvjeq0uNyzHLuWiIGp5yLqhRNy4PMfHx8cbXERERio6O1tSpU/X555+rQ4cOkiqnMYfD8bPt/tw+I0eO1LBhw4zlihpZWVmy2+016rvFYlFYWJgyMjJq1CdXS09PN3X7ZqvhCee7IcbgCcfJU2pwLhpPDU957zXGGlartcYTEG65nHSxpk2bKiIiQunp6brhhhskXZhtCQ4ONvbJz883ZmeCgoJkt9tVUFDgNBuTn5+vjh07VlvHZrPJZrNVua22J8bhcFyRbxDurtkQYzJjDU843w0xBk84Tp5Sg3PReGp4ynvPrDXc/lGf0tJSnTp1SsHBwQoNDVVQUJDTDbp2u10HDhwwAkpUVJS8vb2d9snNzdXx48cVHR3t7u4CAACTcPlMzJo1a9SrVy+FhITozJkzevfdd3X+/HkNHDhQFotFQ4cOVUpKisLDwxUWFqaUlBQ1adJEAwYMkCT5+vpq8ODBSk5OVkBAgPz9/ZWcnKyIiAinm30BAMDVzeUh5qefftKf/vQn5efnq3nz5urQoYMWLFhgXN8aMWKESkpK9Nprr+ncuXNq3769Zs2aZXySSJLGjx8vb29vLV261HjY3cyZM3lGDAAAMLg8xEyfPv2y2y0Wi0aPHq3Ro0dXu4+Pj48SEhKUkJDg4t4BAABPwdQGAAAwJUIMAAAwJUIMAAAwJUIMAAAwJbc/7A4AAFx5ZROHV7n+hkGLqlz//thO7uyOSzATAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATIkQAwAATMnq6gZTUlK0a9cunTp1Sj4+PoqOjtYDDzygVq1aGfu8/PLL+vzzz51e16FDBy1YsMBYLi0tVXJysnbu3KmSkhJ17dpViYmJuuaaa1zdZQAAYEIuDzEHDhzQ7bffrnbt2qmsrEzr1q3T/PnztWTJEjVt2tTYLy4uTpMnT/5PR6zOXVm1apX27NmjRx55RAEBAVqzZo0WLlyopKQkeXkxgQQAwNXO5Wlg1qxZGjRokNq0aaO2bdtq8uTJys7OVlpamtN+VqtVQUFBxn/+/v7GtsLCQm3fvl3jxo1TbGysIiMjNXXqVB0/flz79+93dZcBAIAJuXwm5lKFhYWS5BRSpAszNomJifLz81NMTIzGjBmjwMBASVJaWprKysoUGxtr7N+iRQtFREQoNTVVcXFxleqUlpaqtLTUWLZYLGrWrJnxdU1U7FfT/V3N3XUbYlxmquEJ57shxuAJx8lTanAuGk8Ni8WiG57bXu32Dx6IcUmNi//f0MzwvdatIcbhcGj16tXq1KmTIiIijPXx8fG68cYbFRISoszMTK1fv17z5s3TwoULZbPZlJeXJ6vVWin4BAYGKi8vr8paKSkp2rhxo7EcGRmppKQktWzZstb9DgsLq/VrLnWiDq8JDw93a43atu9JNS7HLOf7clwxhoaq4e7z3Rjfe3WpcTlmORcNUaNBzvcdvareMGiRy2pcjpnO9+W44/uUW0PM66+/ruPHj2vevHlO6/v162d8HRERoXbt2mny5Mnau3ev+vTpU217Doej2m0jR47UsGHDjOWKxJeVlSW73V6j/losFoWFhSkjI+OytdwlPT3d1O2brYYnnO+GGIMnHCdPqcG5uLpqeMr5ru04rFZrjScg3BZi3njjDe3Zs0dz58792U8UBQcHq2XLlsYBCwoKkt1uV0FBgdNsTH5+vjp27FhlGzabTTabrcpttT35DofjivyDcXfNhhiTGWt4wvluiDF4wnHylBqci6urhqecb3eMw+U39jocDr3++uv617/+pSeeeEKhoaE/+5qzZ88qJydHwcHBkqSoqCh5e3s73cSbm5ur48ePKzo62tVdBgAAJuTymZjXX39dO3bs0GOPPaZmzZoZ97D4+vrKx8dHRUVF2rBhg/r27augoCBlZWVp7dq1CggIUO/evY19Bw8erOTkZAUEBMjf31/JycmKiIhwutkXAABcvVweYrZt2yZJeuqpp5zWT548WYMGDZKXl5dOnDihL774QufOnVNwcLC6dOmi6dOnG58mkqTx48fL29tbS5cuNR52N3PmTJ4RAwAAJLkhxGzYsOGy2318fDRr1qyfbcfHx0cJCQlKSEhwVdcAAIAHYVoDAACYEiEGAACYEiEGAACYktv/7AAAwHWqe9T9+2M7NXBPgCuPEAOYRNnE4VWuv6GaR5/zQ83cqjvfl3vUPXC1IcQAAHCF8UtK3XBPDAAAMCVCDAAAMCVCDAAAMCVCDAAAMCVCDAAAMCVCDAAAMCVCDAAAMKWr8jkx1T5ESpK27G64jgAAgDq7KkPM5fBIbwAAzIHLSQAAwJQIMQAAwJQIMQAAwJQIMQAAwJQIMQAAwJQIMQAAwJQIMQAAwJQIMQAAwJQIMQAAwJQIMQAAwJQIMQAAwJQIMQAAwJQIMQAAwJQIMQAAwJSsV7oDABqPsonDq1x/w6BFVa5/f2wnd3YHAC6LmRgAAGBKhBgAAGBKhBgAAGBKhBgAAGBKhBgAAGBKhBgAAGBKjf4j1h999JE++OAD5eXl6brrrtODDz6omJiYK90tAABwiYZ+TEOjDjF///vftWrVKiUmJqpjx4765JNP9Mwzz2jp0qUKCQm50t2DSVT3ppJ4/gkAmFmjvpz0l7/8RYMHD9aQIUOMWZiQkBBt27btSncNAABcYY02xNjtdqWlpal79+5O62NjY3Xw4MEr1CsAANBYNNrLSfn5+SovL1dgYKDT+sDAQOXl5VXav7S0VKWlpcayxWJRs2bNZLVWHqJXu47V1u0Y6l/lepvNVsOeX75Gde03RI3atu8pNTjfV1cNzvfVVYPz7Xk1qvq5XR2Lw+Fw1HjvBvTTTz/p4Ycf1vz58xUdHW2sf++99/TFF1/ohRdecNp/w4YN2rhxo7Hcv39/PfLIIw3VXQAA0MAa7eWk5s2by8vLq9Ksy5kzZyrNzkjSyJEjtWrVKuO/iRMnOs3M1MT58+c1c+ZMnT9/vj5dv2LtU6Nx1fCEMVCj8bRPjcZVwxPG4Ak1Gm2IsVqtioqK0v79+53W79+/Xx07Vp6ustls8vX1dfqvtlNhDodDR48elbsmp9zdPjUaVw1PGAM1Gk/71GhcNTxhDJ5Qo9HeEyNJw4YN07JlyxQVFaXo6Gh98sknys7O1m233XaluwYAAK6wRh1i+vXrp7Nnz+rdd99Vbm6u2rRpoz/+8Y9q2bLlle4aAAC4whp1iJGk22+/XbfffnuD1LLZbLrnnnvqdEd2Y2ifGo2rhieMgRqNp31qNK4anjAGT6jRaD+dBAAAcDmN9sZeAACAyyHEAAAAUyLEAAAAUyLEoNHidi0AwOU0+k8nuUtOTo62bdum1NRU46nAQUFBio6O1m233aaQkJAr20Ho/vvv13PPPafrrrvuSncFLpabm6tt27bphx9+UF5enry8vNSyZUv17t1bgwYNkpcXv18B+HlX5aeTfvjhBz3zzDO65ppr1L17dwUGBsrhcCg/P1/79+9XTk6O/vjHP6pTp05u7Ud2drY2bNigyZMn17mNkpISpaWlyd/fv9IP+5KSEv3jH//QwIED69XPkydP6tChQ4qOjlbr1q116tQpbd26VaWlpbr55pvVtWvXerW/evXqKtdv3bpVN910kwICAiRJ48ePr1edixUUFOjzzz9Xenq6goODNXDgwHoF16NHj8rPz0+hoaGSpC+++EIff/yxsrOzFRISov/6r/9S//7969XnN954QzfeeKNiYmLq1c7P+fDDD3XkyBH16NFD/fr10xdffKGUlBQ5HA717t1b9957r7y9vevc/pEjR/T0008rNDRUPj4+OnTokG666SbZ7Xb9+9//VuvWrTVr1iw1a9bMhaMC4ImuypmY1atXa/DgwXrwwQer3L5q1SqtXr1azz77rFv7UfGDtK4h5vTp01qwYIGys7MlSTExMXrkkUcUHBwsSSosLNQrr7xSrxDz9ddfa9GiRWratKmKi4v16KOP6qWXXtL1118vSVqwYIFmzZpVryCzdetWXX/99fLz86u07dSpU2ratGmd264wadIkPf/88woICFBmZqZmz54tSYqIiNCePXu0efNmLViwQK1bt65T+3/+85/161//WqGhofrb3/6mlStXasiQIbrpppt0+vRpLV++XMXFxRo8eHCdx/DRRx/po48+UlhYmG655RYNGjRIQUFBdW6vKhs3btTmzZsVGxurVatWKTMzU5s3b9Ydd9whi8WiLVu2yGq1avTo0XWusWrVKt1xxx0aNWqUpAuB76OPPtKCBQtUUFCgefPmad26dXrooYfqNZaioiLt2LHDmG21WCwKDAxUx44d1b9/f5f8u7qcvLw8ffLJJ7rnnnvq3VZOTo78/Pwq9dlutys1NVWdO3euV/tnz57Vjz/+qLZt28rf31/5+fnavn277Ha7+vbt65bZ0N/+9reaNWuWwsPDXd623W7X3r17lZGRoaCgIPXu3bve5zsnJ0c2m03NmzeXJH3//feVflG5+I8V18XmzZvVt29ftz/Qdffu3UpLS1NcXJyio6P17bffavPmzSovL1efPn1066231qv9kpIS7dixw2mmNTQ0VDfccIO6devmolFccFWGmOPHj2vq1KnVbr/tttv08ccf17vO7t27L7v9//7v/+rV/ltvvaU2bdro2WefVWFhoVavXq05c+boqaeectnlsI0bN2r48OG67777tHPnTv3pT3/SL37xC40ZM0aStHbtWm3atKleIea+++7T3/72N40bN86pnTFjxmjKlCku+Qaal5en8vJySdLbb7+t1q1b6/HHH1eTJk1UWlqqxYsXa/369fr9739fp/ZPnz6tsLAwSdK2bdv04IMPOn0jaN++vd577716hRhJmj17thG61q9fr/j4eA0ZMkTx8fEuuQRTEar79OmjY8eO6fHHH9eUKVN00003SZJat26tN998s14h5ujRo/rtb39rLA8YMED/+7//q7y8PAUFBemBBx7Qyy+/XK8Qc/LkST399NMqKSlRTEyMQkJC5HA4dObMGb355pt65513NHv2bLdeqszLy9M777xTrxCTm5urRYsWKS0tTRaLRQMGDFBiYqLxA7mgoEBz587V+vXr61zj8OHDmj9/vs6fPy9fX1/NmTNHS5Yskbe3txwOhzZt2qR58+YpKiqqTu1v3bq1yvXZ2dn69NNPjSA+dOjQug5Bs2fP1h//+Ef5+fkpPz9f8+bN0+nTp9WyZUtlZ2dr3bp1mj9/vlq0aFHnGkuXLtXdd9+t+Ph4ffXVV3r++efVs2dPdezYUenp6XryySc1Y8YM9ezZs8413nzzTb311lvq0qWLhgwZot69e8tqde2P6W3btmnlypW6/vrrtWXLFiUmJuq1117TjTfeKC8vL61atUolJSV1Ph8ZGRl6+umnVVRUJKvVqry8PMXHx+vIkSPatm2bevfurUceeaRes7kXuypDTHBwsA4ePKhWrVpVuT01NdWYzaiP5557rt5tXE5qaqrmzJmj5s2bq3nz5po5c6Zee+01PfHEE3ryySfVpEmTetc4ceKE8QPnxhtv1EsvvaQ+ffoY2/v376/t27fXq8bIkSPVrVs3LVu2TD179tT999/v8jfuxQ4fPqyHH37YOD42m0133323lixZUuc2fXx8lJ+fr5CQEP30009q37690/b27dsrMzOzXv2WLswcdevWTQ888IB27dqlTz/9VM8995wCAwM1aNAg3XLLLUaYqovc3Fy1a9dOktS2bVtZLBa1bdvW2B4ZGanc3Nx6jSEwMFC5ubm69tprJV34y/Tl5eXy9fWVJIWFhamgoKBeNV5//XXFxMTot7/9baV/S3a7XS+//LJef/11Pfnkk3Wu8eOPP152++nTp+vcdoW33npLXl5eeuaZZ3Tu3DmtXbtWTz31lGbPni1/f/96ty9d+EWkb9++Gj9+vD7++GM999xz6t69ux5++GFJF2YZ3333XT366KN1an/16tVq0aJFpZDtcDj0xRdfyNvbWxaLpV4h5tChQ7Lb7cZ4vLy89MorrygoKEhnz57VokWLtH79ev3mN7+pc40TJ04YM7WbNm3SmDFjdOeddxrb//rXv2rDhg31CjGS9PDDD2vXrl1atmyZfH19ddNNN2nw4MGKiIioV7sVPvzwQ02YMEG33nqrvv32Wz377LMaN26c8WT86Ohovf/++3U+HytXrlT37t2VmJgoLy8vbdq0Sd9//70WLFig9PR0zZ8/X++++269fhG62FUZYn71q19pxYoVSktLU2xsrAIDA2WxWJSXl6f9+/dr+/btLrn/IigoSBMmTFDv3r2r3H7s2DHNnDmzzu2XlJRU+saQmJioN954Q0899ZSmTZtW57ar4uXlJZvN5nTZp1mzZiosLKx32+3bt1dSUpJee+01/fGPf7zsTFldWSwWSVJpaakCAwOdtgUGBio/P7/ObcfFxWnbtm16+OGHFRMTo3/+859OP/z/8Y9/1CtcXMpqtapfv37q16+fsrOztX37dn3++efatGlTvX4rDwoK0smTJxUSEqL09HSVl5fr5MmTatOmjaQL38grptPr6oYbbtCKFSv061//WlarVe+++646d+4sHx8fSRd++NfnN2bp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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "graph=dataframe.head(20)\n",
    "graph.plot(kind=\"bar\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Open</th>\n",
       "      <th>High</th>\n",
       "      <th>Low</th>\n",
       "      <th>Close</th>\n",
       "      <th>Adj Close</th>\n",
       "      <th>Volume</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>4938</th>\n",
       "      <td>1649.699951</td>\n",
       "      <td>1654.0</td>\n",
       "      <td>1637.050049</td>\n",
       "      <td>1650.800049</td>\n",
       "      <td>1650.800049</td>\n",
       "      <td>5260900</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             Open    High          Low        Close    Adj Close   Volume\n",
       "4938  1649.699951  1654.0  1637.050049  1650.800049  1650.800049  5260900"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "actual_price=data.tail(1)\n",
    "actual_price"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Price pridtion using SVR Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Close</th>\n",
       "      <th>HIGHLOW_PCT</th>\n",
       "      <th>PCT_Change</th>\n",
       "      <th>Volume</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>18.872499</td>\n",
       "      <td>1.152474</td>\n",
       "      <td>7.842854</td>\n",
       "      <td>2057200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>19.110001</td>\n",
       "      <td>5.180532</td>\n",
       "      <td>-0.727270</td>\n",
       "      <td>1573960</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>18.580000</td>\n",
       "      <td>10.306786</td>\n",
       "      <td>-5.204084</td>\n",
       "      <td>10822400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>18.480000</td>\n",
       "      <td>4.031390</td>\n",
       "      <td>-3.637076</td>\n",
       "      <td>6484440</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>17.557501</td>\n",
       "      <td>6.222405</td>\n",
       "      <td>-5.350396</td>\n",
       "      <td>2814060</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Close  HIGHLOW_PCT  PCT_Change    Volume\n",
       "0  18.872499     1.152474    7.842854   2057200\n",
       "1  19.110001     5.180532   -0.727270   1573960\n",
       "2  18.580000    10.306786   -5.204084  10822400\n",
       "3  18.480000     4.031390   -3.637076   6484440\n",
       "4  17.557501     6.222405   -5.350396   2814060"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#here high (highest stock price of the day) and close(the stock price at the end of the day)\n",
    "#Calculating percent volatility\n",
    "data['HIGHLOW_PCT']=(data['High']-data['Close'])/(data['Close'])*100\n",
    "#Calculating new and old prices\n",
    "data['PCT_Change']=(data['Close']-data['Open'])/(data['Open'])*100\n",
    "# Extracting required data from file\n",
    "data=data[['Close','HIGHLOW_PCT','PCT_Change','Volume']]\n",
    "display(data.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "50\n",
      "       Close  HIGHLOW_PCT  PCT_Change    Volume      label\n",
      "0  18.872499     1.152474    7.842854   2057200  17.750000\n",
      "1  19.110001     5.180532   -0.727270   1573960  18.325001\n",
      "2  18.580000    10.306786   -5.204084  10822400  18.655001\n",
      "3  18.480000     4.031390   -3.637076   6484440  18.532499\n",
      "4  17.557501     6.222405   -5.350396   2814060  18.937500\n"
     ]
    }
   ],
   "source": [
    "#forecast volume to calculate future stocks\n",
    "forecast_col='Close'\n",
    "#We have to replace to na data with negative 99999.It will be useful when we lacking with data\n",
    "data.fillna(-99999,inplace=True)\n",
    "# if the length of data frame is returning decimal point or float it will round up to integer\n",
    "# 0.1 means tomorrow data ,we can change accordingly\n",
    "forecast_out=int(math.ceil(0.01*len(data)))\n",
    "print (forecast_out)\n",
    "data['label']=data[forecast_col].shift(-forecast_out)\n",
    "print (data.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Setting up features and labels,X->feature and y->label\n",
    "#drop useless features\n",
    "#This code will return new data frame and convereted to \n",
    "#X_lately is the one we are predict against\n",
    "X = np.array(data.drop('label', axis=1))\n",
    "X = preprocessing.scale(X)\n",
    "X = X[:-forecast_out]\n",
    "X_lately=X[-forecast_out:]\n",
    "\n",
    "\n",
    "data.dropna(inplace=True)\n",
    "y=np.array(data['label'])\n",
    "#print (len(X),len(y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Train and test data set\n",
    "#Test size=0.2 means we are using 20% data as a testing data\n",
    "#cross validation will take features and lables data and shuffle them and give X_train,y_train,X_test and y_test\n",
    "X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8154464542183053\n"
     ]
    }
   ],
   "source": [
    "#classification inorder to get X_train and Y_train\n",
    "#if i use Support vector machine i'm getting 80% Accuracy and  Linear regression i'm getting 75% Accuracy\n",
    "#clf=LinearRegression(n_jobs=-1) =>75% Accuracy\n",
    "clf=svm.SVR()\n",
    "clf.fit(X_train,y_train)\n",
    "accuracy=clf.score(X_test,y_test)\n",
    "print (accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1412.48819802 1420.45791727 1425.23961244 1321.27352016 1312.8359379\n",
      " 1433.47233028 1370.35551565 1408.6870981  1347.41420452 1415.62615467\n",
      " 1383.96914016 1185.67939491 1400.16779182 1417.3202665  1151.65427971\n",
      "  791.60670554 1381.28814469 1234.08618705 1418.68601644 1410.39214494\n",
      " 1269.52508702 1400.69251667  790.68282178 1413.22724389 1314.22972224\n",
      " 1179.57511876 1441.36257907 1220.11512578 1235.94978687 1433.33537595\n",
      " 1135.50417321 1412.2054687  1347.62738175 1417.60684767 1332.72217723\n",
      " 1408.61084826  823.35152539 1390.37781035 1233.5726499  1334.81439658\n",
      " 1357.26590568 1414.90975191 1392.15243655 1396.31341478 1136.19283012\n",
      " 1372.31767617 1276.35637152 1426.74026915 1304.02350048 1395.83610457] 0.8154464542183053 50\n"
     ]
    }
   ],
   "source": [
    "#We can pass single value or array of values or we are passing 99days of value\n",
    "forecast_set=clf.predict(X_lately)\n",
    "print (forecast_set,accuracy,forecast_out)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
