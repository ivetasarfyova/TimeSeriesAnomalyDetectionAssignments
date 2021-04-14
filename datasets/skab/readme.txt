Datasets description - https://github.com/waico/SKAB 

Datasets
The SKAB v0.9 corpus contains 35 individual data files in .csv format. Each file represents a single experiment and contains a single anomaly. The dataset represents a multivariate time series collected from the sensors installed on the testbed. The data folder contains datasets from the benchmark. The structure of the data folder is presented in the structure file. Columns in each data file are following:

datetime - Represents dates and times of the moment when the value is written to the database (YYYY-MM-DD hh:mm:ss)
Accelerometer1RMS - Shows a vibration acceleration (Amount of g units)
Accelerometer2RMS - Shows a vibration acceleration (Amount of g units)
Current - Shows the amperage on the electric motor (Ampere)
Pressure - Represents the pressure in the loop after the water pump (Bar)
Temperature - Shows the temperature of the engine body (The degree Celsius)
Thermocouple - Represents the temperature of the fluid in the circulation loop (The degree Celsius)
Voltage - Shows the voltage on the electric motor (Volt)
RateRMS - Represents the circulation flow rate of the fluid inside the loop (Liter per minute)
anomaly - Shows if the point is anomalous (0 or 1)
changepoint - Shows if the point is a changepoint for collective anomalies (0 or 1)
Exploratory Data Analysis (EDA) for SKAB is presented at kaggle (Russian comments included, English version is upcoming).

anomaly-free
L¦¦ anomaly-free.csv     # Data obtained from the experiments with normal mode
valve2 
+¦¦ 1.csv
+¦¦ 2.csv
+¦¦ 3.csv
L¦¦ 4.csv
valve1             
+¦¦ 1.csv
+¦¦ 2.csv
+¦¦ 3.csv
+¦¦ 4.csv
+¦¦ 5.csv
+¦¦ 6.csv
+¦¦ 7.csv
+¦¦ 8.csv
+¦¦ 9.csv
+¦¦ 10.csv
+¦¦ 11.csv
+¦¦ 12.csv
+¦¦ 12.csv
+¦¦ 13.csv
+¦¦ 14.csv
+¦¦ 15.csv
L¦¦ 16.csv
other                   # Data obtained from the other experiments
+¦¦ 13.csv              # Sharply behavior of rotor imbalance
+¦¦ 14.csv              # Linear behavior of rotor imbalance
+¦¦ 15.csv              # Step behavior of rotor imbalance
+¦¦ 16.csv              # Dirac delta function behavior of rotor imbalance
+¦¦ 17.csv              # Exponential behavior of rotor imbalance
+¦¦ 18.csv              # The slow increase in the amount of water in the circuit
+¦¦ 19.csv              # The sudden increase in the amount of water in the circuit
+¦¦ 20.csv              # Draining water from the tank until cavitation
+¦¦ 21.csv              # Two-phase flow supply to the pump inlet (cavitation)
L¦¦ 22.csv              # Water supply of increased temperature