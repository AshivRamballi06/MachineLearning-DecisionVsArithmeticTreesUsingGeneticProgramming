========================================================================
COS314 - ASSIGNMENT 3: GENETIC PROGRAMMING CLASSIFIER
========================================================================

This archive contains the executable JAR file for Assignment 3, 
comparing Decision Tree and Arithmetic Genetic Programming algorithms 
for Breast Cancer classification.

------------------------------------------------------------------------
1. SETUP INSTRUCTIONS (CRITICAL) FOR REXREATING REPORT RESULTS
------------------------------------------------------------------------
To ensure the program successfully reads the dataset files, you must 
run it from the Command Prompt / Terminal in the exact folder where 
the files are located.

1. Extract "Assignment3.jar", "Breast_train.csv", and "Breast_test.csv" 
   into the SAME folder (e.g., a new folder on your Desktop).
2. Open your Command Prompt (Windows) or Terminal (Mac/Linux).
3. Navigate to that folder using the 'cd' command:
   Example: cd Desktop\COS314_Assignment3
4. Run the program using the following command:
   java -jar "Assignment 3.jar"

------------------------------------------------------------------------
2. RUNNING THE DECISION TREE GP
------------------------------------------------------------------------
When prompted by the main menu, select option [1] or [3].
The program will ask for parameters. You can press ENTER to use the 
defaults, or type the following recommended parameters:

  Training file         : Breast_train.csv
  Test file             : Breast_test.csv
  Seed                  : 56982693090
  Generations           : 100
  Population size       : 200
  Initial tree depth    : 7(it automatically starts from 2)
  Tournament size       : 7
  Max tree depth        : 5
  Crossover rate %      : 85
  Mutation rate %       : 15

------------------------------------------------------------------------
3. RUNNING THE ARITHMETIC GP
------------------------------------------------------------------------
When prompted by the main menu, select option [2] or [3].
Use the following specific parameters for the Arithmetic GP evaluation:

  Training file         : Breast_train.csv
  Test file             : Breast_test.csv
  Seed                  : 500
  Number of runs        : 1
  Initial tree depth    : 3
  Tournament size       : 15
  Max offspring depth   : 5
  Crossover rate %      : 85
  Mutation rate %       : 15
  Mutation offspring dep: 7

------------------------------------------------------------------------
4. TROUBLESHOOTING
------------------------------------------------------------------------
- "Cannot read training file": This means you did not use the 'cd' 
  command to navigate into the folder containing the CSVs before running 
  the java command. Ensure both the JAR and the CSVs are side-by-side.
- "No test file found": Ensure Breast_test.csv is in the folder. If 
  not, the program will simply skip test evaluation and only show 
  training results.
========================================================================
