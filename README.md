# The Transmission of Foreign Demand Shocks and Challenges for Stabilization Policies

This respository contains code and data for the empirical and theorical models in the paper "The Transmission of Foreign Demand Shocks and Challenges for Stabilization Policies" by [Jeppe Druedahl](https://sites.google.com/view/jeppe-druedahl/), [Søren Hove Ravn](https://sites.google.com/site/sorenhoveravn/), [Laura Sunder-Plassmann](https://sites.google.com/site/lsunderplassmann/), [Jacob Marott Sundram](https://sundram.dk/) and [Nicolai Waldstrøm](https://nwaldstrom.github.io/).

## Empirical analysis

All figures in section 2 and the empirical appendix of the paper (except the SVAR analysis) are produced simply by running **Empirical\RUN_MAIN.ipynb**.

To produce the SVAR figures, MATLAB is required. Here, you should start by running the cells up until and including "Save data for MATLAB" in **RUN_MAIN.ipynb**. Then, you should run the MATLAB file **Empirical\SVAR\RUN_MAIN.m**. Finally, you should run the remaining cells in **Empirical\RUN_MAIN.ipynb**.

## Theoretical model

Running the HANK model in the theorical part of the paper requries the following non-standard packages:
1. [EconModel](https://github.com/NumEconCopenhagen/EconModel)
2. [ConSav](https://github.com/NumEconCopenhagen/ConsumptionSaving)
3. [GEModelTools](https://github.com/NumEconCopenhagen/GEModelTools)

The figures used in chapter 6 of the paper are produced in the notebook **1. ForeignEconShocks.ipynb** while the tables for chapter 7 are produced in **2. StabilizationPolicy.ipynb**, both contained in the **model** folder. 

