# REPN400
Simulation code for the paper:

Wang L., Nour Eddine, S., Brothers T., Jensen O., Kuperberg G., _Predictive coding explains the dynamics of neural activity within the left ventromedial temporal lobe during reading comprehension._

We use model simulations and experimental data to show that predictive coding can explain both univariate and multivariate neural activity produced by expected or unexpected words during reading comprehension.

---

## System Requirements

- This code has been tested on Python 3.9.8 in Windows 11. The code took roughly 20 minutes to run on a PC with i7-8700 CPU @ 3.20GHz, 3192 Mhz, 6 Cores, 12 Logical Processors, 16 GB RAM. 

### Software Dependencies:
- Python 3.9
- NumPy 1.26.2
- Pandas 1.4.3
- Matplotlib 3.5.2
- SciPy 1.11.4
- Scikit-learn 1.1.2

## Installation Guide
1. **Install Python**: Ensure that Python 3.9 or higher is installed on your system. You can download it from [the official Python website](https://www.python.org/downloads/).
2. **Install Dependencies**: Install the required Python packages (install time is ~5-10 minutes on a normal desktop):
   ```
   pip install -r requirements.txt
   ```

## Reproducing simulations
To reproduce the simulations, navigate to the directory where you have downloaded the files and simply run the following line:
   ```
   python REPN400_simulations.py
   ```
   This will run the simulations and reproduce Figures 1B, 2C, 2D, 3C, S1 and S2. It will create two new folders, `./data/` and `./plots/` where it will save the simulation data files and the plot image files respectively. The data and plots have been pre-computed and uploaded to [OSF](https://osf.io/7cn2f/?view_only=ab6f8fc62dbe4487a0b9c106e9658408).
## Expected outputs
If the code runs smoothly, the following plots should be produced:
![Fig1B](https://github.com/samer-noureddine/REPN400/blob/main/precomputed_plots/Fig1B_univariate_summed_lexsem_diff.png?raw=true)
![Fig2C](https://github.com/samer-noureddine/REPN400/blob/main/precomputed_plots/Fig2C_separatewindows_post_exp_lexsem.png?raw=true)
![Fig2C](https://github.com/samer-noureddine/REPN400/blob/main/precomputed_plots/Fig2C_separatewindows_post_unexp_lexsem.png?raw=true)
![Fig2C](https://github.com/samer-noureddine/REPN400/blob/main/precomputed_plots/Fig2C_separatewindows_pre_lexsem.png?raw=true)
![Fig2D](https://github.com/samer-noureddine/REPN400/blob/main/precomputed_plots/Fig2D_tempRSA_boxplot_lexsem_fullspace.png?raw=true)
![Fig3C](https://github.com/samer-noureddine/REPN400/blob/main/precomputed_plots/Fig3C_spatRSA_summed_lexsem_dif.png?raw=true)
![FigS1](https://github.com/samer-noureddine/REPN400/blob/main/precomputed_plots/FigS1_univariate_summed_lexsem_raw.png?raw=true)
![FigS2](https://github.com/samer-noureddine/REPN400/blob/main/precomputed_plots/FigS2_spatRSA_summed_lex_and_sem_dif.png?raw=true)


## Support
For assistance or inquiries, please contact [samer.a.noureddine@gmail.com](mailto:samer.a.noureddine@gmail.com).
