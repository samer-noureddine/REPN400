# REPN400
Code for the paper:

Wang L., Nour Eddine, S., Brothers T., Jensen O., Kuperberg G., _Predictive coding explains the dynamics of neural activity within the left ventromedial temporal lobe during reading comprehension._

We use model simulations and experimental data to argue that language-related brain activity reflects two distinct kinds of information - predictions and prediction errors.

---

## System Requirements

- This code has been tested on Python 3.9.8 in Windows 11. The code took roughly 4 hours to run on PC with i7-8700 CPU @ 3.20GHz, 3192 Mhz, 6 Cores, 12 Logical Processors, 16 GB RAM.

### Software Dependencies:
- Python 3.9
- NumPy 1.26.2
- Pandas 1.4.3
- Matplotlib 3.5.2
- SciPy 1.11.4
- Scikit-learn 1.1.2

## Installation Guide
1. **Install Python**: Ensure that Python 3.9 or higher is installed on your system. You can download it from [the official Python website](https://www.python.org/downloads/).
2. **Install Dependencies**: Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

## Reproduce our simulations
To reproduce our simulations, navigate to the directory where you have downloaded the files and simply run the following line:
   ```
   python -m REPN400_simulations.py
   ```
   This will run the simulations and reproduce Figures 1B, 2C, 2D, 3C, S1. It will create two new folders, `./data/` and `./plots/` where it will save the simulation data files and the plot image files respectively.

## Support
For any issues or questions, please open an issue on the GitHub repository or contact the maintainers at [samer.a.noureddine@gmail.com](mailto:samer.a.noureddine@gmail.com).

---

**Note**: Replace placeholders like GitHub links, email, and specific software details with the actual data relevant to your software. Ensure that all instructions are clear and accurate according to your software's setup and requirements.
