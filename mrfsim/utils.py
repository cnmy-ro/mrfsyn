import random
import numpy as np



class PerlinNoise:
    """
    Based on: https://github.com/alexandr-gnrk/perlin-1d/blob/main/perlin_noise.py
    """
    def __init__(self, 
            seed, amplitude=1, frequency=1, 
            octaves=1):
        self.seed = random.Random(seed).random()
        self.amplitude = amplitude
        self.frequency = frequency
        self.octaves = octaves
        self.mem_x = dict()

    def __noise(self, x):
        # made for improve performance
        if x not in self.mem_x:
            self.mem_x[x] = random.Random(self.seed + x).uniform(-1, 1)
        return self.mem_x[x]

    def __interpolated_noise(self, x):
        prev_x = int(x) # previous integer
        next_x = prev_x + 1 # next integer
        frac_x = x - prev_x # fractional of x
        res = self.__cubic_interp(
            self.__noise(prev_x - 1), 
            self.__noise(prev_x), 
            self.__noise(next_x),
            self.__noise(next_x + 1),
            frac_x)
        return res

    def get(self, x):
        result = self.__interpolated_noise(x * self.frequency) * self.amplitude
        return result

    def __cubic_interp(self, v0, v1, v2, v3, x):
        p = (v3 - v2) - (v0 - v1)
        q = (v0 - v1) - p
        r = v2 - v0
        s = v1
        return p * x**3 + q * x**2 + r * x + s


def jiang_random_alphas():
    num_reps = 1000
    Nrf = 200
    fa_pattern = [] # Sinusoidal pattern
    for _ in range(num_reps // Nrf):
        alpha_max = np.random.randint(5, 91) / 180 * np.pi
        for n in range(Nrf): fa_pattern.append(np.sin(n * np.pi / Nrf) * alpha_max)
    return fa_pattern


def jiang_random_trs():
    num_reps = 1000
    pnoise = PerlinNoise(0, amplitude=1, frequency=0.025)
    tr_pattern = [(pnoise.get(i)+1)/2 * (14.5-11.5) + 11.5 for i in range(num_reps)]
    return tr_pattern


def sample_t1t2_parameter_space_with_const_rel_grid(t1_range=[10, 5000], t2_range=[6, 3000], t1_rel_step=0.01, t2_rel_step=0.01, t1_min_abs_step=1, t2_min_abs_step=1):
    t1_values = constant_rel_step(t1_range[0], t1_range[1], t1_min_abs_step, t1_rel_step)
    t2_values = constant_rel_step(t2_range[0], t2_range[1], t2_min_abs_step, t2_rel_step)
    parameter_values = {'t1': t1_values, 't2': t2_values}
    param_list = generate_parameter_combinations_table(parameter_values)
    return param_list


def sample_t1t2_parameter_space_with_const_abs_grid(t1_range=[10, 5000], t2_range=[6, 3000], t1_step=5, t2_step=5):
    t1_values = np.arange(t1_range[0], t1_range[1], t1_step)
    t2_values = np.arange(t2_range[0], t2_range[1], t2_step)
    parameter_values = {'t1': t1_values, 't2': t2_values}
    param_list = generate_parameter_combinations_table(parameter_values)
    return param_list


def constant_rel_step(start: int, stop: int, min_abs_step: int, rel_step: float) -> list:
    values = []
    val = start
    while val < stop:
        values.append(val)
        abs_step = rel_step * val
        step = max(abs_step, min_abs_step)
        val += int(step)       
    return values


def generate_parameter_combinations_table(parameter_values: dict, apply_physical_constraints: bool=True) -> dict:
    """ 
    Constructs a meshgrid of given parameters values. This is can be used as the 'table' form generator for MRFDict().
    """

    t1_values = np.array(parameter_values['t1'])
    t2_values = np.array(parameter_values['t2'])

    if 'df' in parameter_values.keys():
        df_values = eval(parameter_values['df'])
        t1_table, t2_table, df_table = np.meshgrid(t1_values, t2_values, df_values)
        t1_table, t2_table, df_table = t1_table.flatten(), t2_table.flatten(), df_table.flatten()
    else:
        t1_table, t2_table = np.meshgrid(t1_values, t2_values)
        t1_table, t2_table = t1_table.flatten(), t2_table.flatten()

    # Enforce the T1 < T2 condition
    if apply_physical_constraints:
        valid_values_mask = t1_table > t2_table
        t1_table, t2_table = t1_table[valid_values_mask], t2_table[valid_values_mask]
    
    param_list = {'t1': t1_table, 't2': t2_table}

    if 'df' in parameter_values.keys():
        df_table = df_table[valid_values_mask]
        param_list['df'] = df_table
        
    return param_list
