
import matplotlib.pyplot as plt   
from seaborn import color_palette, set_palette
from matplotlib import rc


#from matplotlib.backends.backend_pgf import FigureCanvasPgf

#matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
#pgf_with_custom_preamble = {
#"font.family": "serif", # use serif/main font for text elements
#"text.usetex": True,    # use inline math for ticks
#}
#matplotlib.rcParams.update(pgf_with_custom_preamble)
#plt.style.use('seaborn-white')


plt.style.use('seaborn-white')
set_palette("colorblind")

rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{underscore}')
