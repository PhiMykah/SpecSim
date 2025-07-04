import nmrPype as pype
from ..datatypes import Vector

import sys
import argparse
from random import randint
import nmrPype as pype

def parse_command_line(argument_list : str | list) -> argparse.Namespace :
    parser = argparse.ArgumentParser(description='specsim : simulate NMR spectral simulator')
    parser.add_argument('-tab', type=str, default='master.tab', help='Peak Table Input.') #
    parser.add_argument('-fid', type=str, default='test.fid', help='NMRPipe-format Time-Domain Input.') #
    parser.add_argument('-ft1', type=str, default='test.ft1', help='Corresponding NMRPipe-format Interferogram Input.') #
    parser.add_argument('-ft2', type=str, default='test.ft2', help='Corresponding NMRPipe-format Freq-Domain Input.') #
    parser.add_argument('-apod', type=str, default=None, help='Optional NMRPipe-format Apodization Profile.')
    parser.add_argument('-out', type=str, default=None, help='NMRPipe-format Time-Domain Output, or Keyword None.') #
    parser.add_argument('-basis', type=str, default=None, metavar='FILEPATH', help='Save Each Peak in a Basis Set, Designate the Folder Path.')
    parser.add_argument('-ndim', type=int, default=2, help='Number of dimensions to simulate (integer, by default 2)')
    parser.add_argument('-res', type=str, default=None, help='NMRPipe-format Time-Domain Residual, or Keyword None.')
    parser.add_argument('-scale', type=float, nargs='+', default=[1.0, 1.0], help="Amplitude Scaling Factors (list of floats)") #
    parser.add_argument('-rx1', type=int, default=0, help='First Point Location for Calculating Residual.')
    parser.add_argument('-rxn', type=int, default=0, help='Last Point Location for Calculating Residual.')
    parser.add_argument('-mode', type=str, choices=['lsq', 'basin', 'minimize', 'brute', 'danneal'], default='lsq', help='Optimization mode (lsq, basin, minimize, brute).') #
    parser.add_argument('-trials', type=int, default=100, help='Number of Optimization Trials.') #
    parser.add_argument('-maxFail', type=int, default=0, help='Max Optimization Fails Before Quitting.')
    parser.add_argument('-iseed', type=int, default=randint(1, sys.maxsize), help='Random Number Seed.')
    parser.add_argument('-verb', "-verbose", action='store_true', help='Verbose Mode ON (Default OFF).') #
    parser.add_argument('-noverb', "-noverbose", action='store_true', help='Verbose Mode OFF.') #
    parser.add_argument('-report', action='store_true', help='Report Mode ON.')
    parser.add_argument('-freq', type=float, nargs='+', default=None, help='Frequency Positions (list of floats).')
    parser.add_argument('-model', type=str, choices=['exp', 'gauss', 'comp'], default='exp', help='Optimization mode (exponential, gaussian, composite).')
    parser.add_argument('-initXDecay', '-initDecay', type=float, nargs='+', default=[2.0], help='Initial x-axis decay values in Hz (list of floats, one for each model).')
    parser.add_argument('-initYDecay', type=float, nargs='+', default=[0.0], help='Initial y-axis decay values in Hz (list of floats, one for each model).')
    parser.add_argument('-xDecayBounds', type=float, nargs=2, default=[0.0, 100.0], metavar=('LOWER', 'HIGHER'), help='Lower and upper bounds for x-decay in Hz.')
    parser.add_argument('-yDecayBounds', type=float, nargs=2, default=[0.0, 20.0], metavar=('LOWER', 'HIGHER'), help='Lower and upper bounds for y-decay in Hz.')
    parser.add_argument('-ampBounds', type=float, nargs=2, default=[0.0, 10.0], metavar=('LOWER', 'HIGHER'), help='Lower and upper bounds for amplitude.')
    parser.add_argument('-p0Bounds', type=float, nargs=2, default=[-180.0, 180.0], metavar=('LOWER', 'UPPER'), help='Lower and upper bounds for p0 phase correction.')
    parser.add_argument('-p1Bounds', type=float, nargs=2, default=[-180.0, 180.0], metavar=('LOWER', 'UPPER'), help='Lower and upper bounds for p1 phase correction.')
    parser.add_argument('-step', type=float, default=0.1, help='Step-size for optimizations that require step-size (e.g. basin).')
    parser.add_argument('-off', type=float, nargs='+', default=[0.0, 0.0], help="Optional Frequency offset value in pts. (list of floats)") #
    parser.add_argument('-j1', type=float, default=None, help='Coupling 1 (Cosine Modulation, Pts Hz ppm %%).')
    parser.add_argument('-j2', type=float, default=None, help='Coupling 2 (Cosine Modulation, Pts Hz ppm %%).')
    parser.add_argument('-j3', type=float, default=None, help='Coupling 3 (Cosine Modulation, Pts Hz ppm %%).')
    parser.add_argument('-xP0', type=float, nargs='+', default=[0.0], help='Zero Order Phase of All Signals for x-axis (list of floats, one for each model).') #
    parser.add_argument('-xP1', type=float, nargs='+', default=[0.0], help='First Order Phase of All Signals for x-axis (list of floats, one for each model).') #
    parser.add_argument('-yP0', type=float, nargs='+', default=[0.0], help='Zero Order Phase of All Signals for y-axis (list of floats, one for each model).') #
    parser.add_argument('-yP1', type=float, nargs='+', default=[0.0], help='First Order Phase of All Signals for y-axis (list of floats, one for each model).') #
    parser.add_argument('-ts', action='store_true', help='Scale Time-Domain Signal by Decay Integral.')
    parser.add_argument('-nots', action='store_true', help='No Time-Domain Scale (Default OFF).')
    parser.add_argument('-notdd', action='store_true', help='Interpret Linewidth in Frequency Domain (Default OFF).')
    parser.add_argument('-tdd', action='store_true', help='Interpret Linewidth as Time Domain Decay (Default OFF).')
    parser.add_argument('-tdj', action='store_true', help='Interpret J-Modulation in Time Domain (Default OFF).')
    parser.add_argument('-notdj', action='store_true', help='Interpret J-Modulation in Frequency Domain (Default OFF).')

    return parser.parse_args(argument_list)

def get_dimension_info(data_frame : pype.DataFrame, data_type : str, dimension_count : int = 2) -> Vector[float]:
    """
    Obtain each dimension information from the data frame.

    Parameters
    ----------
    data_frame : pype.DataFrame
        Target data frame
    data_type : str
        Header key for the data frame
    dimension_count : int
        Number of dimensions to collect information from
    Returns
    -------
    Vector[float]
        Value for each dimension
    """
    data : list[float] = []
    for i in range(dimension_count):
        data.append(data_frame.getParam(data_type, i+1))

    return Vector(data)

def get_total_size(data_frame : pype.DataFrame, header_key : str, dimension_count : int = 2) -> Vector[int]:
    """
    Obtain the total size of the data frame.

    Parameters
    ----------
    data_frame : nmrPype.DataFrame
        Target data frame
    header_key : str
        NMR Header key for size
    dimension_count : int
        Number of dimensions to collect information from

    Returns
    -------
    Vector[int]
        Total size of the data frame
    """
    data : list[int] = []
    for i in range(dimension_count):
        data.append(int(data_frame.getParam(header_key, i+1)))

    return Vector(data)

class SpecSimArgs :
    """
    A class to parse and store command-line arguments for spectral simulation.

    Attributes
    ----------
    tab : str
        Path to the tabulated data file.
    fid : str
        Path to the FID data file.
    ft1 : str
        Path to the Interferogram data file
    ft2 : str
        Path to the Fourier transformed data file.
    apod : str
        Apodization function to be applied.
    out : str
        Output file path.
    basis : str
        Save Each Peak in a Basis Set, Designate the Folder Path.
    res : float
        Resolution of the simulation.
    ndim : int
        Number of dimensions to simulate
    scale : list[float]
        Scaling factor for the simulation.
    rx1 : float
        Receiver gain for the first receiver.
    rxn : float
        Receiver gain for the nth receiver.
    trials : int
        Number of trials to be performed.
    maxFail : int
        Maximum number of allowed failures.
    iseed : int
        Initial seed for random number generation.
    verb : bool
        Verbose mode flag.
    noverb : bool
        Non-verbose mode flag.
    report : str
        Path to the report file.
    freq : float
        Frequency of the simulation.
    model : str
        Simulation Model for the simulation
    initXDecay : float
        Initial x-axis decay value in Hz for each model.
    initYDecay : float
        Initial y-axis decay value in Hz for each model.
    xDecayBounds : list[float]
        Lower bound and upper bound for x-decay in Hz.
    yDecayBounds : list[float]
        Lower bound and upper bound for y-decay in Hz.
    ampBounds : list[float]
        Lower and upper bounds for amplitude.
    p0Bounds : list[float]
        Lower and upper bounds for p0 phase.
    p1Bounds : list[float]
        Lower and upper bounds for p1 phase.
    step : float
        Step-size for optimizations that require step-size (e.g. basin).
    offsets : list[float]
        offset for each dimension
    j1 : float
        First J-coupling constant.
    j2 : float
        Second J-coupling constant.
    j3 : float
        Third J-coupling constant.
    xP0 : list[float]
        Zero-order phase correction for x-axis for each model.
    xP1 : list[float]
        First-order phase correction for x-axis for each model.
    yP0 : list[float]
        Zero-order phase correction for y-axis for each model.
    yP1 : list[float]
        First-order phase correction for y-axis for each model.
    ts : bool
        Time-domain simulation flag.
    nots : bool
        No time-domain simulation flag.
    notdd : bool
        No time-domain deconvolution flag.
    tdd : bool
        Time-domain deconvolution flag.
    tdj : bool
        Time-domain J-coupling flag.
    notdj : bool
        No time-domain J-coupling flag.
    """
    def __init__(self, args : argparse.Namespace | None = None) -> None :
        if args is None:
            return
        self.tab : str = args.tab
        self.fid : str = args.fid
        self.ft1: str = args.ft1
        self.ft2: str = args.ft2
        self.apod : str | None = args.apod
        self.out : str | None = args.out
        self.basis : str | None = args.basis
        self.ndim : int = args.ndim
        self.res : str | None = args.res
        self.scale : list[float] | None = args.scale
        self.rx1: int = args.rx1
        self.rxn : int = args.rxn
        self.mode : str = args.mode
        self.trials : int = args.trials
        self.maxFail : int = args.maxFail
        self.iseed : int = args.iseed
        self.verb : bool = args.verb
        self.noverb : bool = args.noverb
        self.report : bool = args.report
        self.freq : list[float] | None = args.freq
        self.model : str = args.model
        self.initXDecay : list[float] = args.initXDecay
        self.initYDecay : list[float] = args.initYDecay
        self.xDecayBounds : list[float] = args.xDecayBounds
        self.yDecayBounds : list[float] = args.yDecayBounds
        self.ampBounds : list[float] = args.ampBounds
        self.p0Bounds : list[float] = args.p0Bounds
        self.p1Bounds : list[float] = args.p1Bounds
        self.step : float = args.step
        self.offsets : list[float] = args.off
        self.j1 : float | None = args.j1
        self.j2 : float | None = args.j2
        self.j3 : float | None = args.j3
        self.xP0: list[float] = args.xP0
        self.xP1: list[float] = args.xP1
        self.yP0: list[float] = args.yP0
        self.yP1: list[float] = args.yP1
        self.ts : bool = args.ts
        self.nots : bool = args.nots
        self.notdd : bool = args.notdd
        self.tdd : bool = args.tdd
        self.tdj : bool = args.tdj
        self.notdj : bool = args.notdj


    def set_values(self, tab: str = 'master.tab', fid: str = 'test.fid', ft1: str = 'test.ft1', ft2: str = 'test.ft2', apod: str | None = None, out: str | None = None,
                 basis: str | None = None, ndim: int = 2, res: str | None = None, scale: list[float] | None = [1.0, 1.0], rx1: int = 0, rxn: int = 0, mode: str = 'lsq',
                 trials: int = 100, maxFail: int = 0, iseed: int = randint(1, sys.maxsize), verb: bool = False, noverb: bool = False, report: bool = False,
                 freq: list[float] | None = None, model: str = 'exp', initXDecay: list[float] = [2.0], initYDecay: list[float] = [0.0],
                 xDecayBounds: list[float] = [0.0, 100.0], yDecayBounds: list[float] = [0.0, 20.0], ampBounds: list[float] = [0.0, 10.0],
                 p0Bounds: list[float] = [-180.0, 180.0], p1Bounds: list[float] = [-180.0, 180.0], step: float = 0.1, offsets: list[float] = [0.0, 0.0],
                 j1: float | None = None, j2: float | None = None, j3: float | None = None, xP0: list[float] = [0.0], xP1: list[float] = [0.0], yP0: list[float] = [0.0],
                 yP1: list[float] = [0.0], ts: bool = False, nots: bool = False, notdd: bool = False, tdd: bool = False, tdj: bool = False, notdj: bool = False) -> None:
        self.tab = tab
        self.fid = fid
        self.ft1 = ft1
        self.ft2 = ft2
        self.apod = apod
        self.out = out
        self.basis = basis
        self.ndim = ndim
        self.res = res
        self.scale = scale
        self.rx1 = rx1
        self.rxn = rxn
        self.mode = mode
        self.trials = trials
        self.maxFail = maxFail
        self.iseed = iseed
        self.verb = verb
        self.noverb = noverb
        self.report = report
        self.freq = freq
        self.model = model
        self.initXDecay = initXDecay
        self.initYDecay = initYDecay
        self.xDecayBounds = xDecayBounds
        self.yDecayBounds = yDecayBounds
        self.ampBounds = ampBounds
        self.p0Bounds = p0Bounds
        self.p1Bounds = p1Bounds
        self.step = step
        self.offsets = offsets
        self.j1 = j1
        self.j2 = j2
        self.j3 = j3
        self.xP0 = xP0
        self.xP1 = xP1
        self.yP0 = yP0
        self.yP1 = yP1
        self.ts = ts
        self.nots = nots
        self.notdd = notdd
        self.tdd = tdd
        self.tdj = tdj
        self.notdj = notdj


    def __str__(self) -> str :
        return (f"SpecSimArgs(tab={self.tab}, fid={self.fid}, ft1={self.ft1}, ft2={self.ft2}, apod={self.apod}, out={self.out}, "
                f"res={self.res}, ndim={self.ndim}, scale={self.scale}, rx1={self.rx1}, rxn={self.rxn}, mode={self.mode}, trials={self.trials}, "
                f"maxFail={self.maxFail}, iseed={self.iseed}, verb={self.verb}, noverb={self.noverb}, "
                f"report={self.report}, freq={self.freq}, initXDecay={self.initXDecay}, initYDecay={self.initYDecay}, "
                f"xDecayBounds={self.xDecayBounds}, yDecayBounds={self.yDecayBounds}, "
                f"ampBounds={self.ampBounds}, p0Bounds={self.p0Bounds}, p1Bounds={self.p1Bounds}, step={self.step}, "
                f"offsets={self.offsets}, j1={self.j1}, j2={self.j2}, j3={self.j3}, "
                f"xP0={self.xP0}, xP1={self.xP1}, yP0={self.yP0}, yP1={self.yP1}, "
                f"ts={self.ts}, nots={self.nots}, notdd={self.notdd}, tdd={self.tdd}, tdj={self.tdj}, "
                f"notdj={self.notdj})")

if __name__ == "__main__":
    args : argparse.Namespace = parse_command_line(sys.argv[1:])
    specsim_args = SpecSimArgs(args)
    print(specsim_args)