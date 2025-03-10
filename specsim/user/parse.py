import sys
import argparse
from random import randint
import nmrPype as pype

def parse_command_line(argument_list : str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='specsim: simulate NMR spectral simulator')
    parser.add_argument('-tab', type=str, default='master.tab', help='Peak Table Input.') #
    parser.add_argument('-fid', type=str, default='test.fid', help='NMRPipe-format Time-Domain Input.') #
    parser.add_argument('-ft1', type=str, default='test.ft1', help='Corresponding NMRPipe-format Interferogram Input.') #
    parser.add_argument('-ft2', type=str, default='test.ft2', help='Corresponding NMRPipe-format Freq-Domain Input.') #
    parser.add_argument('-apod', type=str, default=None, help='Optional NMRPipe-format Apodization Profile.')
    parser.add_argument('-out', type=str, default=None, help='NMRPipe-format Time-Domain Output, or Keyword None.') #
    parser.add_argument('-res', type=str, default=None, help='NMRPipe-format Time-Domain Residual, or Keyword None.')
    parser.add_argument('-scale', type=float, default=1.0, help="Amplitude Scaling Factor") #
    parser.add_argument('-rx1', type=int, default=0, help='First Point Location for Calculating Residual.')
    parser.add_argument('-rxn', type=int, default=0, help='Last Point Location for Calculating Residual.')
    parser.add_argument('-mode', type=str, choices=['lsq', 'basin', 'minimize', 'brute'], default='lsq', help='Optimization mode (lsq, basin, minimize, brute).') #
    parser.add_argument('-trials', type=int, default=100, help='Number of Optimization Trials.') #
    parser.add_argument('-maxFail', type=int, default=0, help='Max Optimization Fails Before Quitting.')
    parser.add_argument('-iseed', type=int, default=randint(1, sys.maxsize), help='Random Number Seed.')
    parser.add_argument('-verb', "-verbose", action='store_true', help='Verbose Mode ON (Default OFF).') #
    parser.add_argument('-noverb', "-noverbose", action='store_true', help='Verbose Mode OFF.') #
    parser.add_argument('-report', action='store_true', help='Report Mode ON.')
    parser.add_argument('-freq', type=float, nargs='+', default=None, help='Frequency Positions (list of floats).')
    parser.add_argument('-eDecay', type=float, nargs='+', default=None, help='Exponential Decays (list of floats).')
    parser.add_argument('-eAmp', type=str, default='Auto', help='Exponential Amplitudes, or Keyword Auto.')
    parser.add_argument('-gDecay', type=str, default=None, help='Gaussian Decays (Pts Hz ppm %%).')
    parser.add_argument('-gAmp', type=str, default='Auto', help='Gaussian Amplitudes, or Keyword Auto.')
    parser.add_argument('-xOff', type=float, default=0.0, help="Optional Frequency offset value in pts.") #
    parser.add_argument('-j1', type=str, default=None, help='Coupling 1 (Cosine Modulation, Pts Hz ppm %%).')
    parser.add_argument('-j2', type=str, default=None, help='Coupling 2 (Cosine Modulation, Pts Hz ppm %%).')
    parser.add_argument('-j3', type=str, default=None, help='Coupling 3 (Cosine Modulation, Pts Hz ppm %%).')
    parser.add_argument('-xP0', type=float, default=0.0, help='Zero Order Phase of All Signals for x-axis.') #
    parser.add_argument('-xP1', type=float, default=0.0, help='First Order Phase of All Signals for x-axis.') #
    parser.add_argument('-yP0', type=float, default=0.0, help='Zero Order Phase of All Signals for y-axis.') #
    parser.add_argument('-yP1', type=float, default=0.0, help='First Order Phase of All Signals for y-axis.') #
    parser.add_argument('-ePhase', type=float, default=0.0, help='Additional Phase for Each Exponential Signal.')
    parser.add_argument('-gPhase', type=float, default=0.0, help='Additional Phase for Each Gaussian Signal.')
    parser.add_argument('-ts', action='store_true', help='Scale Time-Domain Signal by Decay Integral.')
    parser.add_argument('-nots', action='store_true', help='No Time-Domain Scale (Default OFF).')
    parser.add_argument('-notdd', action='store_true', help='Interpret Linewidth in Frequency Domain (Default OFF).')
    parser.add_argument('-tdd', action='store_true', help='Interpret Linewidth as Time Domain Decay (Default OFF).')
    parser.add_argument('-tdj', action='store_true', help='Interpret J-Modulation in Time Domain (Default OFF).')
    parser.add_argument('-notdj', action='store_true', help='Interpret J-Modulation in Frequency Domain (Default OFF).')

    return parser.parse_args(argument_list)

def get_dimension_info(data_frame: pype.DataFrame, data_type : str) -> tuple[float, float]:
    """
    Obtain x-dimension and y-dimension information from the data frame.

    Parameters
    ----------
    data_frame : pype.DataFrame
        Target data frame
    data_type : str
        Header key for the data frame

    Returns
    -------
    tuple[float, float]
        x-dimension and y-dimension values
    """
    return (data_frame.getParam(data_type, 1), data_frame.getParam(data_type, 2))

def get_total_size(data_frame : pype.DataFrame, header_key : str) -> tuple[int, int]:
    """
    Obtain the total size of the data frame.

    Parameters
    ----------
    data_frame : nmrPype.DataFrame
        Target data frame

    header_key : str
        NMR Header key for size

    Returns
    -------
    tuple[int, int]
        Total size of the data frame
    """
    return tuple(map(int, get_dimension_info(data_frame, header_key)))

class SpecSimArgs:
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
    res : float
        Resolution of the simulation.
    scale : float
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
    eDecay : float
        Exponential decay constant.
    eAmp : float
        Exponential amplitude.
    gDecay : float
        Gaussian decay constant.
    gAmp : float
        Gaussian amplitude.
    xOff : float
        X-axis offset.
    j1 : float
        First J-coupling constant.
    j2 : float
        Second J-coupling constant.
    j3 : float
        Third J-coupling constant.
    xP0 : float
        Zero-order phase correction for x-axis.
    xP1 : float
        First-order phase correction for x-axis.
    yP0 : float
        Zero-order phase correction for y-axis.
    yP1 : float
        First-order phase correction for y-axis.
    ePhase : float
        Exponential phase correction.
    gPhase : float
        Gaussian phase correction.
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
    def __init__(self, args: argparse.Namespace):
        self.tab: str = args.tab
        self.fid: str = args.fid
        self.ft1: str = args.ft1
        self.ft2: str = args.ft2
        self.apod: str = args.apod
        self.out: str = args.out
        self.res: str = args.res
        self.scale: float = args.scale
        self.rx1: int = args.rx1
        self.rxn: int = args.rxn
        self.mode: str = args.mode
        self.mode: str = args.mode
        self.trials: int = args.trials
        self.maxFail: int = args.maxFail
        self.iseed: int = args.iseed
        self.verb: bool = args.verb
        self.noverb: bool = args.noverb
        self.report: bool = args.report
        self.freq: list[float] = args.freq
        self.eDecay: list[float] = args.eDecay
        self.eAmp: str = args.eAmp
        self.gDecay: str = args.gDecay
        self.gAmp: str = args.gAmp
        self.xOff: float = args.xOff
        self.j1: str = args.j1
        self.j2: str = args.j2
        self.j3: str = args.j3
        self.xP0: float = args.xP0
        self.xP1: float = args.xP1
        self.yP0: float = args.yP0
        self.yP1: float = args.yP1
        self.ePhase: float = args.ePhase
        self.gPhase: float = args.gPhase
        self.ts: bool = args.ts
        self.nots: bool = args.nots
        self.notdd: bool = args.notdd
        self.tdd: bool = args.tdd
        self.tdj: bool = args.tdj
        self.notdj: bool = args.notdj

    def __str__(self):
        return (f"SpecSimArgs(tab={self.tab}, fid={self.fid}, ft={self.ft}, apod={self.apod}, out={self.out}, "
                f"res={self.res}, scale={self.scale}, rx1={self.rx1}, rxn={self.rxn}, trials={self.trials}, "
                f"maxFail={self.maxFail}, iseed={self.iseed}, verb={self.verb}, noverb={self.noverb}, "
                f"report={self.report}, freq={self.freq}, eDecay={self.eDecay}, eAmp={self.eAmp}, "
                f"gDecay={self.gDecay}, gAmp={self.gAmp}, xOff={self.xOff}, j1={self.j1}, j2={self.j2}, "
                f"j3={self.j3}, xP0={self.xP0}, xP1={self.xP1}, yP0={self.yP0}, yP1={self.yP1}, ePhase={self.ePhase}, gPhase={self.gPhase}, "
                f"ts={self.ts}, nots={self.nots}, notdd={self.notdd}, tdd={self.tdd}, tdj={self.tdj}, "
                f"notdj={self.notdj})")
        
if __name__ == "__main__":
    args = parse_command_line(sys.argv[1:])
    specsim_args = SpecSimArgs(args)
    print(specsim_args)