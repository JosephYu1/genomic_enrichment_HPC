#!/bin/python
###
#   name    | mary lauren benton
#   created | 2017
#   updated | 2018.10.09
#             2018.10.11
#             2018.10.29
#             2019.02.01
#             2019.04.08
#             2019.06.10
#             2019.11.05
#             2021.02.24
#
#   name    | joseph yu
#   updated | 2021.07.15
#             2021.08.03
#             2021.08.15
#             2022.01.06
#
#   depends on:
#       BEDtools v2.23.0-20 via pybedtools
#       /dors/capra_lab/data/dna/[species]/[species]/[species]_trim.chrom.sizes
#       /dors/capra_lab/data/dna/[species]/[species]_chrom-sizes.bed
#       /dors/capra_lab/users/bentonml/data/dna/[species]/[species]_blacklist_gap.bed
#       /dors/capra_lab/data/dna/[species]/[species]-blacklist.bed
###

import csv
import os
import sys, traceback
import gzip
import argparse
import datetime
import math
import numpy as np
from functools import partial
from multiprocessing import Pool
from pybedtools import BedTool
from pybedtools.helpers import BEDToolsError, cleanup, get_tempdir, set_tempdir


###
#   arguments
###
arg_parser = argparse.ArgumentParser(description="Calculate enrichment between bed files.")

arg_parser.add_argument("region_file_1", help='bed file 1 (shuffled)')
arg_parser.add_argument("region_file_2", help='bed file 2 (not shuffled)')

arg_parser.add_argument("-i", "--iters", type=int, default=100,
                        help='number of simulation iterations; default=100')

arg_parser.add_argument("-s", "--species", type=str, default='hg19', choices=['hg19', 'hg38', 'mm10', 'dm3'],
                        help='species and assembly; default=hg19')

arg_parser.add_argument("-b", "--blacklist", type=str, default=None,
                        help='custom blacklist file; default=None')

arg_parser.add_argument("--GC_blacklist", type=str, default=None,
                        help='custom blacklist file for GC_option (content of file is up to user); default=None')

arg_parser.add_argument("-n", "--num_threads", type=int,
                        help='number of threads; default=SLURM_CPUS_PER_TASK or 1')

arg_parser.add_argument("--print_counts_to", type=str, default=None,
                        help="print expected counts to file")

arg_parser.add_argument("--elem_wise", action='store_true', default=False,
                        help='perform element-wise overlaps; default=False')

arg_parser.add_argument("--by_hap_block", action='store_true', default=False,
                        help='perform haplotype-block overlaps; default=False')

arg_parser.add_argument("--GC_option", action='store_true', default=False,
                        help='perform shuffling with regions of similar GC content; default=False')

arg_parser.add_argument("--GC_max", type=int, default=None,
                        help="custom max GC percent threshold (integer, ex: 80) for GC_option, must be used with --GC_min, default is --GC_margin default settings")

arg_parser.add_argument("--GC_min", type=int, default=None,
                        help="custom min GC percent threshold (integer, ex: 20) for GC option, must be used with --GC_max, default is --GC_margin default settings")

arg_parser.add_argument(
    "--GC_window_cache",
    type=str,
    default=None,
    help=(
        "Optional precomputed compressed genome GC window cache. "
        "Expected tab-delimited columns: chrom, start, end, gc_fraction. "
        "Example row: chr1\\t0\\t100\\t0.420000. "
        "If not provided, the script looks in the script directory for a matching "
        "*.bed.gz cache based on species and --GC_bp_resolution. If none exists, "
        "it creates one automatically."
    )
)

arg_parser.add_argument(
    "--GC_whitelist_cache",
    type=str,
    default=None,
    help=(
        "Optional precomputed final GC-compatible whitelist BED file used by "
        "bedtools shuffle with incl=. Expected tab-delimited columns: chrom, start, end. "
        "Example row: chr1\\t10000\\t25000. "
        "If not provided, the script looks in the script directory for a matching "
        "*.bed file based on species, --GC_bp_resolution, GC range, and blacklist. "
        "If none exists, it creates one automatically."
    )
)

#
# restricted_float
#
# updated | 2021.7.19
#
# Description:
#       This function is a wrapper for the float function to check for valid
#       input for the --GC_margin option.
#
#       Acceptable range: Non negative decimals
#
# input:
#       x: input commandline parameters for the --GC_margin option
#
# output:
#       returns the valid float parameters or raise an ArgumentTypeError.
#
def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x <= 0.0:
        raise argparse.ArgumentTypeError("%r not a positive percentage" % (x,))
    return x

arg_parser.add_argument("--GC_margin", type=restricted_float, default=0.1,
                        help='adjust GC content allowed margin in GC_option; \
                        default=0.1(10%% GC content error margin)')

arg_parser.add_argument("--GC_bp_resolution", type=int, default=100,
                        help='adjust GC content bp resolution in GC_option; \
                        default=100(bp)')

args = arg_parser.parse_args()

# save parameters
ANNOTATION_FILENAME = args.region_file_1
TEST_FILENAME = args.region_file_2
COUNT_FILENAME = args.print_counts_to
ITERATIONS = args.iters
SPECIES = args.species
ELEMENT = args.elem_wise
HAPBLOCK= args.by_hap_block
CUSTOM_BLIST = args.blacklist
GC_BLACKLIST = args.GC_blacklist
GC_CTRL_OPT = args.GC_option
GC_CTRL_RANGE = args.GC_margin
GC_CTRL_RESOLUTION = args.GC_bp_resolution
GC_MAX = args.GC_max
GC_MIN = args.GC_min
GC_WINDOW_CACHE = args.GC_window_cache
GC_WHITELIST_CACHE = args.GC_whitelist_cache

# calculate the number of threads
if args.num_threads:
    if args.num_threads >= 40:
        print("Capping the thread count at 40.")
        num_threads = 40
    else:
        num_threads = args.num_threads
else:
    num_threads = int(os.getenv('SLURM_CPUS_PER_TASK', 1))

# if running on slurm, set tmp to runtime dir
set_tempdir(os.getenv('ACCRE_RUNTIME_DIR', get_tempdir()))


###
#   functions
###

#
# loadConstants
#
# updated | 2021.7.19
#
# Description:
#       This function returns the file path for the specified blacklist file.
#
# input:
#       species: The species the genome build belongs to
#       custom:  The custom black list file specified by the user
#
# output:
#       return: return the default blacklist file path from the blackListFile
#       dir matching the species specified or the custom blacklist file path.
#
def loadConstants(species, custom=''):
    if custom is not None:
        return custom
    else:
        return {'hg19' : "./blackListFile/hg19_blacklist_gap.bed",
                'hg38' : "./blackListFile/hg38_blacklist_gap.bed",
                'mm10' : "./blackListFile/mm10_blacklist_gap.bed",
                'dm3'  : "./blackListFile/dm3_blacklist_gap.bed"
                }[species]


#
# caclulateObserved
#
# updated |
#
# Description:
#       This function calculates the observed intersection results for the two
#       bed files passed in.
#
# input:
#       annotation:  BEDTOOL object with the intersection called on
#       test:        BEDTOOL object passed into the intersection function
#       elementwise: flags for elementwise calculation
#       hapblock:    flags for haplotype-block overlaps
#
# output:
#       returns the observed overlap between two bed files
#
def calculateObserved(annotation, test, elementwise, hapblock):
    obs_sum = 0

    if elementwise:
        obs_sum = annotation.intersect(test, u=True).count()
    else:
        obs_intersect = annotation.intersect(test, wo=True)

        if hapblock:
            obs_sum = len(set(x[-2] for x in obs_intersect))
        else:
            for line in obs_intersect:
                obs_sum += int(line[-1])

    return obs_sum


#
# get_script_dir
#
# updated | 2026.05.06
#
# Description:
#       This function returns the directory containing this script.
#
# input:
#       None
#
# output:
#       return: absolute path to the script directory
#
def get_script_dir():
    """
    Return the directory where this script lives.
    """
    return os.path.dirname(os.path.abspath(__file__))


#
# open_maybe_gzip
#
# updated | 2026.05.06
#
# Description:
#       This function opens either a plain text file or gzip-compressed
#       text file. Files ending in ".gz" are opened with gzip.open,
#       otherwise they are opened with the regular open function.
#
# input:
#       filename: path to the file to open
#       mode:     file open mode; default="rt"
#
# output:
#       return: open file handle
#
def open_maybe_gzip(filename, mode="rt"):
    """
    Open plain text or gzip-compressed files.
    """
    if filename.endswith(".gz"):
        return gzip.open(filename, mode)
    return open(filename, mode)


#
# safe_basename
#
# updated | 2026.05.06
#
# Description:
#       This function extracts the base filename from a path and removes
#       common BED file extensions. The returned string is used when
#       constructing cache filenames.
#
# input:
#       path: path to a file
#
# output:
#       return: simplified filename string safe for cache naming
#
def safe_basename(path):
    """
    Convert a path into a safe short basename for cache filenames.
    """
    if path is None:
        return "none"

    base = os.path.basename(path)
    return (
        base
        .replace(".bed.gz", "")
        .replace(".bed", "")
        .replace("/", "_")
        .replace(" ", "_")
    )


#
# get_auto_gc_window_cache_path
#
# updated | 2026.05.06
#
# Description:
#       This function constructs the default filepath for the compressed
#       genome GC window cache. This cache stores all genome windows and
#       their GC content for a specific species and GC window resolution.
#
#       The output file is stored in the same directory as this script.
#
# input:
#       species:       genome build/species string
#       GC_resolution: window size used for GC calculation
#
# output:
#       return: default filepath for the genome GC window cache
#
def get_auto_gc_window_cache_path(species, GC_resolution):
    """
    Cache for all genome windows and their GC content.
    Depends only on species and GC window/step size.
    """
    step_size = math.trunc(GC_resolution / 2)

    filename = (
        f"{species}_genome_gc_windows_"
        f"w{GC_resolution}_s{step_size}.bed.gz"
    )

    return os.path.join(get_script_dir(), filename)


#
# get_auto_gc_whitelist_cache_path
#
# updated | 2026.05.06
#
# Description:
#       This function constructs the default filepath for the final
#       GC-compatible whitelist BED file. This whitelist stores genome
#       regions that pass the GC content filter and have regular
#       blacklist/gap regions removed.
#
#       This file is used as the include file for bedtools shuffle when
#       the GC option is enabled.
#
# input:
#       species:             genome build/species string
#       GC_resolution:       window size used for GC calculation
#       lowerGC:             lower GC content cutoff
#       upperGC:             upper GC content cutoff
#       blacklist_file_name: regular blacklist/gap BED file subtracted
#                            from the GC-compatible regions
#
# output:
#       return: default filepath for the final GC-compatible whitelist BED
#
def get_auto_gc_whitelist_cache_path(
    species,
    GC_resolution,
    lowerGC,
    upperGC,
    blacklist_file_name
):
    """
    Cache for the final filtered whitelist BED file.
    """
    step_size = math.trunc(GC_resolution / 2)
    blacklist_label = safe_basename(blacklist_file_name)

    filename = (
        f"{species}_gc_whitelist_"
        f"w{GC_resolution}_s{step_size}_"
        f"gc{lowerGC:.6f}-{upperGC:.6f}_"
        f"excl_{blacklist_label}.bed"
    )

    return os.path.join(get_script_dir(), filename)


#
# resolve_cache_path
#
# updated | 2026.05.06
#
# Description:
#       This function determines whether to use an explicitly supplied
#       cache file or an automatically generated cache path.
#
#       If the explicit path is supplied, the function checks that the
#       file exists and returns it. If no explicit path is supplied, the
#       function checks whether the automatic cache file exists. If it
#       exists, the file is reused. If it does not exist, the automatic
#       path is returned as the location where the cache should be
#       created.
#
# input:
#       explicit_path: user-supplied cache filepath, or None
#       auto_path:     automatically generated cache filepath
#       description:   text description of the cache type for printed
#                      status messages
#
# output:
#       return: tuple of cache filepath and boolean indicating whether
#               that cache already exists
#
def resolve_cache_path(explicit_path, auto_path, description):
    """
    Decide whether to use an explicit cache or an automatic cache path.
    """
    if explicit_path is not None:
        if not os.path.exists(explicit_path):
            print(f"Error: explicitly supplied {description} does not exist: {explicit_path}")
            sys.exit(1)

        print(f"Using explicitly supplied {description}: {explicit_path}")
        return explicit_path, True

    if os.path.exists(auto_path):
        print(f"Found existing automatic {description}: {auto_path}")
        return auto_path, True

    print(f"No existing automatic {description} found.")
    print(f"Will create new {description}: {auto_path}")
    return auto_path, False

#
# caclulateGCBlackListRegion
#
# updated | 2021.7.20
#           2026.05.06
#
# Description:
#       This function calculates the whitelist regions from the GC content
#       restrictions.
#
#       This updated version supports caching two files:
#
#       1. A compressed genome GC window cache. This file stores genome
#          windows and their GC content for a specified species and
#          GC resolution.
#
#          Example row:
#              chr1    0    100    0.420000
#
#       2. A final GC-compatible whitelist BED file. This file stores
#          merged genome regions that pass the GC content filter after
#          subtracting regular blacklist/gap regions. This file is used
#          as the include file for bedtools shuffle.
#
#          Example row:
#              chr1    10000    25000
#
#       If the cache files already exist, they are reused. If they do
#       not exist and no explicit cache file is supplied, the function
#       creates them in the same directory as this script.
#
# input:
#       species:              The species/genome build used
#       GC_resolution:        The window size for the GC content calculation
#       GC_range:             The range of tolerance for GC content to vary
#                             from the annotation median
#       annotation:           BEDTOOL object containing the annotation regions
#       blacklist_file_name:  BED file containing blacklist/gap regions to
#                             subtract from the GC-compatible regions
#       GC_window_cache:      Optional user-supplied compressed genome GC
#                             window cache file
#       GC_whitelist_cache:   Optional user-supplied final GC-compatible
#                             whitelist BED file
#
# output:
#       return: GC-compatible whitelist BEDTOOL object, numpy array of
#               annotation GC content values, and filepath to the final
#               GC-compatible whitelist BED file
#
def calculateGC_blackListRegion(
    species,
    GC_resolution,
    GC_range,
    annotation,
    blacklist_file_name,
    GC_window_cache=None,
    GC_whitelist_cache=None
):
    print("running calculateGC_blackListRegion")

    genomeSizeFile = {
        'hg19' : './genomeGC/hg19_manual.txt',
        'hg38' : './genomeGC/hg38_manual.txt',
        'mm10' : './genomeGC/mm10_manual.txt',
        'dm3'  : './genomeGC/dm3_manual.txt'
    }[species]

    genomeFasta = {
        'hg19' : './genomeFASTA/hg19.fa',
        'hg38' : './genomeFASTA/hg38.fa',
        'mm10' : './genomeFASTA/mm10.fa',
        'dm3'  : './genomeFASTA/dm3.fa'
    }[species]

    print("running annotation GC content calculation")
    annotationGC_result = annotation.nucleotide_content(fi=genomeFasta)

    annotationGC = []
    for entry in annotationGC_result:
        annotationGC.append(float(entry[-8]))

    np_annotationGC = np.array(annotationGC)
    median = np.median(np_annotationGC)

    if GC_MAX is not None and GC_MIN is not None:
        upperGC = GC_MAX
        lowerGC = GC_MIN
    else:
        print("Using median to set GC range")
        upperGC = median * (1 + GC_range)
        lowerGC = median * (1 - GC_range)

    print(f"Annotation median GC: {median}")
    print(f"Allowed GC range: {lowerGC} to {upperGC}")

    auto_window_cache = get_auto_gc_window_cache_path(
        species,
        GC_resolution
    )

    window_cache_path, window_cache_exists = resolve_cache_path(
        explicit_path=GC_window_cache,
        auto_path=auto_window_cache,
        description="GC window cache"
    )

    auto_whitelist_cache = get_auto_gc_whitelist_cache_path(
        species=species,
        GC_resolution=GC_resolution,
        lowerGC=lowerGC,
        upperGC=upperGC,
        blacklist_file_name=blacklist_file_name
    )

    whitelist_cache_path, whitelist_cache_exists = resolve_cache_path(
        explicit_path=GC_whitelist_cache,
        auto_path=auto_whitelist_cache,
        description="GC whitelist cache"
    )

    if whitelist_cache_exists:
        print(f"Using existing final GC whitelist BED: {whitelist_cache_path}")
        genomeGC_whitelist_Object = BedTool(whitelist_cache_path)
        return genomeGC_whitelist_Object, np_annotationGC, whitelist_cache_path

    print(f"Creating final GC whitelist BED: {whitelist_cache_path}")

    raw_whitelist_path = whitelist_cache_path + ".raw_unmerged"

    with open(raw_whitelist_path, "w") as raw_whitelist:

        if window_cache_exists:
            print(f"Reading genome GC windows from cache: {window_cache_path}")

            with open_maybe_gzip(window_cache_path, "rt") as cache_file:
                for line in cache_file:
                    if not line.strip() or line.startswith("#"):
                        continue

                    fields = line.rstrip("\n").split("\t")

                    if len(fields) < 4:
                        continue

                    chrom = fields[0]
                    start = fields[1]
                    end = fields[2]
                    gc = float(fields[3])

                    if lowerGC <= gc <= upperGC:
                        raw_whitelist.write(f"{chrom}\t{start}\t{end}\n")

        else:
            print("Creating genome windows")
            splitBed = BedTool()
            coverageOverlap = math.trunc(GC_resolution / 2)

            splitGenome = splitBed.window_maker(
                g=genomeSizeFile,
                w=int(GC_resolution),
                s=coverageOverlap
            )

            print("Calculating genome-wide GC content")
            genomeGC_result = splitGenome.nucleotide_content(fi=genomeFasta)

            print(f"Writing compressed genome GC window cache: {window_cache_path}")

            with gzip.open(window_cache_path, "wt") as window_cache_file:
                window_cache_file.write("#chrom\tstart\tend\tgc_fraction\n")

                for window in genomeGC_result:
                    chrom = window[0]
                    start = window[1]
                    end = window[2]
                    gc = float(window[-8])

                    window_cache_file.write(
                        f"{chrom}\t{start}\t{end}\t{gc:.6f}\n"
                    )

                    if lowerGC <= gc <= upperGC:
                        raw_whitelist.write(f"{chrom}\t{start}\t{end}\n")

    print("Sorting and merging GC-compatible regions")

    merged_gc_regions = BedTool(raw_whitelist_path).sort().merge()

    print(f"Subtracting blacklist regions: {blacklist_file_name}")

    bedFile = BedTool(blacklist_file_name)

    merged_gc_regions.subtract(bedFile).sort().merge().saveas(
        whitelist_cache_path
    )

    if os.path.exists(raw_whitelist_path):
        os.remove(raw_whitelist_path)

    print(f"Saved final GC whitelist BED: {whitelist_cache_path}")

    genomeGC_whitelist_Object = BedTool(whitelist_cache_path)

    return genomeGC_whitelist_Object, np_annotationGC, whitelist_cache_path

#
# caclulateExpected_with_GC
#
# updated | 2021.7.19
#           2021.8.15
#
# Description:
#       This function caclulates the expected intersection results with random
#       shuffling. The GC option would use the BEDTOOLS object as a whitelist
#       file, whereas the default option would use the passed in object as a
#       blacklist file when running shuffling.
#
# input:
#       annotation:    BEDTOOL object with the intersection called on
#       test:          BEDTOOL object passed into the intersection function
#       elementwise:   flags for elementwise calculation
#       hapblock:      flags for haplotype-block overlaps
#       species:       species for the genome build used
#       iters:         number of iteration for the calculation
#
# output:
#       returns the calculated overlaps the random shuffling intersection.
#
def calculateExpected_with_GC(annotation, test, elementwise, hapblock, species, blackList_file_name, iters):

    exp_sum = 0
    rand_file = None

    try:

        if GC_CTRL_OPT:
            print("iteration ", iters, end='\r', file=sys.stderr)
            rand_file = annotation.shuffle(genome=species, incl=blackList_file_name, chrom=True, noOverlapping=True)

        else:
            rand_file = annotation.shuffle(genome=species, excl=blackList_file_name, chrom=True, noOverlapping=True)

        if elementwise:
            exp_sum = rand_file.intersect(test, u=True).count()
        else:
            exp_intersect = rand_file.intersect(test, wo=True)

            if hapblock:
                exp_sum = len(set(x[-2] for x in exp_intersect))
            else:
                for line in exp_intersect:
                    exp_sum += int(line[-1])
                    
    except BEDToolsError:
        exp_sum = -999

    return exp_sum

#
# caclulateEmpiricalP
#
# updated | 2021.8.16
#
# Description:
#       This function caclulates empirical P value for the observed vs expected
#
# input:
#       obs:
#       exp_sum_list:
#
# output:
#       returns the formatted result of the calulated P value
#
def calculateEmpiricalP(obs, exp_sum_list):
    mu = np.mean(exp_sum_list)
    sigma = np.std(exp_sum_list)
    dist_from_mu = [exp - mu for exp in exp_sum_list]
    p_sum = sum(1 for exp_dist in dist_from_mu if abs(exp_dist) >= abs(obs - mu))

    # add pseudocount only to avoid divide by 0 errors
    if mu == 0:
        fold_change = (obs + 1.0) / (mu + 1.0)
    else:
        fold_change = obs / mu

    p_val = (p_sum + 1.0) / (len(exp_sum_list) + 1.0)

    return "%d\t%.3f\t%.3f\t%.3f\t%.3f" % (obs, mu, sigma, fold_change, p_val)


###
#   main
###
def main(argv):

    if ( GC_MAX is None and GC_MIN is not None ) or (GC_MAX is not None and GC_MIN is None):
        print("Error: optons --GC_max and --GC_min must be used together")
        exit(1)

    # print header
    print('python {:s} {:s}'.format(' '.join(sys.argv), str(datetime.datetime.now())[:20]))

    # run initial intersection and save
    print("runnning calculateOberved")
    obs_sum = calculateObserved(BedTool(ANNOTATION_FILENAME), BedTool(TEST_FILENAME), ELEMENT, HAPBLOCK)

   # duplicate function for above code block with GC option enabled
    GC_blacklist = None
    np_annotationGC = None
    blackList_file_name = None
    BLACKLIST = loadConstants(SPECIES, CUSTOM_BLIST)

    if GC_CTRL_OPT:

    # If the user provides the older --GC_blacklist argument,
    # treat it as the final GC-compatible whitelist file.
    if GC_BLACKLIST is not None:
        blackList_file_name = GC_BLACKLIST
        print(f"Using GC whitelist from --GC_blacklist: {blackList_file_name}")

    else:
        GC_blacklist, np_annotationGC, blackList_file_name = calculateGC_blackListRegion(
            SPECIES,
            GC_CTRL_RESOLUTION,
            GC_CTRL_RANGE,
            BedTool(ANNOTATION_FILENAME),
            BLACKLIST,
            GC_WINDOW_CACHE,
            GC_WHITELIST_CACHE
        )

    else:
        blackList_file_name = BLACKLIST

    print("running calculateExpected_with_GC")
    pool = Pool(num_threads)
    partial_calcExp = partial(calculateExpected_with_GC, BedTool(ANNOTATION_FILENAME), BedTool(TEST_FILENAME), ELEMENT, HAPBLOCK, SPECIES, blackList_file_name)
    exp_sum_list = pool.map(partial_calcExp, [i for i in range(ITERATIONS)])

    print("Finish calculateExpected_with_GC")
    # wait for results to finish before calculating p-value
    pool.close()
    pool.join()

    # remove iterations that throw bedtools exceptions
    final_exp_sum_list = [x for x in exp_sum_list if x >= 0]
    exceptions = exp_sum_list.count(-999)

    # printing GC content statistics
    if GC_CTRL_OPT and GC_BLACKLIST is None:
        print('The mean of the GC content in ANNOTATION file is: {0}'.format(str(np.mean(np_annotationGC))))
        print('The median of the GC content in ANNOTATION file is: {0}'.format(str(np.median(np_annotationGC))))
        print('The min and max value of the GC content in ANNOTATION file is: {0} and {1}'.format(str(np.min(np_annotationGC)), str(np.max(np_annotationGC))))
        print('The 25th and 75th percentile of the GC content in ANNOTATION file is: {0} and {1}'.format(str(np.percentile(np_annotationGC, 25)), str(np.percentile(np_annotationGC, 75))))

    # calculate empirical p value
    print('Observed\tExpected\tStdDev\tFoldChange\tp-value')
    if exceptions / ITERATIONS <= .1:
        print(calculateEmpiricalP(obs_sum, final_exp_sum_list))
        print(f'iterations not completed: {exceptions}', file=sys.stderr)
    else:
        print(f'iterations not completed: {exceptions}\nresulted in nonzero exit status', file=sys.stderr)
        cleanup()
        sys.exit(1)

    if COUNT_FILENAME is not None:
        with open(COUNT_FILENAME, "w") as count_file:
            count_file.write('{}\n{}\n'.format(obs_sum, '\t'.join(map(str, exp_sum_list))))

    # clean up any pybedtools tmp files
    cleanup()


if __name__ == "__main__":
    main(sys.argv[1:])
