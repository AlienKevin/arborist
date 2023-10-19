import random
import sys

if len(sys.argv) != 2:
    print("Usage: python script.py <sample_percent>")
    print("Example: python sample_benchmarks.py 0.1")
    sys.exit(1)

sample_percent = float(sys.argv[1])

random.seed(0)

include = ['W8T3', 'W308T1', 'W51T1', 'W48T1', 'W156T1', 'W112T1', 'W40T1', 'W56T1', 'W80T1', 'W307T2', 'W295T1', 'W239T2', 'W204T1', 'W110T2', 'W218T1', 'W239T1', 'W307T1', 'W6T1', 'W261T2', 'W253T1', 'W169T1', 'W228T3', 'W228T2', 'W218T2', 'W144T1', 'W120T1', 'W228T1', 'W91T2', 'W268T1', 'W49T1', 'W250T1', 'W303T1', 'W110T3', 'W302T1', 'W8T2', 'W8T1', 'W265T2', 'W265T1', 'W228T4', 'W274T1', 'W87T1', 'W133T1', 'W1T1', 'W138T1', 'W154T1', 'W232T2', 'W240T1', 'W99T1', 'W1T2', 'W284T1', 'W296T1', 'W162T1', 'W309T1', 'W178T1', 'W33T1', 'W263T2', 'W278T1', 'W134T1', 'W158T2', 'W189T1', 'W49T2', 'W34T2', 'W252T1', 'W158T1', 'W252T2', 'W25T1',
           'W287T2', 'W237T1', 'W87T2', 'W34T3', 'W69T1', 'W232T1', 'W77T1', 'W34T1', 'W214T1', 'W226T1', 'W124T1', 'W287T1', 'W74T1', 'W139T1', 'W115T1', 'W304T2', 'W262T1', 'W173T1', 'W74T2', 'W304T1', 'W1T3', 'W164T1', 'W223T1', 'W141T1', 'W305T1', 'W148T1', 'W146T1', 'W233T1', 'W81T1', 'W125T1', 'W146T2', 'W263T1', 'W285T1', 'W190T1', 'W213T1', 'W157T1', 'W157T2', 'W276T1', 'W53T1', 'W127T1', 'W254T1', 'W88T1', 'W69T2', 'W54T2', 'W205T1', 'W91T3', 'W51T2', 'W18T1', 'W188T1', 'W14T1', 'W46T1', 'W52T1', 'W111T2', 'W7T2', 'W9T1', 'W50T1', 'W78T2', 'W111T1', 'W176T1', 'W177T1', 'W149T1', 'W3T1', 'W58T1', 'W149T2', 'W238T1', 'W38T1', 'W78T1', 'W7T1']
include = [x for x in include if x not in ["W7T1", "W51T1", "W78T1"]]

pldi_benchmarks = [
    'W1T1', 'W3T1', 'W6T1', 'W8T1', 'W9T1', 'W14T1', 'W18T1', 'W25T1', 'W33T1', 'W34T1', 'W38T1', 'W40T1', 'W46T1', 'W49T1', 'W50T1', 'W52T1', 'W58T1', 'W69T1', 'W77T1', 'W81T1', 'W87T1', 'W88T1', 'W111T1', 'W115T1',
    'W125T1', 'W127T1', 'W133T1', 'W134T1', 'W138T1', 'W141T1', 'W144T1', 'W146T1', 'W148T1', 'W149T1', 'W157T1', 'W162T1', 'W177T1', 'W178T1', 'W188T1', 'W190T1', 'W204T1', 'W213T1',
    'W223T1', 'W226T1', 'W228T1', 'W232T1', 'W233T1', 'W237T1', 'W238T1', 'W239T1', 'W240T1', 'W252T1', 'W253T1', 'W254T1', 'W262T1', 'W265T1', 'W268T1', 'W274T1', 'W276T1', 'W284T1',
    'W285T1', 'W287T1', 'W296T1',
    'W99T1', 'W164T1', 'W176T1', 'W189T1', 'W295T1',
    'W48T1', 'W53T1', 'W56T1', 'W80T1', 'W112T1', 'W156T1', 'W205T1', 'W218T1',
]

assert(len(pldi_benchmarks) == 76)

# sample 10% of pldi benchmarks
num_sampled_pldi = int(len(pldi_benchmarks) * sample_percent)
sampled_pldi_benchmarks = random.sample(pldi_benchmarks, num_sampled_pldi)
assert(len(sampled_pldi_benchmarks) == num_sampled_pldi)
print(f"Sampled {num_sampled_pldi} PLDI benchmarks")

# sample 10% of new benchmarks
new_benchmarks = [x for x in include if x not in pldi_benchmarks]
num_sampled_new = int(len(new_benchmarks) * sample_percent)
sampled_new_benchmarks = random.sample(new_benchmarks, num_sampled_new)
assert(len(sampled_new_benchmarks) == num_sampled_new)
print(f"Sampled {num_sampled_new} new benchmarks")

sampled_benchmarks = sampled_pldi_benchmarks + sampled_new_benchmarks
print("Sampled benchmarks:")
print(sampled_benchmarks)

for id in sampled_benchmarks:
    print(f"tests/benchmarks/{id}", end=" ")
print("")
