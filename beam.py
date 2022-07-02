import os

start_percentage = 0
end_percentage = 10

max_percentage = 0.5

min_percentage = max_percentage / (2**(end_percentage - start_percentage))

total_frame = 300

start_neigh = 0
end_neigh = 10

modification = "no"
physical = "1e6 0.33"

features = "save"

dt = "0.01"
uniform = "true"

command = "cargo run --example beam --release --features " + features + "  " + physical + "  " + dt + "  1e3 20 80 1.0 0.125 0.0 0  beamuniform" + modification + "  nothing " + str(total_frame) + "  " + modification + " 10  " + uniform
os.system(command)