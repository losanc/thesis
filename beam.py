import os
import sys

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

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        # this is on run on cluster
        first_command = "cargo run --example beam --release --features log " + physical + "  " + dt + "  1e3 20 80 1.0 0.125 0.0 0  beam" +uniform+ modification + "  nothing " + str(total_frame) + "  " + modification + " 10  " + uniform
        os.system(first_command)

        for i in range(start_percentage, end_percentage + 1):
            epi = (2**(i - start_percentage)) * min_percentage
            for j in range(start_neigh, end_neigh):
                command = "cargo run --example beam --release --features log " + physical + "  " + dt + " 1e3 20 80 1.0 0.125 " + str(epi) + "  " + str(j) + " beam" +uniform+ modification + "  nothing " + str(total_frame) + "  " + modification + " 10  " + uniform
                os.system(command)

    else:
        pass
        # this is on my pc for testing