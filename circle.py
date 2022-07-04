import os


start_percentage = 3
end_percentage = 10

max_percentage = 0.5

min_percentage = max_percentage/ (2** (end_percentage - start_percentage))

total_frame = 200


start_neigh = 0
end_neigh = 10

modification = "remove"
physical= "1e6 0.48"


dt = "0.001"

first_command = "cargo run --example circle  --release --features log "+physical+" "+dt+" 1e3 1 25  1.0  0.0 0  circle"+modification+"  nothing " +str(total_frame)+"  "+modification+ " 10  "
os.system(first_command)

for i in range(start_percentage, end_percentage+1):
# for i in range(1, 20):
    epi = (2** (i-start_percentage)) * min_percentage
    # epi = i/100.0
    for j in range(start_neigh,end_neigh):
        command = "cargo run --example circle --release --features log "+physical+"  "+dt+"  1e3 1 25 1.0  "+str(epi)+"  "+str(j)+" circle"+modification+"  nothing " +str(total_frame)+"  "+modification+ " 10  "
        os.system(command)

