import os


start_percentage = 3
end_percentage = 10

max_percentage = 0.5

min_percentage = max_percentage/ (2** (end_percentage - start_percentage))

total_frame = 200


start_neigh = 0
end_neigh = 10

modification = "no"
physical= "1e6 0.48"
features = " --features save,log"

dt = "0.0001"

first_command = "cargo run --example circle  --release "+ features+"  "+physical+" "+dt+" 1e3 1 25  1.0  0.0 0  circle"+modification+"  nothing " +str(total_frame)+"  "+modification+ " 10  "
os.system(first_command)