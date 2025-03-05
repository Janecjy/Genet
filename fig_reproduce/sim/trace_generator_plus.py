import os
input_dir = "../data/synthetic_test"
output_dir = "../data/synthetic_test_plus"

os.makedirs(output_dir, exist_ok=True)
# for each file in input_dir, the format is time bandwidth, create a new file in output_dir that is time bandwidth+5
for file in os.listdir(input_dir):
    with open(os.path.join(input_dir, file), "r") as f:
        lines = f.readlines()
    with open(os.path.join(output_dir, file), "w") as f:
        for line in lines:
            time, bandwidth = line.split()
            f.write(f"{time} {float(bandwidth)+5}\n")