import os, sys
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Downsamples all .npy files in the current folder and below.")
parser.add_argument("--factor", type=int, default=100, help="Downsampling factor.")
parser.add_argument("--startat", type=int, default=0, help="Index to start at.")
args = parser.parse_args()

input_folder = os.getcwd()
folder_name = os.path.split(os.getcwd())[1]
output_folder = "/tmp/{}__ds_{}x_start{}".format(folder_name, args.factor, args.startat)

for root,dirs,files in os.walk("."):
    for f in files:
        if ".npy" in f:
            orig_folder = os.path.abspath(root)
            new_folder  = orig_folder.replace(input_folder, output_folder)
            os.system("mkdir -p {}".format(new_folder))
            X = np.load(os.path.join(orig_folder, f))
            Xds = X[args.startat::args.factor]
            output_file = os.path.join(new_folder,f)
            np.save(output_file, Xds)
            print(output_file)

print("Wrote {}".format(output_folder))
print("ALLDONE.")

            
            
