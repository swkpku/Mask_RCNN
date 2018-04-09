from visdom import Visdom
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--file')

args = parser.parse_args()

viz = Visdom()
iter = 1
loss_mask = [0]
val_loss_mask = [0]

print(args.file)
skip_iters = 100

with open(args.file, 'r') as f:
    for line in f:
        if "mrcnn_class_loss" in line:

            iter = iter + 1

            terms = line.split("-")
            mask_loss_term = terms[7]
            mask_loss = mask_loss_term.split(":")[1]

            loss_mask.append(float(mask_loss))

            if "val_mrcnn_mask_loss" in line:

                terms = line.split("-")
                val_mask_loss_term = terms[13]
                val_mask_loss = val_mask_loss_term.split(":")[1]

                val_loss_mask.append(float(val_mask_loss))
            else:
                val_loss_mask.append(val_loss_mask[-1])


loss_mask_arr = np.array(loss_mask)
val_loss_mask_arr = np.array(val_loss_mask)

viz.line(X=np.column_stack((np.arange(0, iter), np.arange(0, iter))),
         Y=np.column_stack((loss_mask_arr, val_loss_mask_arr)))