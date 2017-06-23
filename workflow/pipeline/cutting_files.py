import h5py
import numpy as np
import vigra

def printname(name):
    print name

if __name__ == "__main__":
	
    path = '/mnt/localdata03/amatskev/neuraldata/'
    f = h5py.File(path + "cremi.splB.train.probs_defect_correct.crop.axes_xyz.split_z.h5", mode="r")
    print "shape_overall: ", np.array(f["z/0/data"]).shape

    b, a, c = input("Gib die parameter ein (getrennt mit Kommas): ")

    crop_probs_z0_data = np.array(f["z/0/data"])[b:a, b:a, :c]
    crop_probs_z1_data = np.array(f["z/1/data"])[b:a, b:a, :c]
    vigra.writeHDF5(crop_probs_z0_data,path + "erstellt/cremi.splB.train.probs_defect_correct.crop.axes_xyz.split_z.h5","z/0/data")
    vigra.writeHDF5(crop_probs_z1_data, path + "erstellt/cremi.splB.train.probs_defect_correct.crop.axes_xyz.split_z.h5","z/1/data")

    g = h5py.File(path + "cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5", mode="r")

    crop_raw_neurons_z0_neuron_ids = np.array(g["z/0/neuron_ids"])[b:a, b:a, :c]
    crop_raw_neurons_z1_neuron_ids = np.array(g["z/1/neuron_ids"])[b:a, b:a, :c]
    crop_raw_neurons_z0_raw = np.array(g["z/0/raw"])[b:a, b:a, :c]
    crop_raw_neurons_z1_raw = np.array(g["z/1/raw"])[b:a, b:a, :c]

    vigra.writeHDF5(crop_raw_neurons_z0_neuron_ids,path + "erstellt/cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5","z/0/neuron_ids")
    vigra.writeHDF5(crop_raw_neurons_z1_neuron_ids,path + "erstellt/cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5","z/1/neuron_ids")
    vigra.writeHDF5(crop_raw_neurons_z0_raw,path + "erstellt/cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5","z/0/raw")
    vigra.writeHDF5(crop_raw_neurons_z1_raw,path + "erstellt/cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5","z/1/raw")


    h = h5py.File(path + "cremi.splB.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5", mode="r")

    crop_wsdt_z0_labels,_,_= vigra.analysis.relabelConsecutive(np.array(h["z/0/labels"])[b:a, b:a, :c])
    crop_wsdt_z1_labels,_,_ = vigra.analysis.relabelConsecutive(np.array(h["z/1/labels"])[b:a, b:a, :c])

    vigra.writeHDF5(crop_wsdt_z0_labels,path + "erstellt/cremi.splB.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5","z/0/labels")
    vigra.writeHDF5(crop_wsdt_z1_labels,path + "erstellt/cremi.splB.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5","z/1/labels")



    i = h5py.File(path + "cremi.splB.train.mcseg_betas.crop.axes_xyz.split_z.h5",mode="r")

    for matrix in i["z/0"]:
        crop_mcseg_z0= i["z/0/" + matrix][b:a, b:a, :c]
        vigra.writeHDF5(crop_mcseg_z0,path + "erstellt/cremi.splB.train.mcseg_betas.crop.axes_xyz.split_z.h5","z/0/" + matrix)

    for matrix in i["z/1"]:
        crop_mcseg_z0= i["z/1/" + matrix][b:a, b:a, :c]
        vigra.writeHDF5(crop_mcseg_z0,path + "erstellt/cremi.splB.train.mcseg_betas.crop.axes_xyz.split_z.h5","z/1/" + matrix)
