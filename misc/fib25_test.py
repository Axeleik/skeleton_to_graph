import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from vigra.impex import readHDF5,writeHDF5
import vigra



def compute_real_names_for_blocks(path_to_res,resolved=False):

    dict_blocks = {
        "block1": "x_5000_5520_y_2000_2520_z_3000_3520",
        "block2": "x_5000_5520_y_2000_2520_z_3480_4000",
        "block3": "x_5000_5520_y_2480_3000_z_3000_3520",
        "block4": "x_5480_6000_y_2000_2520_z_3000_3520",
        "block5": "x_5480_6000_y_2480_3000_z_3000_3520",
        "block6": "x_5480_6000_y_2000_2520_z_3480_4000",
        "block7": "x_5000_5520_y_2480_3000_z_3480_4000",
        "block8": "x_5480_6000_y_2480_3000_z_3480_4000",
    }
    for key in dict_blocks.keys():

        if resolved==True:
            block = readHDF5(path_to_res + key + "/result_resolved.h5", "data")
            writeHDF5(block, path_to_res + "finished_renamed/" + "result_resolved_"+dict_blocks[key]+".h5", "data")


        else:
            block=readHDF5(path_to_res+key+"/result.h5","data")
            writeHDF5(block, path_to_res + "finished_renamed/" + "result_"+dict_blocks[key]+".h5", "data")

def compute_names(path,new_path,prefix_old,prefix_new):

    dict_blocks={
    "1": "x_5000_5520_y_2000_2520_z_3000_3520",
    "2": "x_5000_5520_y_2000_2520_z_3480_4000",
    "3": "x_5000_5520_y_2480_3000_z_3000_3520",
    "4": "x_5480_6000_y_2000_2520_z_3000_3520",
    "5": "x_5480_6000_y_2480_3000_z_3000_3520",
    "6": "x_5480_6000_y_2000_2520_z_3480_4000",
    "7": "x_5000_5520_y_2480_3000_z_3480_4000",
    "8": "x_5480_6000_y_2480_3000_z_3480_4000",
    }

    for key in dict_blocks.keys():

            block = readHDF5(path + prefix_old + key + ".h5", "data")
            writeHDF5(block, new_path + prefix_new + dict_blocks[key]+ ".h5", "data",compression="gzip")


#FOR RESULTS IN FOLDERS
def compute_names_of_results_in_folders(path,new_path,name,new_name):

    dict_blocks={
    "block1": "x_5000_5520_y_2000_2520_z_3000_3520",
    "block2": "x_5000_5520_y_2000_2520_z_3480_4000",
    "block3": "x_5000_5520_y_2480_3000_z_3000_3520",
    "block4": "x_5480_6000_y_2000_2520_z_3000_3520",
    "block5": "x_5480_6000_y_2480_3000_z_3000_3520",
    "block6": "x_5480_6000_y_2000_2520_z_3480_4000",
    "block7": "x_5000_5520_y_2480_3000_z_3480_4000",
    "block8": "x_5480_6000_y_2480_3000_z_3480_4000",
    }

    for key in dict_blocks.keys():

            block = readHDF5(path + key +"/" + name +".h5", "data")
            writeHDF5(block, new_path + new_name + "_" + dict_blocks[key]+ ".h5", "data",compression="gzip")



def load_big_cut_out_subvolume(path,save_path,boundaries=np.s_[5000:6000,2000:3000,3000:4000],relabel=False):

    with h5py.File(path) as f:
        subvolume = f["data"][boundaries]

    print "loading from",path

    if relabel:
        labeled_volume = vigra.analysis.labelVolume(subvolume.astype("uint32"))
        print "--> RELABELED"
        print "saving subvolume to", save_path

        writeHDF5(labeled_volume,save_path,"data",compression="gzip")

    else:
        print "saving subvolume to", save_path

        writeHDF5(subvolume,save_path,"data",compression="gzip")

def load_subvolume_and_split_in_blocks(path_to_gt,save_path_blocks,prefix="prefix"):


    dict_blocks = {
        "block1": "x_5000_5520_y_2000_2520_z_3000_3520",
        "block2": "x_5000_5520_y_2000_2520_z_3480_4000",
        "block3": "x_5000_5520_y_2480_3000_z_3000_3520",
        "block4": "x_5480_6000_y_2000_2520_z_3000_3520",
        "block5": "x_5480_6000_y_2480_3000_z_3000_3520",
        "block6": "x_5480_6000_y_2000_2520_z_3480_4000",
        "block7": "x_5000_5520_y_2480_3000_z_3480_4000",
        "block8": "x_5480_6000_y_2480_3000_z_3480_4000",
    }



    # subvolume = readHDF5(path_to_subvolume, "data")


    for key in dict_blocks.keys():
        name = dict_blocks[key]

        with h5py.File(path_to_gt) as f:
            subvolume = f["segmentation"][int(name[2:6]):int(name[7:11]),
                    int(name[14:18]):int(name[19:23]),
                    int(name[26:30]):int(name[31:35])]


            labeled_volume=vigra.analysis.labelVolume(subvolume.astype("uint32"))

            writeHDF5(labeled_volume, save_path_blocks + prefix + key + ".h5", "data",compression="gzip")



def save_comp(path,save_path):
    files = os.listdir(path)

    print files

    for i,file in enumerate(files):

        print file

        block = readHDF5(path + file, "data")
        writeHDF5(block,save_path + file, "data", compression="gzip")



def distribute_folders(path, prefix,new_path,new_name):
    block_files = os.listdir(path)

    for file in block_files:
        number = file[len(prefix):-3]

        block=readHDF5(path+file,"data" )
        writeHDF5(block,new_path+"block{}/".format(number)+ new_name,"data" )


def rerun_connected_compts(path,new_path):

    block_files = os.listdir(path)

    print block_files
    for file in block_files:

        print file

        block=readHDF5(path+file,"data" )
        block=block.astype("uint32")
        labeled_volume = vigra.analysis.relabelVolume(block)
        writeHDF5(labeled_volume,new_path+file,"data",compression="gzip")

def rerun_connected_compts_in_folders(path,file_name,new_file_name):


    for i in [1]:

        print i
        block=readHDF5(path+"block{}/".format(i)+file_name,"data" )
        block=block.astype("uint32")
        labeled_volume = vigra.analysis.labelVolume(block)
        writeHDF5(labeled_volume,path+"block{}/".format(i)+new_file_name,"data",compression="gzip")


def load_train_subvolume_and_split_in_blocks(path, save_path_blocks,new_prefix="new_prefix_",relabel=False):
    """
    4000:5000, 2000:3000, 4000:4500
    """


    train_blocks = {
        "fm_train_block1": "x_4000_4500_y_2000_2500_z_4000_4500",
        "fm_train_block2": "x_4000_4500_y_2500_3000_z_4000_4500",
        "fm_train_block3": "x_4500_5000_y_2000_2500_z_4000_4500",
        "fm_train_block4": "x_4500_5000_y_2500_3000_z_4000_4500",
    }

    new_path=os.path.join(save_path_blocks,new_prefix[:-1])

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    for key in train_blocks.keys():

        name = train_blocks[key]
        print new_prefix+key
        with h5py.File(path) as f:
            subvolume = f["data"][int(name[2:6]):int(name[7:11]),
                        int(name[14:18]):int(name[19:23]),
                        int(name[26:30]):int(name[31:35])]

            if relabel:

                labeled_volume = vigra.analysis.labelVolume(subvolume.astype("uint32"))
                print "-->RELABELED"
                writeHDF5(labeled_volume, new_path + "/" + new_prefix + key + ".h5", "data", compression="gzip")

            else:

                writeHDF5(subvolume, new_path + "/" + new_prefix + key + ".h5", "data", compression="gzip")

def relabelcons(path,new_path):
    files_list=os.listdir(path)

    for file in files_list:

        block=readHDF5(path+file,"data" )
        block=block.astype("uint32")
        labeled_volume,_,_ = vigra.analysis.relabelConsecutive(block)
        writeHDF5(labeled_volume,new_path+file,"data",compression="gzip")




if __name__ == "__main__":


    path="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/PAPER/renamed_and_stiched/"
    new_path="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/for_figure/"
    path_names=["resolved_local_alba/","oracle_local_alba/","gt/","init_seg_alba/"]
    new_path_names=["res/","oracle/","gt/","init/"]

    for i,_ in enumerate(path_names):

        np=new_path+new_path_names[i]
        op=path+path_names[i]
        relabelcons(op,np)


    gt_big=readHDF5("/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/gt/"
                    "gt_x_5000_6000_y_2000_3000_z_3000_4000.h5","data")

    # gt_big_relabeled=vigra.analysis.labelVolume(gt_big.astype("uint32"))

    # writeHDF5(gt_big_relabeled,"/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/gt/"
    #                 "gt_x_5000_6000_y_2000_3000_z_3000_4000.h5","data",compression="gzip")





    path_resolved_alba="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/" \
                       "PAPER/blocks/fib25_FINAL_albatross_resolving/"

    path_renamed_resolved_alba="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/" \
                             "PAPER/renamed_and_stiched/resolved_local_alba/"

    name_resolved_alba="result_resolved_local"

    name_renamed_resolved_alba="resolvedLocal"



    path_oracle_alba="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/" \
                     "PAPER/blocks/fib25_FINAL_alba_oracle/"

    path_renamed_oracle_alba="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/" \
                             "PAPER/renamed_and_stiched/oracle_local_alba/"

    name_oracle_alba="result_resolved_local"

    name_renamed_oracle_alba="oracleLocal"



    path_init_alba="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/" \
                       "PAPER/blocks/fib25_FINAL_albatross_resolving/"

    path_renamed_init_alba="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/" \
                             "PAPER/renamed_and_stiched/init_seg_alba/"

    name_init_alba="result"

    name_renamed_init_alba="initResult"




    path_gt="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/" \
                       "PAPER/blocks/gt_blocks/"

    path_renamed_gt="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/" \
                             "PAPER/renamed_and_stiched/gt/"

    name_gt="gt"

    name_renamed_gt="gt"



    # compute_names_of_results_in_folders(path_gt,path_renamed_gt,
    #                                     name_gt,name_renamed_gt)



    # compute_names_of_results_in_folders(path_resolved_alba,path_renamed_resolved_alba,
    #                                     name_resolved_alba,name_renamed_resolved_alba)
    # compute_names_of_results_in_folders(path_oracle_alba,path_renamed_oracle_alba,
    #                                     name_oracle_alba,name_renamed_oracle_alba)
    # compute_names_of_results_in_folders(path_init_alba,path_renamed_init_alba,
    #                                     name_init_alba,name_renamed_init_alba)





    # rerun_connected_compts("/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/gt/gt_blocks/",
    #                        "/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/gt/relabeled/")


    path_to_gt="/mnt/localdata0/amatskev/neuraldata/fib25_gt/fib25-gt.h5"
    path_to_raw="/mnt/localdata1/amatskev/neuraldata/unprocessed/fib25-raw.h5"
    path_to_wtsd="/mnt/localdata1/amatskev/neuraldata/unprocessed/fib25-vanilla-watershed-relabeled.h5"
    path_to_pmaps="/mnt/localdata1/amatskev/neuraldata/unprocessed/fib25-membrane-predictions_squeezed.h5"
    save_path_train_blocks="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/fib_25_fmerge_train_blocks/"
    boundaries = np.s_[4000:5000, 2000:3000, 4000:4500]

    # load_train_subvolume_and_split_in_blocks(path_to_gt,save_path_train_blocks,"gt_",True)
    # load_train_subvolume_and_split_in_blocks(path_to_raw,save_path_train_blocks,"raw_")
    # load_train_subvolume_and_split_in_blocks(path_to_wtsd,save_path_train_blocks,"wtsd_",True)
    # load_train_subvolume_and_split_in_blocks(path_to_pmaps,save_path_train_blocks,"pmap_")
    # load_big_cut_out_subvolume(path_to_gt,save_path_train_blocks+"gt_x_4000_5000_y_2000_3000_z_4000_4500.h5",boundaries,True)
    # load_big_cut_out_subvolume(path_to_wtsd,save_path_train_blocks+"wtsd_x_4000_5000_y_2000_3000_z_4000_4500.h5",boundaries,True)
    # load_big_cut_out_subvolume(path_to_raw,save_path_train_blocks+"raw_x_4000_5000_y_2000_3000_z_4000_4500.h5",boundaries)
    # load_big_cut_out_subvolume(path_to_pmaps,save_path_train_blocks+"pmap_x_4000_5000_y_2000_3000_z_4000_4500.h5",boundaries)



    path_big_raw_volume="/mnt/localdata0/amatskev/neuraldata/fib25_gt/fib25-gt.h5"
    new_save_path_subvolume="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/fib_25_fmerge_train_blocks/gt_x_4000_5000_y_2000_3000_z_4000_4500.h5"
    boundaries = np.s_[4000:5000, 2000:3000, 4000:4500]

    # load_big_cut_out_subvolume(path_big_raw_volume,new_save_path_subvolume,boundaries)



    published_blocks_path="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/neuroproof_published/blocks/"
    names_published_blocks="neuroproof_published.h5"
    names_published_blocks_new="neuroproof_published_relabeled.h5"

    # rerun_connected_compts_in_folders(published_blocks_path,names_published_blocks,names_published_blocks_new)


    gt_path="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/gt/gt_blocks/"
    # rerun_connected_compts(gt_path,"/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/gt/blocks_relabeled_again/")


    rf_path_old="/mnt/localdata1/amatskev/neuraldata/constantins_res/rf/"
    prefix_rf_old="res_rfpmap_"
    name_rf_new="rfconst.h5"

    cantor_path_old="/mnt/localdata1/amatskev/neuraldata/constantins_res/cantor/"
    prefix_cantor_old="res_cantorpmap_"
    name_cantor_new="cantorconst.h5"

    new_path="/mnt/localdata1/amatskev/neuraldata/constantins_res/blocks_eval/"



    # distribute_folders(rf_path_old,prefix_rf_old,new_path,name_rf_new)
    #
    # distribute_folders(cantor_path_old,prefix_cantor_old,new_path,name_cantor_new)


    rf_path_new="/mnt/localdata1/amatskev/neuraldata/constantins_res/rf_renamed/"
    cantor_path_new="/mnt/localdata1/amatskev/neuraldata/constantins_res/cantor_renamed/"
    prefix_cantor_new="cantorconst_"

    prefix_rf_new="rfconst_"
    # compute_names(rf_path_old,rf_path_new,prefix_rf_old,prefix_rf_new)
    # compute_names(cantor_path_old,cantor_path_new,prefix_cantor_old,prefix_cantor_new)


    uncomp="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/fib_25_blocks/"
    comp="/net/hci-storage02/userfolders/amatskev/neuraldata/fib25_hdf5/fib_25_blocks/"
    # save_comp(uncomp,comp)


    neuroproof_published_path="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/" \
                              "neuroproof_published/neuroproof-fib25seg-med.h5"
    neuroproof_published_blocks_path="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/" \
                              "neuroproof_published/blocks/"
    neoroproof_published_subvolume_path="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/" \
                              "neuroproof_published/neuroproof_published_subvolume.h5"

    # load_big_cut_out_subvolume(neuroproof_published_path,neoroproof_published_subvolume_path)
    #
    # load_subvolume_and_split_in_blocks(neuroproof_published_path,neuroproof_published_blocks_path,
    #                                    "neuroproof_published_")


    path_to_subvol="/mnt/localdata0/amatskev/neuraldata/fib25_gt/subvolume_x_5000_6000_y_2000_3000_z_3000_4000.h5"
    path_to_blocks="/mnt/localdata0/amatskev/neuraldata/fib25_gt/blocks_unnamed/"
    path_to_results="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/results_fib25/"
    path_to_renamed="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/results_fib25/renamed_result_resolved/"

    prefix="gt_"

    # compute_names_of_results_in_folders(path_to_results,path_to_renamed,"result_resolved_local")

    # load_big_cut_out_subvolume(path_to_gt,path_to_subvol)
    # load_subvolume_and_split_in_blocks(path_to_gt,path_to_blocks,prefix)
    # labeled_volume=vigra.analysis.labelVolume(volume.astype("uint32"))




    path="/mnt/localdata0/amatskev/neuraldata/fib25_gt/fib25-gt.h5"
    new_path="/mnt/localdata0/amatskev/neuraldata/fib25_gt/subvolume_x_5000_6000_y_2000_3000_z_3000_4000.h5"
    # new_path="/net/hci-storage02/userfolders/amatskev/neuraldata/finished_renamed/"
    # load_big_cut_out_subvolume(path,new_path)

    # compute_names(path,new_path,name)

    """
    
    fib25_filepath="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/fib_25_blocks/"
    train_raw=readHDF5(fib25_filepath+"raw_train.h5","data")
    test_raw=readHDF5(fib25_filepath+"raw_test.h5","data")

    train_raw=train_raw*255
    test_raw=test_raw*255

    train_raw=train_raw.astype("uint8")
    test_raw=test_raw.astype("uint8")

    writeHDF5(train_raw,fib25_filepath+"raw_train_normalized.h5","data")
    writeHDF5(test_raw, fib25_filepath + "raw_test_normalized.h5", "data")






    cremi_filepath="/mnt/ssd/jhennies/neuraldata/cremi_2016/170606_resolve_false_merges/"

    names=["RAW","PROBS","WATERSHED","GT"]

    test=[  readHDF5(fib25_filepath+"raw_test_normalized.h5","data"),readHDF5(fib25_filepath+"probabilities_test.h5","data"),
            readHDF5(fib25_filepath+"overseg_test.h5","data"),readHDF5(fib25_filepath+"gt_test.h5","data") ]
    train=[ readHDF5(fib25_filepath+"raw_train_normalized.h5","data"),readHDF5(fib25_filepath+"probabilities_train.h5","data"),
            readHDF5(fib25_filepath+"overseg_train.h5","data"),readHDF5(fib25_filepath+"gt_train.h5","data") ]

    block1=[readHDF5(fib25_filepath+"raw_block1.h5","data"),readHDF5(fib25_filepath+"probs_cantor_block1.h5","data"),
            readHDF5(fib25_filepath+"watershed_block1.h5","data"),None ]


    cremi_Az0=[ readHDF5(cremi_filepath+"cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5","z/0/raw"),
                readHDF5(cremi_filepath+"cremi.splA.train.probs.crop.axes_xyz.split_z.h5","z/0/data"),
                readHDF5(cremi_filepath+"cremi.splA.train.wsdt_relabel.crop.axes_xyz.split_z.h5","z/0/labels"),
                readHDF5(cremi_filepath+"cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5","z/0/neuron_ids") ]

    print "LOADED"
    print "-------"



    for idx,name in enumerate(names):
        print "------------------------"
        print name,": "

        print "test"
        print "type:",type(test[idx])
        print "dtype:",test[idx].dtype
        print "shape:",test[idx].shape
        print "min:",np.min(test[idx])
        print "max:",np.max(test[idx])
        print " "

        print "train"
        print "type:",type(train[idx])
        print "dtype:",train[idx].dtype
        print "shape:",train[idx].shape
        print "min:",np.min(train[idx])
        print "max:",np.max(train[idx])
        print " "

        if block1[idx] is not None:

            print "block1"
            print "type:",type(block1[idx])
            print "dtype:", block1[idx].dtype
            print "shape:",block1[idx].shape
            print "min:",np.min(block1[idx])
            print "max:",np.max(block1[idx])
            print " "

        print "cremi"
        print "type:",type(cremi_Az0[idx])
        print "dtype:",cremi_Az0[idx].dtype
        print "shape:",cremi_Az0[idx].shape
        print "min:",np.min(cremi_Az0[idx])
        print "max:",np.max(cremi_Az0[idx])
        print " "










    folderpath="/mnt/localdata1/amatskev/neuraldata/fib25_hdf5/numpy_images/"

    files=["fib25-raw_5-6_2-3_3-4.npy","fib25-membrane-predictions_squeezed_5-6_2-3_3-4.npy",
           "fib25-vanilla-watershed-relabeled_5-6_2-3_3-4.npy","fib25-cantor-pmap_5-6_2-3_3-4.npy"]

    output_folders=["raw/","probs_squeezed/","watershed/","probs_cantor/"]

    blocks=["block{}".format(i) for i in xrange(1,9)]
    print blocks

    for spez_folder in output_folders:

        for block in blocks:

            array=np.load(folderpath+spez_folder+block+".npy")
            writeHDF5(array,folderpath+"final/"+spez_folder[:-1]+"_"+block+".h5","data")







    """
    # for index,subvolume_file in enumerate(files):
    #
    #     subvolume_path=folderpath+subvolume_file
    #     subvolume = np.load(subvolume_path)
    #
    # 
    #     np.save(folderpath+output_folders[index]+"block1.npy",subvolume[0:520,0:520,0:520])
    #     np.save(folderpath+output_folders[index]+"block2.npy",subvolume[0:520,0:520,480:1000])
    #     np.save(folderpath+output_folders[index]+"block3.npy",subvolume[0:520,480:1000,0:520])
    #     np.save(folderpath+output_folders[index]+"block4.npy",subvolume[480:1000,0:520,0:520])
    #     np.save(folderpath+output_folders[index]+"block5.npy",subvolume[480:1000,480:1000,0:520])
    #     np.save(folderpath+output_folders[index]+"block6.npy",subvolume[480:1000,0:520,480:1000])
    #     np.save(folderpath+output_folders[index]+"block7.npy",subvolume[0:520,480:1000,480:1000])
    #     np.save(folderpath+output_folders[index]+"block8.npy",subvolume[480:1000,480:1000,480:1000])


        





