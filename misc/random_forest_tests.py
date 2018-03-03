from sklearn.ensemble import RandomForestClassifier
import cPickle as pickle
import numpy as np



if __name__ == "__main__":
    """
    self.rf = RFType(
                n_estimators=self.n_trees,
                n_jobs=self.n_threads,
                verbose=2 if ExperimentSettings().verbose else 0,
                max_depth=self.max_depth,
                class_weight=self.class_weight
            )
            self.rf.fit(train_data, train_labels)
            
            [:, 1]

    """
    main_location="/mnt/localdata0/amatskev/neuraldata/results/cache/train_path_data/rf_paths/"


    features_train = np.load(main_location+"features_train.npy")
    labels_train = np.load(main_location+ "labels_train.npy")
    features_test = np.load(main_location+ "features_test.npy")
    labels_test = np.load(main_location + "labels_test.npy")

    rf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=30,
        verbose=0,
        max_depth=None,
        class_weight=None
    )
    rf.fit(features_train, labels_train)

    predicted_probs=rf.predict_proba(features_test)[:, 1]

    predicted_labels=np.array([True if prob>0.3 else False for prob in predicted_probs])

    assert len(labels_test)==len(predicted_labels)


    #FILTERING
    positives_prob = predicted_labels == True
    positives_test = labels_test == True
    negatives_prob = predicted_labels == False
    negatives_test = labels_test == False

    #RATES
    true_positives=float(np.count_nonzero(positives_test[positives_prob]==True))
    false_positives=float(np.count_nonzero(positives_test[positives_prob]==False))
    true_negatives=float(np.count_nonzero(negatives_test[negatives_prob]==True))
    false_negatives=float(np.count_nonzero(negatives_test[negatives_prob]==False))

    #RATIOS

    precision=true_positives/(true_positives+false_positives)
    recall=true_positives/(true_positives+false_negatives)
    accuracy=(true_negatives+true_positives)/float(len(predicted_labels))
    f1=2*precision*recall/(precision+recall)


    print "precision(correctly trues in prediction): ",precision
    print "recall(right trues in gt predicted right): ",recall
    print "accuracy(percentage of right predictions): ",accuracy
    print "f1: ",f1

    print "________________"










