import librosa
import os
import json

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050 # number of samples per sec


def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):

    # dictionary to store the data
    data = {
        "mapping": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:

            label = dirpath.split("/")[-1]
            data["mapping"].append(label)
            print("\nProcessing: '{}'".format(label))

            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # load audio file
                signal, sample_rate = librosa.load(file_path)

                # need only 1sec to our CNN network
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    # get 1sec
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # extract MFCCs of 1sec
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate,n_mfcc=num_mfcc,hop_length=hop_length,n_fft=n_fft)
#                                                   n_mfcc=num_mfcc, n_fft=n_fft,
#                                                  hop_length=hop_length)

                    # store the data
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i-1))

    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
