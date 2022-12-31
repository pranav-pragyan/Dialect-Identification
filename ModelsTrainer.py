import os
import pickle
import warnings
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from FeaturesExtractor import FeaturesExtractor

warnings.filterwarnings("ignore")


class ModelsTrainer:

    def __init__(self, bhojpuri_files_path, mathili_files_path, rajsthani_files_path):
        self.bhojpuri_training_path = bhojpuri_files_path
        self.mathili_training_path = mathili_files_path
        self.rajsthani_training_path = rajsthani_files_path
        self.features_extractor = FeaturesExtractor()

    def process(self):
        bhojpuri, mathili, rajsthani = self.get_file_paths(
            self.bhojpuri_training_path, self.mathili_training_path, self.rajsthani_training_path)

        # collect voice features
        bhojpuri_voice_features = self.collect_features(bhojpuri)
        mathili_voice_features = self.collect_features(mathili)
        rajsthani_voice_features = self.collect_features(rajsthani)

        # generate gaussian mixture models
        bhojpuri_gmm = GMM(n_components=16, max_iter=200,
                           covariance_type='full', n_init=3)
        mathili_gmm = GMM(n_components=16, max_iter=200,
                          covariance_type='full', n_init=3)
        rajsthani_gmm = GMM(n_components=16, max_iter=200,
                            covariance_type='full', n_init=3)

        # fit features to models
        bhojpuri_gmm.fit(bhojpuri_voice_features)
        mathili_gmm.fit(mathili_voice_features)
        rajsthani_gmm.fit(rajsthani_voice_features)

        # save models
        self.save_gmm(bhojpuri_gmm, "bhojpuri")
        self.save_gmm(mathili_gmm, "mathili")
        self.save_gmm(rajsthani_gmm, "rajsthani")

    def get_file_paths(self, bhojpuri_training_path, mathili_training_path, rajsthani_training_path):
        # get file paths
        bhojpuri = [os.path.join(bhojpuri_training_path, f)
                    for f in os.listdir(bhojpuri_training_path)]
        mathili = [os.path.join(mathili_training_path, f)
                   for f in os.listdir(mathili_training_path)]
        rajsthani = [os.path.join(rajsthani_training_path, f)
                     for f in os.listdir(rajsthani_training_path)]
        return bhojpuri, mathili, rajsthani

    def collect_features(self, files):
        features = np.asarray(())
        # extract features for each dialect
        for file in files:
            print("%5s %10s" % ("PROCESSNG ", file))
            # extract MFCC & delta MFCC features from audio
            vector = self.features_extractor.extract_features(file)
            # stack the features
            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))
        return features

    def save_gmm(self, gmm, name):
        """ Save Gaussian mixture model using pickle.
            Args:
                gmm        : Gaussian mixture model.
                name (str) : File name.
        """
        filename = name + ".gmm"
        with open(filename, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
        print("%5s %10s" % ("SAVING", filename,))


if __name__ == "__main__":
    models_trainer = ModelsTrainer(
        "training/bhojpuri", "training/mathili", "training/rajsthani")
    models_trainer.process()
