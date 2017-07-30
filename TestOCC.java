package vision.src;

//Code Developed by Shehroz S Khan, University of Waterloo, March 2012/16 - Copyright Protected

//This program computes the best parameters  for OCCJKNN using N-fold CV

public class TestOCC {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		//merge_datasets();
		//Input file name
		String inputFile = "//home//shehroz//workspace//OCC//src//vision//resources//breast-cancer.arff";

		crossValidationOCC cv = new crossValidationOCC();
		
		//Parameter Setting
		cv.setFileName(inputFile);
		cv.setCVFoldsOuter(3); //Outer Cross validation
		cv.setCVFoldsInner(2); //Inner Cross validation for parameter Optimization
		cv.setRejectionRate(1.5);//1.5 ==> 99.3%, 1.7239 ==> 99.73%, 3==>99.99%
		cv.setMaxNN(5);//Maximum number of nearest neighbours to test
		cv.setNumEnsemble(5);//Number of Ensemble
		//Random Subspace
		int [] subspace= new int []{50, 75};//%age of features to be used as subspace
		cv.setRandomsubspace(subspace);
		//Kernel Features
		//cv.setKernelSigma(10); %This is not included in the current version

		//Training OCC Classifiers
		cv.training();		
	} //end for main
} //End of class