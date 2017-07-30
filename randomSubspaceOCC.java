/*package vision.src;

//Code Developed by Shehroz S Khan, University of Waterloo, March 2012 - Copyright Protected

import java.util.ArrayList;
import java.util.Collections;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import weka.core.SelectedTag;

class RandomNumberGenerator {

	ArrayList<Integer> numbersList;

	public RandomNumberGenerator(int length) 
	{
		try{
			numbersList = new ArrayList<Integer>();
		}
		catch(Exception e)
		{			
		}
		for(int x=0;x<length;x++)
			numbersList.add(x);
		Collections.shuffle(numbersList);
		for(int x=0;x<length;x++)
			System.out.print(numbersList.get(x)+",");
		System.out.println();
		//Collections.sort(numbersList);
	}

	public int generateNewRandom(int n) {
		return (Integer) numbersList.get(n);
	}
	public ArrayList<Integer> generateNewRandomArray(int max){
		ArrayList<Integer> temp;
		temp = new ArrayList<Integer>();
		for(int i=0;i<max;i++){
			temp.add(numbersList.get(i));
		}
		Collections.sort(temp);
		return temp;			
	}

	public ArrayList<Integer> getTrimmedArray(int count){
		ArrayList<Integer> temp = new ArrayList<Integer>();
		for(int i=0;i<count;i++)
			temp.add(numbersList.get(i));
		Collections.sort(temp);
		return temp;
	}

	public void randomizeArray(){
		Collections.shuffle(numbersList);
	}
}

public class randomSubspaceOCC extends crossValidationOCC{

	private int[][] randomSubspaceValues;
	private int votes;
	
	public randomSubspaceOCC() {
	}

	public randomSubspaceOCC(int sz, int v, String s){
		setFileName(s);
		try{
			readData(s);
		}
		catch(Exception e)
		{
			System.err.print(e.toString());
			System.exit(0);
		}
		System.out.println(data.numAttributes()+" "+data.numInstances());

		setRandomSubspaces(sz,v);
	}

	public void setRandomSubspaces(int size, int votes){
		this.votes = votes;
		System.out.println(data.numAttributes()+" "+size+" "+votes);
		randomSubspaceValues = new int[votes][data.numAttributes()-1-size];
		RandomNumberGenerator rg = new RandomNumberGenerator(data.numAttributes() - 1);
		//ArrayList<Integer> delAttribute = rg.generateNewRandomArray(data.numAttributes() - 1 - subspace_size);
		for(int i=0;i<votes;i++){
			rg.randomizeArray();
			ArrayList<Integer> al = rg.getTrimmedArray(data.numAttributes() -1 - size);
			for(int j=al.size()-1;j >= 0;j--){
				randomSubspaceValues[i][al.size() -1 - j] = al.get(j);
			}
		}

		for(int i=0;i<votes;i++){
			System.out.print("Vote:"+i+"|");
			for(int j=0;j<randomSubspaceValues[i].length;j++){
				System.out.print(randomSubspaceValues[i][j] + ",");
			}
			System.out.println();
		}
	}

	public int [] setTestDataPerPartition(int num){
		int partitions=getCVfoldsOuter();
		int [] data_per_fold = new int [partitions];
		for(int i=0;i<partitions;i++)
			data_per_fold[i] = (int) num/partitions;
		int extra_data = num%partitions;
		//Distribute extra data evenly till they are all consumed
		for(int i=0;i<extra_data;i++) 
			data_per_fold[i]++;

		for(int i=0;i<partitions;i++)
			System.out.print(data_per_fold[i]+" ");
		System.out.println();
		return data_per_fold;
	}

	//OCC-JKNN for CV with best JNN and KNN
	public double OCCJKNNRsub(Instances trainSet, Instances testSet,Instances outlierSet, int JNN, int KNN) throws Exception{

		PolyKernelJKNN poly1 = new PolyKernelJKNN();
		poly1.setPower(getDegree()); //Polynomial of degree 'P'

		int TP = 0;
		int FN = 0;
		int TN = 0;
		int FP = 0;

		int targetVotes [] = new int [testSet.numInstances()];
		int outlierVotes [] = new int [outlierSet.numInstances()];

		for(int rs=0;rs<votes;rs++){
			//System.out.println("OCCJKNN Vote No." + rs);
			Instances newTestSet = new Instances(testSet);
			Instances newTrainSet = new Instances(trainSet);
			Instances newOutlierSet = new Instances(outlierSet);
			for(int rm=0;rm <randomSubspaceValues[rs].length;rm++)
			{
				//System.out.println("Deleting"+randomSubspaceValues[rs][rm]);
				newTestSet.deleteAttributeAt(randomSubspaceValues[rs][rm]);
				newTrainSet.deleteAttributeAt(randomSubspaceValues[rs][rm]);
				newOutlierSet.deleteAttributeAt(randomSubspaceValues[rs][rm]);
			}
			for(int t=0; t<testSet.numInstances();t++) { //For every testset
				poly1.setaccept();
				poly1.setreject();
				poly1.computeKNNmetric(newTrainSet, newTestSet.instance(t), JNN, KNN);
				//System.out.println(t + "/" + newTestSet.numInstances() + "//" + targetVotes[0].length);
				targetVotes[t] += poly1.getaccept();
			}

			for(int t=0; t<outlierSet.numInstances();t++) { //For every outlierset
				poly1.setaccept();
				poly1.setreject();
				poly1.computeKNNmetric(newTrainSet, newOutlierSet.instance(t), JNN, KNN);
				outlierVotes[t] += poly1.getaccept();
			}

		}

		for(int t=0; t<testSet.numInstances();t++) { //For every testset
			if((float)targetVotes[t]/votes >= 0.5)
				TP++;
			else
				FN++;
		}

		for(int t=0; t<outlierSet.numInstances();t++) { //For every outlier
			if((float)outlierVotes[t]/votes >= 0.5)
				FP++;
			else
				TN++;
		}

		double accuracy=(double)(TP+TN)/(TP+FN+FP+TN);
		
		return accuracy;
	} //end of OCCJKNNRsub

	
	//Rsub-OSVM for CV
	public double OSVMRsub(Instances trainSet, Instances testSet,Instances outlierSet, double bestGamma, double bestNu) throws Exception {
		int TP = 0;
		int FN = 0;
		int TN = 0;
		int FP = 0;
		//double accuracy [] = new double [super.getGamma().length];

		LibSVM osvm = new LibSVM();
		//Set svm type to OSVM
		SelectedTag svmType = new SelectedTag(2, LibSVM.TAGS_SVMTYPE);
		osvm.setSVMType(svmType);
		//Set kernel type to RBF
		SelectedTag kernelType = new SelectedTag(super.getKernel(),LibSVM.TAGS_KERNELTYPE);
		osvm.setKernelType(kernelType);
		
		if(getKernel() == 1) { //if Polynomial kernel
			osvm.setDegree(super.getDegree()); //set degree
		}
		else if(getKernel()==2) { //if Gaussian kernel
			osvm.setShrinking(true);
		}
		
		osvm.setGamma(bestGamma);
		osvm.setNu(bestNu);

		int targetVotes [] = new int [testSet.numInstances()];
		int outlierVotes [] = new int [outlierSet.numInstances()];

		for(int rs=0;rs<votes;rs++){
			//System.out.println("OSVM Vote No." + rs);
			Instances newTestSet = new Instances(testSet);
			Instances newTrainSet = new Instances (trainSet);
			Instances newOutlierSet = new Instances(outlierSet);
			for(int rm=0;rm <randomSubspaceValues[rs].length;rm++)
			{
				newTestSet.deleteAttributeAt(randomSubspaceValues[rs][rm]);
				newTrainSet.deleteAttributeAt(randomSubspaceValues[rs][rm]);
				newOutlierSet.deleteAttributeAt(randomSubspaceValues[rs][rm]);
			}

			newTrainSet=adjustClassAttribute(newTrainSet);
			newTestSet=adjustClassAttribute(newTestSet);
			newOutlierSet=adjustClassAttribute(newOutlierSet);

			osvm.buildClassifier(newTrainSet);
			Evaluation eval = new Evaluation (newTrainSet);
			for(int k=0;k<newTestSet.numInstances();k++){
				double res = eval.evaluateModelOnce(osvm, newTestSet.instance(k));
				if(res == 0)
					targetVotes[k]++;
				}

			for(int k=0;k<newOutlierSet.numInstances();k++){
				double res = eval.evaluateModelOnce(osvm, newOutlierSet.instance(k));
				if(res == 0)
					outlierVotes[k]++;
			}
		} //end for votes

		for(int k=0; k<testSet.numInstances();k++) { //For every testset
			if((double)targetVotes[k]/votes >= 0.5)
				TP++;
			else
				FN++;
		}

		for(int k=0; k<outlierSet.numInstances();k++) { //For every outlier
			if((double)outlierVotes[k]/votes >= 0.5)
				FP++;
			else
				TN++;
		}	

		double accuracy=(double)(TP+TN)/(TP+FN+FP+TN);
		return accuracy;
	} //end of OSVMRsub

}//End of class
*/