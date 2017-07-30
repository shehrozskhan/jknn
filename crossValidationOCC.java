package vision.src;

//Code Developed by Shehroz S Khan, University of Waterloo, March 2012/16 - Copyright Protected

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.experiment.Stats;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

class crossValidationOCC {

	private String fileName;
	private String targetClassName;
	private boolean ensemble;
	private int foldsOuter; //Number of Outer CV folds
	private int foldsInner; //Number of Inner CV folds
	private int maxNN; //Number of maximum nearest J and K neighbours
	private int bestJEuc;
	private int bestKEuc;
	private int numEnsemble;
	private int [] subspace;
	private int [] subspaceSize;//size of each subspace
	private int [] testIndex; //index of testing data objects
	private int [] trainIndex; //Index of training data objects
	private double rsubTpr;
	private double rsubTnr;
	private double rsubPrecision;
	private double rsubMCC;
	private double rejectionRate; //for rejecting outliers from the target data for optimizing parameters
	private double empiricalTheta;
	private double [] tprEuc;
	private double [] tprEuc11;
	private double [] tprEuc11Theta;
	private double [] tnrEuc;
	private double [] tnrEuc11;
	private double [] precisionEuc;
	private double [] precisionEuc11;
	private double [] precisionEuc11Theta;
	private double [] mccEuc;
	private double [] mccEuc11;
	private double [] mccEuc11Theta;
	private double [] tnrEuc11Theta;
	private double [] rprojGmeanEuc;
	private double [] rprojGmeanEuc11;
	private double [] rprojGmeanEuc11Theta;
	private double [] rprojTprEuc;
	private double [] rprojTprEuc11;
	private double [] rprojTprEuc11Theta;
	private double [] rprojTnrEuc;
	private double [] rprojTnrEuc11;
	private double [] rprojTnrEuc11Theta;
	private double [] rprojPrecisionEuc;
	private double [] rprojPrecisionEuc11;
	private double [] rprojPrecisionEuc11Theta;
	private double [] rprojMCCEuc;
	private double [] rprojMCCEuc11;
	private double [] rprojMCCEuc11Theta;
	private double [][] rsubGmeanEuc;
	private double [][] rsubGmeanEuc11;
	private double [][] rsubGmeanEuc11Theta;
	private double [][] rsubTprEuc;
	private double [][] rsubTprEuc11;
	private double [][] rsubTprEuc11Theta;
	private double [][] rsubTnrEuc;
	private double [][] rsubTnrEuc11;
	private double [][] rsubTnrEuc11Theta;
	private double [][] rsubPrecisionEuc;
	private double [][] rsubPrecisionEuc11;
	private double [][] rsubPrecisionEuc11Theta;
	private double [][] rsubMCCEuc;
	private double [][] rsubMCCEuc11;
	private double [][] rsubMCCEuc11Theta;

	private Instances data; //full data
	private Instances target; //target data
	private Instances outlier; //outlier data
	private Instances proxyOutliersTrainSet; //proxy outliers rejected from target
	private ArrayList<String> targetLabels;
	private ArrayList<String> outlierLabels;
	private ArrayList<ArrayList<String>> rsubTargetLabelsEuc;
	private ArrayList<ArrayList<String>> rsubOutlierLabelsEuc;
	private ArrayList<ArrayList<String>> rsubTargetLabelsEuc11;
	private ArrayList<ArrayList<String>> rsubOutlierLabelsEuc11;
	private ArrayList<ArrayList<String>> rsubTargetLabelsEuc11Theta;
	private ArrayList<ArrayList<String>> rsubOutlierLabelsEuc11Theta;
	private ArrayList<ArrayList<String>> rprojTargetLabelsEuc;
	private ArrayList<ArrayList<String>> rprojOutlierLabelsEuc;
	private ArrayList<ArrayList<String>> rprojTargetLabelsEuc11;
	private ArrayList<ArrayList<String>> rprojOutlierLabelsEuc11;
	private ArrayList<ArrayList<String>> rprojTargetLabelsEuc11Theta;
	private ArrayList<ArrayList<String>> rprojOutlierLabelsEuc11Theta;

	public void setEnsemble (boolean e) {
		ensemble=e;
	}

	public boolean getEnsemble(){
		return ensemble;
	}

	public ArrayList<ArrayList<String>> getRprojTargetLabelsEuc() {
		return rprojTargetLabelsEuc;
	}

	public ArrayList<ArrayList<String>> getRprojTargetLabelsEuc11() {
		return rprojTargetLabelsEuc11;
	}

	public ArrayList<ArrayList<String>> getRprojTargetLabelsEuc11Theta() {
		return rprojTargetLabelsEuc11Theta;
	}

	public ArrayList<ArrayList<String>> getRprojOutlierLabelsEuc() {
		return rprojOutlierLabelsEuc;
	}

	public ArrayList<ArrayList<String>> getRprojOutlierLabelsEuc11() {
		return rprojOutlierLabelsEuc11;
	}

	public ArrayList<ArrayList<String>> getRprojOutlierLabelsEuc11Theta() {
		return rprojOutlierLabelsEuc11Theta;
	}

	public ArrayList<ArrayList<String>> getRsubTargetLabelsEuc() {
		return rsubTargetLabelsEuc;
	}

	public ArrayList<ArrayList<String>> getRsubTargetLabelsEuc11() {
		return rsubTargetLabelsEuc11;
	}

	public ArrayList<ArrayList<String>> getRsubTargetLabelsEuc11Theta() {
		return rsubTargetLabelsEuc11Theta;
	}

	public ArrayList<ArrayList<String>> getRsubOutlierLabelsEuc() {
		return rsubOutlierLabelsEuc;
	}

	public ArrayList<ArrayList<String>> getRsubOutlierLabelsEuc11() {
		return rsubOutlierLabelsEuc11;
	}

	public ArrayList<ArrayList<String>> getRsubOutlierLabelsEuc11Theta() {
		return rsubOutlierLabelsEuc11Theta;
	}

	public int [] getRandomsubspace() {
		return subspace;
	}

	public void setRandomsubspace(int [] ss){
		subspace = ss;
	}

	public int [] getSubspaceSize() {
		return subspaceSize;
	}

	private void setSubspaceSize () {
		subspaceSize = new int [getRandomsubspace().length]; 
	}

	public String getFileName() {
		return fileName;
	}

	public void setFileName(String fn) {
		fileName=fn;
	}

	public String getTargetClassName(){
		return targetClassName;
	}

	public void setTargetClassName(String tcn) {
		targetClassName=tcn;
	}

	public int getCVfoldsOuter(){
		return foldsOuter;
	}

	public int getCVfoldsInner(){
		return foldsInner;
	}

	public void setCVFoldsOuter(int f){
		foldsOuter=f;
	}

	public void setCVFoldsInner(int f){
		foldsInner=f;
	}

	public int getMaxNN() {
		return maxNN;
	}

	public void setMaxNN(int n) {
		maxNN=n;
	}

	public int getBestJEuc() {
		return bestJEuc;
	}

	public void setBestJEuc(int jp1) {
		bestJEuc=jp1;
	}

	public void setBestKEuc(int jp1) {
		bestKEuc=jp1;
	}

	public int getBestKEuc() {
		return bestKEuc;
	}

	public int [] getTestSetIndex(){
		return testIndex;
	}

	public int [] getTrainSetIndex() {
		return trainIndex;
	}

	public Instances getFullData(){
		return data;
	}

	public Instances getTargetData() {
		return target;
	}

	private void setTargetData() {
		target = new Instances(data,data.numInstances());
	}

	public int getNumTargetData() {
		return target.numInstances();
	}
	public Instances getOutlierData () {
		return outlier;
	}

	public Instances getProxyOutlierTrainSet() {
		return proxyOutliersTrainSet;
	}

	private void setOutlierData() {
		outlier = new Instances(data,data.numInstances());
	}

	public int getNumOutlierData() {
		return outlier.numInstances();
	}

	public double getRejectionRate(){
		return rejectionRate;
	}

	public void setRejectionRate(double d) {
		rejectionRate=d;
	}

	public double getEmpiricalTheta(){
		return empiricalTheta;
	}

	public void setEmpiricalTheta(double e) {
		empiricalTheta=e;
	}

	public double getRsubTpr() {
		return rsubTpr;
	}

	public void setRsubTpr(double tpr) {
		rsubTpr = tpr;
	}

	public double getRsubTnr() {
		return rsubTnr;
	}

	public void setRsubTnr(double tnr) {
		rsubTnr = tnr;
	}

	public double getRsubPrecision() {
		return rsubPrecision;
	}

	public void setRsubPrecision(double prec) {
		rsubPrecision = prec;
	}

	public double getRsubMCC() {
		return rsubMCC;
	}

	public void setRsubMCC(double prec) {
		rsubMCC = prec;
	}

	public double [][] getRsubTprEuc() {
		return rsubTprEuc;
	}

	public double [][] getRsubTprEuc11() {
		return rsubTprEuc11;
	}

	public double [][] getRsubTprEuc11Theta() {
		return rsubTprEuc11Theta;
	}

	public double [][] getRsubTnrEuc() {
		return rsubTnrEuc;
	}

	public double [][] getRsubTnrEuc11() {
		return rsubTnrEuc11;
	}

	public double [][] getRsubTnrEuc11Theta() {
		return rsubTnrEuc11Theta;
	}

	public double [][] getRsubPrecisionEuc() {
		return rsubPrecisionEuc;
	}

	public double [][] getRsubPrecisionEuc11() {
		return rsubPrecisionEuc11;
	}

	public double [][] getRsubPrecisionEuc11Theta() {
		return rsubPrecisionEuc11Theta;
	}

	public double [][] getRsubMCCEuc() {
		return rsubMCCEuc;
	}

	public double [][] getRsubMCCEuc11() {
		return rsubMCCEuc11;
	}

	public double [][] getRsubMCCEuc11Theta() {
		return rsubMCCEuc11Theta;
	}

	public double [] getRprojTprEuc() {
		return rprojTprEuc;
	}

	public double [] getRprojTprEuc11() {
		return rprojTprEuc11;
	}

	public double [] getRprojTprEuc11Theta() {
		return rprojTprEuc11Theta;
	}

	public double [] getRprojTnrEuc() {
		return rprojTnrEuc;
	}

	public double [] getRprojTnrEuc11() {
		return rprojTnrEuc11;
	}

	public double [] getRprojTnrEuc11Theta() {
		return rprojTnrEuc11Theta;
	}

	public double [] getRprojPrecisionEuc() {
		return rprojPrecisionEuc;
	}

	public double [] getRprojPrecisionEuc11() {
		return rprojPrecisionEuc11;
	}

	public double [] getRprojPrecisionEuc11Theta() {
		return rprojPrecisionEuc11Theta;
	}

	public double [] getRprojMCCEuc() {
		return rprojMCCEuc;
	}

	public double [] getRprojMCCEuc11() {
		return rprojMCCEuc11;
	}

	public double [] getRprojMCCEuc11Theta() {
		return rprojMCCEuc11Theta;
	}

	public double [] getTprEuc() {
		return tprEuc;
	}

	public void setTprEuc(int s) {
		tprEuc = new double [s];
	}

	public double [] getTprEuc11() {
		return tprEuc11;
	}

	public void setTprEuc11(int s) {
		tprEuc11 = new double [s];
	}

	public double [] getTprEuc11Theta() {
		return tprEuc11Theta;
	}

	public void setTprEuc11Theta(int s) {
		tprEuc11Theta = new double [s];
	}

	public double [] getTnrEuc() {
		return tnrEuc;
	}

	public void setTnrEuc(int s) {
		tnrEuc = new double [s];
	}

	public double [] getTnrEuc11() {
		return tnrEuc11;
	}

	public void setTnrEuc11(int s) {
		tnrEuc11 = new double [s];
	}

	public double [] getTnrEuc11Theta() {
		return tnrEuc11Theta;
	}

	public void setTnrEuc11Theta(int s) {
		tnrEuc11Theta = new double [s];
	}

	public double [] getPrecisionEuc() {
		return precisionEuc;
	}

	public void setPrecisionEuc(int s) {
		precisionEuc = new double [s];
	}

	public double [] getPrecisionEuc11() {
		return precisionEuc11;
	}

	public void setPrecisionEuc11(int s) {
		precisionEuc11 = new double [s];
	}

	public double [] getPrecisionEuc11Theta() {
		return precisionEuc11Theta;
	}

	public void setPrecisionEuc11Theta(int s) {
		precisionEuc11Theta = new double [s];
	}

	public double [] getMCCEuc() {
		return mccEuc;
	}

	public void setMCCEuc(int s) {
		mccEuc = new double [s];
	}

	public double [] getMCCEuc11() {
		return mccEuc11;
	}

	public void setMCCEuc11(int s) {
		mccEuc11 = new double [s];
	}

	public double [] getMCCEuc11Theta() {
		return mccEuc11Theta;
	}

	public void setMCCEuc11Theta(int s) {
		mccEuc11Theta = new double [s];
	}

	public int getNumEnsemble() {
		return numEnsemble;
	}

	public void setNumEnsemble(int e) {
		numEnsemble=e;
	}

	public ArrayList<String> getTargetLabels() {
		return targetLabels;
	}

	public void setTargetLabels(ArrayList<String> tl) {
		targetLabels = tl;
	}

	public ArrayList<String> getOutlierLabels() {
		return outlierLabels;
	}

	public void setOutlierLabels(ArrayList<String> ol) {
		outlierLabels = ol;
	}

	private int [] setTestDataPerFold(int num, int folds){
		int [] data_per_fold = new int [folds];
		for(int i=0;i<folds;i++)
			data_per_fold[i] = (int) num/folds;
		int extra_data = num%folds;
		//Distribute extra data evenly till they are all consumed
		for(int i=0;i<extra_data;i++) 
			data_per_fold[i]++;

		//for(int i=0;i<folds;i++)
		//	System.out.print("Per_Fold="+data_per_fold[i]+" ");
		//System.out.println();
		return data_per_fold;
	}

	void generateCVFolds(int foldSize, int N, int size) {
		//for (int i=0;i<folds;i++) {
		boolean flag [] = new boolean [N];
		int m=0;
		testIndex = new int [size];
		trainIndex = new int [N-size];
		for(int j=foldSize;j<foldSize+size;j++){
			testIndex[m]=j;
			flag[j]=true;
			//System.out.print(test[m]+" ");
			m++;
		}
		//System.out.println();
		m=0;
		for(int l=0;l<N;l++) {
			if(flag[l]==false) {
				trainIndex[m]=l;
				//System.out.print(train[m]+" ");
				m++;
			}
		}
		//System.out.println("m="+m);
		foldSize+=size;
	}

	//Reads the data and randomize
	void readData (String str) throws Exception {

		DataSource source = new DataSource(str);
		data = source.getDataSet();
		// setting class attribute if the data format does not provide this information
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		//randomizeData(); 
	}

	//Randomize the data
	/*
	public void randomizeData(){
		Random seed = new Random();
		data.randomize(seed);
	}
	 */

	//Get user input to capture 'target' or positive class
	public String getUserInput() throws IOException {
		BufferedReader reader;
		String str = new String();
		reader = new BufferedReader(new InputStreamReader(System.in));
		System.out.println("Available classes in the data:");
		for(int i=0;i< data.classAttribute().numValues();i++) 
			System.out.println("("+(i+1)+") "+data.classAttribute().value(i));
		while(!str.matches("^\\d+$") || Integer.parseInt(str)  > data.classAttribute().numValues() || Integer.parseInt(str)==0) {
			System.out.print("Enter the (numeric) choice for Positive Class: ");
			str = reader.readLine();
		}

		int choice = Integer.parseInt(str);
		//System.out.println("Choice="+choice);
		String targetClassName = data.classAttribute().value(choice-1);
		//System.out.println("Target Class = "+ targetClassName);
		return targetClassName;
	}

	//Separate target and outlier data based on user input
	private void generateTargetData(Instances data){
		setTargetData();
		setOutlierData();

		for(int i=0;i<data.numInstances();i++) {
			if(data.instance(i).stringValue(data.classIndex()).equals(targetClassName))
				target.add(data.instance(i));
			else outlier.add(data.instance(i));
		}

		//System.out.println(target);
		//System.out.println(outlier);
	}

	//Generate Outlier data from the target training data, that will be treated as members of
	//negative class

	//Remove outliers from the training set based on IQR and generate new training set
	private Instances generateNewTarget(Instances targetTrainSet) throws Exception{
		//Calculate mean of the target data
		double [] center = new double [targetTrainSet.numAttributes()-1];
		for (int i=0;i<targetTrainSet.numAttributes()-1;i++){
			for(int j=0;j<targetTrainSet.numInstances();j++){
				center[i]+=targetTrainSet.instance(j).value(i)/targetTrainSet.numInstances();
			}
		}

		//find the distance of every instance from the center
		EuclideanDistance ed = new EuclideanDistance();
		double [] dist = new double [targetTrainSet.numInstances()];
		for (int i=0;i<targetTrainSet.numInstances();i++){
			double [] a = new double [targetTrainSet.numAttributes()-1];
			for (int j=0;j<targetTrainSet.numAttributes()-1;j++){
				a[j]= targetTrainSet.instance(i).value(j);
				//dist[i]+=Math.pow(targetTrainSet.instance(i).value(j)-center[j], 2);
			}
			dist[i]=ed.compute(a, center);
			//dist[i]=Math.sqrt(dist[i]);
			//System.out.print(dist[i]+",");
		}
		//System.out.println();
		//Remove the proxy outliers from the target data
		//Get the index of proxyoutliers
		proxyOutliersTrainSet=new Instances (targetTrainSet, targetTrainSet.numInstances());
		ArrayList<Integer>indexProxyOutliers=removeProxyOutliersFromTarget(dist);
		System.out.println("num Outliers from target="+indexProxyOutliers.size());
		if (indexProxyOutliers.size()<getCVfoldsInner())
			throw new Exception("Number of outliers are less than Number of Inner CV folds. "
					+ "Reduce the value in cv.setRejectionRate()\n");

		for (int i=0;i<indexProxyOutliers.size();i++){
			//System.out.print(indexProxyOutliers.get(i)+",");
			proxyOutliersTrainSet.add(i, targetTrainSet.instance(indexProxyOutliers.get(i)));
		}
		//System.out.println();
		//Match Index of outliers with target and find index of inliers
		ArrayList<Integer> inliers = new ArrayList<Integer>();
		for (int i=0;i<targetTrainSet.size();i++){
			int flag=0;			
			for (int j=0;j<indexProxyOutliers.size();j++){
				if (i != indexProxyOutliers.get(j)){
					flag=flag+0;
				}
				else if (i == indexProxyOutliers.get(j)){
					flag=flag+1;
					break;
				}
			}
			//If not match then add as inlier
			if(flag==0)
				inliers.add(i);
		}
		//System.out.println("size Inliers="+inliers.size());		
		/*for (int i=0;i<inliers.size();i++){
			System.out.print(inliers.get(i)+",");
		}
		System.out.println();
		 */
		//Create new target/positive class with only inliers
		Instances newTargetTrainSet = new Instances(targetTrainSet, targetTrainSet.size());
		for (int i=0;i<inliers.size();i++){
			newTargetTrainSet.add(i,targetTrainSet.instance(inliers.get(i)));
		}
		return newTargetTrainSet;
	}

	//This function removes proxy-outliers from the data using IQR technique
	private ArrayList<Integer> removeProxyOutliersFromTarget(double dist []){
		ArrayList<Integer> indexProxyOutliers = new ArrayList<Integer>();

		DescriptiveStatistics da = new DescriptiveStatistics(dist);
		double w=getRejectionRate();
		double q3 = da.getPercentile(75);
		double q1 = da.getPercentile(25);
		double iqr = q3-q1;
		int k=0;
		double upperLimit=q3+w*iqr;
		double lowerLimit=q1-w*iqr;
		for (int i=0;i<dist.length;i++){
			if(dist[i]>upperLimit || dist[i]<lowerLimit){
				indexProxyOutliers.add(k, i);
				k++;
			}
		}
		return indexProxyOutliers;		
	}


	//OCC-JKNN with best parameters
	private double  OCCJKNN(Instances trainSet, Instances testSet,Instances outlierSet, 
			int JNN, int KNN, double theta, int fold, boolean jk) throws Exception{

		targetLabels = new ArrayList<String>();
		outlierLabels = new ArrayList<String>();

		JKNN jknn = new JKNN();

		int TP = 0;
		int FP = 0;
		int TN = 0;
		int FN = 0;
		//Remove Last Attribute
		int [] rfeat = new int [trainSet.numAttributes()-1];
		for (int i=0;i<trainSet.numAttributes()-1;i++) 
			rfeat[i]=i;
		trainSet=createNewDataset(trainSet, rfeat);
		rfeat = new int [testSet.numAttributes()-1];
		for (int i=0;i<testSet.numAttributes()-1;i++) 
			rfeat[i]=i;
		testSet=createNewDataset(testSet, rfeat);
		for(int t=0;t<testSet.numInstances();t++){ //For every testset
			jknn.setaccept(0);
			jknn.setreject(0);
			jknn.computeKNNmetric(trainSet, testSet.instance(t), JNN, KNN, theta);
			TP+=jknn.getaccept();
			FN+=jknn.getreject();
			targetLabels.add(jknn.getLabel());
		}
		
		//Remove Last Attribute
		rfeat = new int [outlierSet.numAttributes()-1];
		for (int i=0;i<outlierSet.numAttributes()-1;i++) 
			rfeat[i]=i;
		outlierSet=createNewDataset(outlierSet, rfeat);
		for(int t=0; t<outlierSet.numInstances();t++) { //For every outlierset
			//System.out.println("outlier");
			jknn.setaccept(0);
			jknn.setreject(0);
			//System.out.println("\n*"+outlierSet.instance(t)+" JNN="+JNN+" KNN="+KNN);
			jknn.computeKNNmetric(trainSet, outlierSet.instance(t), JNN, KNN, theta);
			FP+=jknn.getaccept();
			TN+=jknn.getreject();
			outlierLabels.add(jknn.getLabel());
		}

		double tpr = (double) TP/(TP+FN);
		double tnr = (double) TN/(TN+FP);
		double precision = (double) TP/(TP+FP);
		double mcc = (double) (TP*TN-FP*FN)/Math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
		double gmean = Math.sqrt(tpr*tnr);
		//System.out.println("Eucgmean="+gmean+" tpr="+tpr+" tnr="+tnr+ " precision="+precision+ " mcc="+mcc);

		//		if (ensemble==false){
		if (getEnsemble()==false) {
			if (jk==true) {
				tprEuc[fold]=tpr;
				tnrEuc[fold]=tnr;
				precisionEuc[fold]=precision;
				mccEuc[fold]=mcc;
			}
			else if (jk==false && theta==1){
				tprEuc11[fold]=tpr;
				tnrEuc11[fold]=tnr;
				precisionEuc11[fold]=precision;
				mccEuc11[fold]=mcc;
			}
			else if (jk==false && theta!=1){
				tprEuc11Theta[fold]=tpr;
				tnrEuc11Theta[fold]=tnr;
				precisionEuc11Theta[fold]=precision;
				mccEuc11Theta[fold]=mcc;
			}
		}
		//System.out.println("TP="+TP+" TN="+TN);
		//double accuracy = (double) (TP+TN)/(TP+FN+FP+TN);
		return gmean;
	}

	//Finds the best parameters J and K
	private void findBestJK(double [][] gmean) {
		double max=0;
		for (int i=0;i<gmean.length;i++){
			for(int j=0;j<gmean.length;j++){
				//System.out.print(gmean[i][j]+"\t");
				if (gmean[i][j]>max){
					max=gmean[i][j];
				}
			}
			//System.out.println();
		}
		//System.out.println("Max Value="+max);
		//Find the index of max value in the gmean array
		for (int i=0;i<gmean.length;i++) {
			for (int j=0;j<gmean.length;j++) {
				if(max==gmean[i][j]){
					setBestJEuc(i+1);
					setBestKEuc(j+1);
					break;
				}
			}
		}
	}

	//Get max of each attribute of the data
	private double[] getMaxAttribute(Instances mydata) {
		double [] max = new double [mydata.numAttributes()-1];
		for (int i=0;i<mydata.numAttributes()-1;i++) {
			Stats at = mydata.attributeStats(i).numericStats;
			max[i]=at.max;
		}
		return max;
	}

	//Get min of each attribute of the data
	private double[] getMinAttribute(Instances mydata) {
		double [] min = new double [mydata.numAttributes()-1];
		for (int i=0;i<mydata.numAttributes()-1;i++) {
			Stats at = mydata.attributeStats(i).numericStats;
			min[i]=at.min;
		}
		return min;
	}

	//Normalize data in [0, 1]
	private Instances normalizeData(Instances mydata, double [] max, double [] min) throws Exception {
		//Instances newDataset = new Instances(data,data.numAttributes());
		for (int i=0;i<mydata.numInstances();i++) {
			for(int j=0;j<mydata.numAttributes()-1;j++) {
				double val = (mydata.instance(i).value(j)-min[j])/(max[j]-min[j]);
				mydata.instance(i).setValue(j, val);
			}
		}
		return mydata;
	}

	private Instances createRPDataset(Instances mydata, RealMatrix R) {
		//Perform Random Projection
		Instances rpdata = new Instances(mydata, mydata.numInstances());

		//convert Instances data to double array and take its transpose
		double [][] dataArray = new double [mydata.numInstances()][mydata.numAttributes()];
		for (int i=0;i<mydata.numInstances();i++) 
			dataArray[i] = mydata.instance(i).toDoubleArray(); 
		RealMatrix D1 = new Array2DRowRealMatrix(dataArray);
		RealMatrix D = D1.transpose();

		RealMatrix RkN = R.multiply(D);
		RealMatrix TRkN = RkN.transpose();
		double [][] rpMatrix = new double [mydata.numInstances()][mydata.numAttributes()];
		for (int i=0;i<mydata.numInstances();i++) {
			rpMatrix[i]=TRkN.getRow(i);
			Instance a = new DenseInstance(mydata.numAttributes());
			for (int j=0;j<mydata.numAttributes()-1;j++) {
				a.setValue(mydata.attribute(j), rpMatrix[i][j]);
			}
			//Add class label
			a.setValue(mydata.classAttribute(),mydata.instance(i).classValue());
			rpdata.add(i, a);
		}

		return rpdata;
	}

	//Get Random Projection Matrix
	public RealMatrix getRPMatrix() {
		//Generate random projections
		double [][] RP = new double [data.numAttributes()][data.numAttributes()];
		for (int i=0;i<data.numAttributes();i++) {
			for (int j=0;j<data.numAttributes();j++) {
				Random r = new Random();
				double x = r.nextDouble();
				if (x<=(double)1/6) 
					RP[i][j]=Math.sqrt(3);
				else if (x>=(double)5/6)
					RP[i][j]=-Math.sqrt(3);
				else 
					RP[i][j]=0;
			}
		}
		RealMatrix R = new Array2DRowRealMatrix(RP);

		return R;
	}

	//This function does the training of the classifiers along with Cross Validation and 
	//Parameter Optimization
	void training() throws Exception{

		//Output files
		BufferedWriter out = new BufferedWriter(new FileWriter(getFileName()+"-output.csv"));
		//BufferedWriter doutJK = new BufferedWriter(new FileWriter(getFileName()+"-distanceJK.csv"));
		//BufferedWriter dout11 = new BufferedWriter(new FileWriter(getFileName()+"-distance11.csv"));
		//Read original data file and extract the positive class
		out.write("Rejection Rate="+getRejectionRate()+", maxNN="+getMaxNN()+"\n");
		readData(getFileName());
		String tcn = getUserInput(); //To know which class is Target
		setTargetClassName(tcn);

		//Generate random features index for each subspace
		int [][][] randomSubSpaceIndex = generateRsubIndex();

		//Classification on original data
		System.out.println("\n^^^^^Original Data^^^^^\n");
		out.write("***Original Data\n");
		performCrossValidation(getFullData(),randomSubSpaceIndex,out);
		//doutJK.close();
		//dout11.close();
		out.close();
	}

	//Main Cross Validation method
	private void performCrossValidation(Instances mydata, int [][][] randomSubSpaceIndex, BufferedWriter out) throws Exception {
		int targetTestIndex []; //holds the indexes of target test samples used in a fold
		int targetTrainIndex []; //holds the indexes of target train samples used in a fold
		int outlierTestIndex []; //holds the indexes of outlier test samples used in a fold
		// Note: indexes of outlier trainset are not required because OCC doesnt use them.

		setTprEuc(getCVfoldsOuter());
		setTprEuc11(getCVfoldsOuter());
		setTprEuc11Theta(getCVfoldsOuter());

		setTnrEuc(getCVfoldsOuter());
		setTnrEuc11(getCVfoldsOuter());
		setTnrEuc11Theta(getCVfoldsOuter());

		setPrecisionEuc(getCVfoldsOuter());
		setPrecisionEuc11(getCVfoldsOuter());
		setPrecisionEuc11Theta(getCVfoldsOuter());

		setMCCEuc(getCVfoldsOuter());
		setMCCEuc11(getCVfoldsOuter());
		setMCCEuc11Theta(getCVfoldsOuter());

		generateTargetData(mydata);

		System.out.println("Total Number of data="+data.numInstances());
		System.out.println("Positive Data="+target.numInstances());
		System.out.println("Negative Data="+outlier.numInstances());
		System.out.println("Number of attributes="+(data.numAttributes()-1));
		if (outlier.numInstances()<getCVfoldsOuter())
			throw new Exception("Number of data in negative class is less than outer number of folds. "
					+ "Reduce the number of folds in cv.setCVFoldsOuter() or increase negative data"); 

		rsubGmeanEuc = new double [getCVfoldsOuter()][getRandomsubspace().length];
		rsubGmeanEuc11 = new double [getCVfoldsOuter()][getRandomsubspace().length];
		rsubGmeanEuc11Theta = new double [getCVfoldsOuter()][getRandomsubspace().length];
		rsubTprEuc  = new double [getCVfoldsOuter()][getRandomsubspace().length];
		rsubTprEuc11  = new double [getCVfoldsOuter()][getRandomsubspace().length];
		rsubTprEuc11Theta  = new double [getCVfoldsOuter()][getRandomsubspace().length];
		rsubTnrEuc  = new double [getCVfoldsOuter()][getRandomsubspace().length];
		rsubTnrEuc11  = new double [getCVfoldsOuter()][getRandomsubspace().length];
		rsubTnrEuc11Theta  = new double [getCVfoldsOuter()][getRandomsubspace().length];
		rsubPrecisionEuc = new double [getCVfoldsOuter()][getRandomsubspace().length];
		rsubPrecisionEuc11 = new double [getCVfoldsOuter()][getRandomsubspace().length];
		rsubPrecisionEuc11Theta = new double [getCVfoldsOuter()][getRandomsubspace().length];
		rsubMCCEuc = new double [getCVfoldsOuter()][getRandomsubspace().length];
		rsubMCCEuc11 = new double [getCVfoldsOuter()][getRandomsubspace().length];
		rsubMCCEuc11Theta = new double [getCVfoldsOuter()][getRandomsubspace().length];

		rprojGmeanEuc = new double [getCVfoldsOuter()];
		rprojGmeanEuc11 = new double [getCVfoldsOuter()];
		rprojGmeanEuc11Theta = new double [getCVfoldsOuter()];
		rprojTprEuc  = new double [getCVfoldsOuter()];
		rprojTprEuc11  = new double [getCVfoldsOuter()];
		rprojTprEuc11Theta  = new double [getCVfoldsOuter()];
		rprojTnrEuc  = new double [getCVfoldsOuter()];
		rprojTnrEuc11  = new double [getCVfoldsOuter()];
		rprojTnrEuc11Theta  = new double [getCVfoldsOuter()];
		rprojPrecisionEuc = new double [getCVfoldsOuter()];
		rprojPrecisionEuc11 = new double [getCVfoldsOuter()];
		rprojPrecisionEuc11Theta = new double [getCVfoldsOuter()];
		rprojMCCEuc = new double [getCVfoldsOuter()];
		rprojMCCEuc11 = new double [getCVfoldsOuter()];
		rprojMCCEuc11Theta = new double [getCVfoldsOuter()];

		int [] targetDataPerFold = setTestDataPerFold(getNumTargetData(),getCVfoldsOuter());
		int [] outlierDataPerFold = setTestDataPerFold(getNumOutlierData(),getCVfoldsOuter());
		int counterT=0;
		int counterO=0;
		double gmeanEuc [] = new double[getCVfoldsOuter()];
		double gmeanEuc11 [] = new double[getCVfoldsOuter()];
		double gmeanEuc11Theta [] = new double[getCVfoldsOuter()];

		//if (ensemble==false) {
		out.write("Fold,Euclidean-J,Euclidean-K,Theta");
		out.newLine();
		//}

		//For every outer cross-validation loop
		for(int i=0;i<getCVfoldsOuter();i++){ //For every Outer fold
			System.out.println("\n<<<<<Outer Fold="+(i+1)+">>>>>");

			//Create empty instances for train, test and outliers
			Instances targetTrainSet = new Instances (getTargetData(),getTargetData().numInstances());
			Instances targetTestSet = new Instances (getTargetData(),getTargetData().numInstances());
			Instances outlierTestSet = new Instances (getOutlierData(),getOutlierData().numInstances());

			//Generate target train and test set indexes
			generateCVFolds(counterT, getNumTargetData(), targetDataPerFold[i]); //For Target
			targetTestIndex  = getTestSetIndex();
			targetTrainIndex  = getTrainSetIndex();
			counterT+=targetDataPerFold[i];

			//Generate outlier test set indexes
			generateCVFolds(counterO, getNumOutlierData(), outlierDataPerFold[i]); //For Outlier
			outlierTestIndex = getTestSetIndex();
			counterO+=outlierDataPerFold[i];

			//Generate Target train set
			for(int j=0;j < targetTrainIndex.length; j++) {
				targetTrainSet.add(getTargetData().instance(targetTrainIndex[j]));
			}
			//System.out.println(targetTrainSet);
			double [] maxAttribute = getMaxAttribute(targetTrainSet);
			double [] minAttribute = getMinAttribute(targetTrainSet);
			targetTrainSet = normalizeData(targetTrainSet, maxAttribute, minAttribute);
			//System.out.println(targetTrainSet);
			//Generate Target test set
			for(int j=0;j < targetTestIndex.length;j++) {
				targetTestSet.add(getTargetData().instance(targetTestIndex[j]));
			}
			targetTestSet=normalizeData(targetTestSet, maxAttribute, minAttribute);
			//System.out.println(targetTestSet);

			//Generate Outlier test set
			for(int j=0;j < outlierTestIndex.length; j++) {
				outlierTestSet.add(getOutlierData().instance(outlierTestIndex[j]));
			}
			//System.out.println(outlierTestSet);
			outlierTestSet=normalizeData(outlierTestSet, maxAttribute, minAttribute);

			//Inner Cross Validation for parameter Optimization
			innerCrossValidation(targetTrainSet);

			//if (ensemble==false) {
			out.write((i+1)+","+getBestJEuc()+","+getBestKEuc()+","+getEmpiricalTheta());
			out.newLine();
			//}

			setEnsemble(false); //No ensemble for following methods
			//Perform OCC with best J,K 		
			gmeanEuc[i]=OCCJKNN(targetTrainSet, targetTestSet, outlierTestSet,getBestJEuc(),getBestKEuc(),1, i, true);			
			//Perform OCC with J=K=Theta=1 			
			gmeanEuc11[i]=OCCJKNN(targetTrainSet, targetTestSet, outlierTestSet,1,1,1, i, false);
			//Perform OCC with J=K=1 and Optimized Theta			
			gmeanEuc11Theta[i]=OCCJKNN(targetTrainSet, targetTestSet, outlierTestSet,1,1,getEmpiricalTheta(), i, false);
			System.out.println("Euclidean gmean["+i+"]="+gmeanEuc[i]);
			System.out.println("Euclidean11 gmean["+i+"]="+gmeanEuc11[i]);
			System.out.println("Euclidean11Theta gmean["+i+"]="+gmeanEuc11Theta[i]);
			//System.out.println();

			setEnsemble(true); //Start of Ensemble methods
			//Start Random Subspace Ensemble
			rsubTraining(targetTrainSet,targetTestSet,outlierTestSet,i,randomSubSpaceIndex);

			//Start Random Projection Ensemble
			rprojTraining(targetTrainSet,targetTestSet,outlierTestSet,i);

		} //end for Outer fold

		//Average gmean and Std. Dev
		double avggmeanEuc = computeMean(gmeanEuc);
		double sdevEuc = computeStdDeviation(gmeanEuc,avggmeanEuc);
		double avgTprEuc = computeMean(getTprEuc());
		double sdevTprEuc = computeStdDeviation(getTprEuc(),avgTprEuc);
		double avgTnrEuc = computeMean(getTnrEuc());
		double sdevTnrEuc = computeStdDeviation(getTnrEuc(),avgTnrEuc);
		double avgPrecisionEuc = computeMean(getPrecisionEuc());
		double sdevPrecisionEuc = computeStdDeviation(getPrecisionEuc(),avgPrecisionEuc);
		double avgMCCEuc = computeMean(getMCCEuc());
		double sdevMCCEuc = computeStdDeviation(getMCCEuc(),avgMCCEuc);
		double avggmeanEuc11 = computeMean(gmeanEuc11);
		double sdevEuc11 = computeStdDeviation(gmeanEuc11,avggmeanEuc11);
		double avgTprEuc11 = computeMean(getTprEuc11());
		double sdevTprEuc11 = computeStdDeviation(getTprEuc11(),avgTprEuc11);
		double avgTnrEuc11 = computeMean(getTnrEuc11());		
		double sdevTnrEuc11 = computeStdDeviation(getTnrEuc11(),avgTnrEuc11);
		double avgPrecisionEuc11 = computeMean(getPrecisionEuc11());
		double sdevPrecisionEuc11 = computeStdDeviation(getPrecisionEuc11(),avgPrecisionEuc11);
		double avgMCCEuc11 = computeMean(getMCCEuc11());
		double sdevMCCEuc11 = computeStdDeviation(getMCCEuc11(),avgMCCEuc11);
		double avggmeanEuc11Theta = computeMean(gmeanEuc11Theta);
		double sdevEuc11Theta = computeStdDeviation(gmeanEuc11Theta,avggmeanEuc11Theta);
		double avgTprEuc11Theta = computeMean(getTprEuc11Theta());
		double sdevTprEuc11Theta = computeStdDeviation(getTprEuc11Theta(),avgTprEuc11Theta);
		double avgTnrEuc11Theta = computeMean(getTnrEuc11Theta());
		double sdevTnrEuc11Theta = computeStdDeviation(getTnrEuc11Theta(),avgTnrEuc11Theta);
		double avgPrecisionEuc11Theta = computeMean(getPrecisionEuc11Theta());
		double sdevPrecisionEuc11Theta = computeStdDeviation(getPrecisionEuc11Theta(),avgPrecisionEuc11Theta);
		double avgMCCEuc11Theta = computeMean(getMCCEuc11Theta());
		double sdevMCCEuc11Theta = computeStdDeviation(getMCCEuc11Theta(),avgMCCEuc11Theta);

		//Random Subspace Metrics
		double [][] temp = rsubGmeanEuc;
		double [] avggmeanRsubEuc = computeMean(temp);
		double [] sdevRsubEuc = computeStdDeviation(temp);
		double [] avgRsubTprEuc = computeMean(getRsubTprEuc());
		double [] sdevRsubTprEuc = computeStdDeviation(getRsubTprEuc());
		double [] avgRsubTnrEuc = computeMean(getRsubTnrEuc());
		double [] sdevRsubTnrEuc = computeStdDeviation(getRsubTnrEuc());
		double [] avgRsubPrecisionEuc = computeMean(getRsubPrecisionEuc());
		double [] sdevRsubPrecisionEuc = computeStdDeviation(getRsubPrecisionEuc());
		double [] avgRsubMCCEuc = computeMean(getRsubMCCEuc());
		double [] sdevRsubMCCEuc = computeStdDeviation(getRsubMCCEuc());

		temp = rsubGmeanEuc11;
		double [] avggmeanRsubEuc11 = computeMean(temp);
		double [] sdevRsubEuc11 = computeStdDeviation(temp);
		double [] avgRsubTprEuc11 = computeMean(getRsubTprEuc11());
		double [] sdevRsubTprEuc11 = computeStdDeviation(getRsubTprEuc11());
		double [] avgRsubTnrEuc11 = computeMean(getRsubTnrEuc11());
		double [] sdevRsubTnrEuc11 = computeStdDeviation(getRsubTnrEuc11());
		double [] avgRsubPrecisionEuc11 = computeMean(getRsubPrecisionEuc11());
		double [] sdevRsubPrecisionEuc11 = computeStdDeviation(getRsubPrecisionEuc11());
		double [] avgRsubMCCEuc11 = computeMean(getRsubMCCEuc11());
		double [] sdevRsubMCCEuc11 = computeStdDeviation(getRsubMCCEuc11());

		temp = rsubGmeanEuc11Theta;
		double [] avggmeanRsubEuc11Theta = computeMean(temp);
		double [] sdevRsubEuc11Theta = computeStdDeviation(temp);
		double [] avgRsubTprEuc11Theta = computeMean(getRsubTprEuc11Theta());
		double [] sdevRsubTprEuc11Theta = computeStdDeviation(getRsubTprEuc11Theta());
		double [] avgRsubTnrEuc11Theta = computeMean(getRsubTnrEuc11Theta());
		double [] sdevRsubTnrEuc11Theta = computeStdDeviation(getRsubTnrEuc11Theta());
		double [] avgRsubPrecisionEuc11Theta = computeMean(getRsubPrecisionEuc11Theta());
		double [] sdevRsubPrecisionEuc11Theta = computeStdDeviation(getRsubPrecisionEuc11Theta());
		double [] avgRsubMCCEuc11Theta = computeMean(getRsubMCCEuc11Theta());
		double [] sdevRsubMCCEuc11Theta = computeStdDeviation(getRsubMCCEuc11Theta());

		//Random Projection Metrics
		double [] tempp = rprojGmeanEuc;
		double avggmeanRprojEuc = computeMean(tempp);
		double sdevRprojEuc = computeStdDeviation(tempp,avggmeanRprojEuc);
		double avgRprojTprEuc = computeMean(getRprojTprEuc());
		double sdevRprojTprEuc = computeStdDeviation(getRprojTprEuc(),avgRprojTprEuc);
		double avgRprojTnrEuc = computeMean(getRprojTnrEuc());
		double sdevRprojTnrEuc = computeStdDeviation(getRprojTnrEuc(),avgRprojTnrEuc);
		double avgRprojPrecisionEuc = computeMean(getRprojPrecisionEuc());
		double sdevRprojPrecisionEuc = computeStdDeviation(getRprojPrecisionEuc(),avgRprojPrecisionEuc);
		double avgRprojMCCEuc = computeMean(getRprojMCCEuc());
		double sdevRprojMCCEuc = computeStdDeviation(getRprojMCCEuc(),avgRprojMCCEuc);

		tempp = rprojGmeanEuc11;
		double avggmeanRprojEuc11 = computeMean(tempp);
		double sdevRprojEuc11 = computeStdDeviation(tempp,avggmeanRprojEuc11);
		double avgRprojTprEuc11 = computeMean(getRprojTprEuc11());
		double sdevRprojTprEuc11 = computeStdDeviation(getRprojTprEuc11(),avgRprojTprEuc11);
		double avgRprojTnrEuc11 = computeMean(getRprojTnrEuc11());
		double sdevRprojTnrEuc11 = computeStdDeviation(getRprojTnrEuc11(),avgRprojTnrEuc11);
		double avgRprojPrecisionEuc11 = computeMean(getRprojPrecisionEuc11());
		double sdevRprojPrecisionEuc11 = computeStdDeviation(getRprojPrecisionEuc11(),avgRprojPrecisionEuc11);
		double avgRprojMCCEuc11 = computeMean(getRprojMCCEuc11());
		double sdevRprojMCCEuc11 = computeStdDeviation(getRprojMCCEuc11(),avgRprojMCCEuc11);

		tempp = rprojGmeanEuc11Theta;
		double avggmeanRprojEuc11Theta = computeMean(tempp);
		double sdevRprojEuc11Theta = computeStdDeviation(tempp,avggmeanRprojEuc11Theta);
		double avgRprojTprEuc11Theta = computeMean(getRprojTprEuc11Theta());
		double sdevRprojTprEuc11Theta = computeStdDeviation(getRprojTprEuc11Theta(),avgRprojTprEuc11Theta);
		double avgRprojTnrEuc11Theta = computeMean(getRprojTnrEuc11Theta());
		double sdevRprojTnrEuc11Theta = computeStdDeviation(getRprojTnrEuc11Theta(),avgRprojTnrEuc11Theta);
		double avgRprojPrecisionEuc11Theta = computeMean(getRprojPrecisionEuc11Theta());
		double sdevRprojPrecisionEuc11Theta = computeStdDeviation(getRprojPrecisionEuc11Theta(),avgRprojPrecisionEuc11Theta);
		double avgRprojMCCEuc11Theta = computeMean(getRprojMCCEuc11Theta());
		double sdevRprojMCCEuc11Theta = computeStdDeviation(getRprojMCCEuc11Theta(),avgRprojMCCEuc11Theta);

		//Print Results
		System.out.println("\n#####Original Data#####");
		DecimalFormat df = new DecimalFormat("0.0000");
		System.out.println("\nFinal Results, gmean(sdev)");			
		System.out.print("EuclideanJK ==> ");
		printResults(avggmeanEuc, sdevEuc);
		System.out.print("Euclidean11 ==> ");
		printResults(avggmeanEuc11, sdevEuc11);
		System.out.print("Euclidean11Theta ==> ");
		printResults(avggmeanEuc11Theta, sdevEuc11Theta);
		
		//out.write("\nFinal Results, gmean(sdev)\n");
		out.write("Original, gmean, tpr, tnr, precision,mcc\n");
		out.write("EuclideanJK,"+df.format(avggmeanEuc)+"("+df.format(sdevEuc)+"),"+
				df.format(avgTprEuc)+"("+df.format(sdevTprEuc)+"),"+
				df.format(avgTnrEuc)+"("+df.format(sdevTnrEuc)+"),"+
				df.format(avgPrecisionEuc)+"("+df.format(sdevPrecisionEuc)+"),"+
				df.format(avgMCCEuc)+"("+df.format(sdevMCCEuc)+")\n");
		out.write("Euclidean11,"+df.format(avggmeanEuc11)+"("+df.format(sdevEuc11)+"),"+
				df.format(avgTprEuc11)+"("+df.format(sdevTprEuc11)+"),"+
				df.format(avgTnrEuc11)+"("+df.format(sdevTnrEuc11)+"),"+
				df.format(avgPrecisionEuc11)+"("+df.format(sdevPrecisionEuc11)+"),"+
				df.format(avgMCCEuc11)+"("+df.format(sdevMCCEuc11)+")\n");
		out.write("Euclidean11Theta,"+df.format(avggmeanEuc11Theta)+"("+df.format(sdevEuc11Theta)+"),"+
				df.format(avgTprEuc11Theta)+"("+df.format(sdevTprEuc11Theta)+"),"+
				df.format(avgTnrEuc11Theta)+"("+df.format(sdevTnrEuc11Theta)+"),"+
				df.format(avgPrecisionEuc11Theta)+"("+df.format(sdevPrecisionEuc11Theta)+"),"+
				df.format(avgMCCEuc11Theta)+"("+df.format(sdevMCCEuc11Theta)+")\n");

		System.out.println("\n#####Random Subspace#####");
		for (int i=0;i<getRandomsubspace().length;i++) {
			out.write("\nRSub-"+getRandomsubspace()[i]+", gmean, tpr, tnr,precision,mcc\n");
			System.out.println("\nFinal Results, gmean(sdev)");			
			System.out.println("Rsub-"+getRandomsubspace()[i]+"-EuclideanJK ==> "+df.format(avggmeanRsubEuc[i])+"("+df.format(sdevRsubEuc[i])+")");
			System.out.println("Rsub-"+getRandomsubspace()[i]+"-Euclidean11 ==> "+df.format(avggmeanRsubEuc11[i])+"("+df.format(sdevRsubEuc11[i])+")");
			System.out.println("Rsub-"+getRandomsubspace()[i]+"-Euclidean11Theta ==> "+df.format(avggmeanRsubEuc11Theta[i])+"("+df.format(sdevRsubEuc11Theta[i])+")");
			out.write("EuclideanJK,"+df.format(avggmeanRsubEuc[i])+"("+df.format(sdevRsubEuc[i])+"),"+
					df.format(avgRsubTprEuc[i])+"("+df.format(sdevRsubTprEuc[i])+"),"+
					df.format(avgRsubTnrEuc[i])+"("+df.format(sdevRsubTnrEuc[i])+"),"+
					df.format(avgRsubPrecisionEuc[i])+"("+df.format(sdevRsubPrecisionEuc[i])+"),"+
					df.format(avgRsubMCCEuc[i])+"("+df.format(sdevRsubMCCEuc[i])+")\n");
			out.write("Euclidean11,"+df.format(avggmeanRsubEuc11[i])+"("+df.format(sdevRsubEuc11[i])+"),"+
					df.format(avgRsubTprEuc11[i])+"("+df.format(sdevRsubTprEuc11[i])+"),"+
					df.format(avgRsubTnrEuc11[i])+"("+df.format(sdevRsubTnrEuc11[i])+"),"+
					df.format(avgRsubPrecisionEuc11[i])+"("+df.format(sdevRsubPrecisionEuc11[i])+"),"+
					df.format(avgRsubMCCEuc11[i])+"("+df.format(sdevRsubMCCEuc11[i])+")\n");
			out.write("Euclidean11Theta,"+df.format(avggmeanRsubEuc11Theta[i])+"("+df.format(sdevRsubEuc11Theta[i])+"),"+
					df.format(avgRsubTprEuc11Theta[i])+"("+df.format(sdevRsubTprEuc11Theta[i])+"),"+
					df.format(avgRsubTnrEuc11Theta[i])+"("+df.format(sdevRsubTnrEuc11Theta[i])+"),"+
					df.format(avgRsubPrecisionEuc11Theta[i])+"("+df.format(sdevRsubPrecisionEuc11Theta[i])+"),"+
					df.format(avgRsubMCCEuc11Theta[i])+"("+df.format(sdevRsubMCCEuc11Theta[i])+")\n");
			System.out.println();			
		}

		System.out.println("\n#####Random Projection#####");
		System.out.println("\nFinal Results, gmean(sdev)");			
		System.out.print("EuclideanJK ==> ");
		printResults(avggmeanRprojEuc, sdevRprojEuc);
		System.out.print("Euclidean11 ==> ");
		printResults(avggmeanRprojEuc11, sdevRprojEuc11);
		System.out.print("Euclidean11Theta ==> ");
		printResults(avggmeanRprojEuc11Theta, sdevRprojEuc11Theta);
		out.write("\nRandom Projection, gmean, tpr, tnr,precision,mcc\n");
		out.write("EuclideanJK,"+df.format(avggmeanRprojEuc)+"("+df.format(sdevRprojEuc)+"),"+
				df.format(avgRprojTprEuc)+"("+df.format(sdevRprojTprEuc)+"),"+
				df.format(avgRprojTnrEuc)+"("+df.format(sdevRprojTnrEuc)+"),"+
				df.format(avgRprojPrecisionEuc)+"("+df.format(sdevRprojPrecisionEuc)+"),"+
				df.format(avgRprojMCCEuc)+"("+df.format(sdevRprojMCCEuc)+")\n");
		out.write("Euclidean11,"+df.format(avggmeanRprojEuc11)+"("+df.format(sdevRprojEuc11)+"),"+
				df.format(avgRprojTprEuc11)+"("+df.format(sdevRprojTprEuc11)+"),"+
				df.format(avgRprojTnrEuc11)+"("+df.format(sdevRprojTnrEuc11)+"),"+
				df.format(avgRprojPrecisionEuc11)+"("+df.format(sdevRprojPrecisionEuc11)+"),"+
				df.format(avgRprojMCCEuc11)+"("+df.format(sdevRprojMCCEuc11)+")\n");
		out.write("Euclidean11Theta,"+df.format(avggmeanRprojEuc11Theta)+"("+df.format(sdevRprojEuc11Theta)+"),"+
				df.format(avgRprojTprEuc11Theta)+"("+df.format(sdevRprojTprEuc11Theta)+"),"+
				df.format(avgRprojTnrEuc11Theta)+"("+df.format(sdevRprojTnrEuc11Theta)+"),"+
				df.format(avgRprojPrecisionEuc11Theta)+"("+df.format(sdevRprojPrecisionEuc11Theta)+"),"+
				df.format(avgRprojMCCEuc11Theta)+"("+df.format(sdevRprojMCCEuc11Theta)+")\n");

	} //end for crossValidationOCC

	//Inner Cross Validation for parameter optimization
	private void innerCrossValidation(Instances targetTrainSet) throws Exception {
		//Center based NN method to remove outliers from training data to be used for 
		//validation and optimizing parameters
		Instances innerTarget=generateNewTarget(targetTrainSet);
		Instances innerOutlier=getProxyOutlierTrainSet();
		//System.out.println(innerOutlier);
		//double accEuc [][][] = new double[getCVfoldsInner()][getMaxNN()][getMaxNN()];

		int [] innerTargetDataPerFold = setTestDataPerFold(innerTarget.numInstances(),getCVfoldsInner());
		int [] innerOutlierDataPerFold = setTestDataPerFold(innerOutlier.numInstances(),getCVfoldsInner());
		int innerTargetTestIndex []; //holds the indexes of target test samples used in a fold
		int innerTargetTrainIndex []; //holds the indexes of target train samples used in a fold
		int innerOutlierTestIndex []; //holds the indexes of outlier test samples used in a fold
		// Note: indexes of outlier trainset are not required because OCC doesnt use them.
		int counterInnerT=0;
		int counterInnerO=0;
		//these two variable store D1,D2 and theta for inner target and outliers
		int countT=0;
		int countO=0; 
		int countTJK=0;
		int countOJK=0;

		double [][] thetaArrayT = new double[innerTarget.size()][3];
		double [][] thetaArrayO = new double[innerOutlier.size()][3];
		double [][][] JKArrayT = new double[getMaxNN()][getMaxNN()][innerTarget.size()];
		double [][][] JKArrayO = new double[getMaxNN()][getMaxNN()][innerOutlier.size()];

		//Inner CV for parameter optimization
		for (int j=0;j<getCVfoldsInner();j++){ //for every inner fold
			//System.out.println("<<<<<Inner Fold="+(j+1)+">>>>>");
			//Create empty instances for train, test and outliers
			Instances innerTargetTrainSet = new Instances (innerTarget,innerTarget.numInstances());
			Instances innerTargetTestSet = new Instances (innerTarget,innerTarget.numInstances());
			Instances innerOutlierTestSet = new Instances (innerOutlier,innerOutlier.numInstances());

			//Generate Inner target train and test set indexes
			generateCVFolds(counterInnerT, innerTarget.numInstances(), innerTargetDataPerFold[j]); //For Target
			innerTargetTestIndex  = getTestSetIndex();
			innerTargetTrainIndex  = getTrainSetIndex();
			counterInnerT+=innerTargetDataPerFold[j];

			//Generate Inner outlier test set indexes
			generateCVFolds(counterInnerO, innerOutlier.numInstances(), innerOutlierDataPerFold[j]); //For Outlier
			innerOutlierTestIndex = getTestSetIndex();
			counterInnerO+=innerOutlierDataPerFold[j];

			//Generate Inner Target test set
			for(int k=0;k < innerTargetTestIndex.length;k++) {
				innerTargetTestSet.add(innerTarget.instance(innerTargetTestIndex[k]));
			}
			//System.out.println(targetTestSet);

			//Generate Target train set
			for(int k=0;k < innerTargetTrainIndex.length; k++) {
				innerTargetTrainSet.add(innerTarget.instance(innerTargetTrainIndex[k]));
			}
			//System.out.println(targetTrainSet);

			//Generate Proxy Outlier test set
			for(int k=0;k < innerOutlierTestIndex.length; k++) {
				innerOutlierTestSet.add(innerOutlier.instance(innerOutlierTestIndex[k]));
			}
			//System.out.println(outlierTestSet);
			//Perform Inner OCC
			//System.out.println("Euclidean, ");
			//accEuc[j]= OCCJKNN(innerTargetTrainSet, innerTargetTestSet, innerOutlierTestSet,Evalue, doutJK);//JKNN

			//Remove Last Attribute
			int [] rfeat = new int [innerTargetTrainSet.numAttributes()-1];
			for (int i=0;i<innerTargetTrainSet.numAttributes()-1;i++) 
				rfeat[i]=i;
			innerTargetTrainSet=createNewDataset(innerTargetTrainSet, rfeat);
			innerTargetTestSet=createNewDataset(innerTargetTestSet, rfeat);
			innerOutlierTestSet=createNewDataset(innerOutlierTestSet, rfeat);
			//find D1,D2 for all inner testsets for finding best J and K
			JKNN jknn = new JKNN();
			for (int JNN=0;JNN<getMaxNN();JNN++) {
				for (int KNN=0;KNN<getMaxNN();KNN++) {		
					countTJK=0;
					for (int i=0;i<innerTargetTestSet.numInstances();i++) {
						jknn.computeKNNmetric(innerTargetTrainSet, innerTargetTestSet.instance(i), JNN+1, KNN+1, 1);
						JKArrayT[JNN][KNN][countTJK]=jknn.getAvgD1()/jknn.getAvgD2();
						countTJK++;
					}
				}
			}
			for (int JNN=0;JNN<getMaxNN();JNN++) {
				for (int KNN=0;KNN<getMaxNN();KNN++) {
					countOJK=0;
					for (int i=0;i<innerOutlierTestSet.numInstances();i++) {
						jknn.computeKNNmetric(innerTargetTrainSet, innerOutlierTestSet.instance(i), JNN+1, KNN+1, 1);
						JKArrayO[JNN][KNN][countOJK]=jknn.getAvgD1()/jknn.getAvgD2();
						countOJK++;
					}
				}
			}

			//Find D1, D2 for all inner testsets for finding empirical theta
			jknn = new JKNN();
			for (int i=0;i<innerTargetTestSet.numInstances();i++) {
				jknn.computeKNNmetric(innerTargetTrainSet, innerTargetTestSet.instance(i), 1, 1, 1);
				thetaArrayT[countT][0]=jknn.getAvgD1();
				thetaArrayT[countT][1]=jknn.getAvgD2();
				thetaArrayT[countT][2]=jknn.getAvgD1()/jknn.getAvgD2();
				countT++;
			}
			for (int i=0;i<innerOutlierTestSet.numInstances();i++) {
				jknn.computeKNNmetric(innerTargetTrainSet, innerOutlierTestSet.instance(i), 1, 1, 1);
				thetaArrayO[countO][0]=jknn.getAvgD1();
				thetaArrayO[countO][1]=jknn.getAvgD2();
				thetaArrayO[countO][2]=jknn.getAvgD1()/jknn.getAvgD2();
				countO++;
			}
		} //end inner CV fold

		//Find Best J and K
		//findParameterJK(JKArrayT,JKArrayO,innerTarget.size(),innerOutlier.size());
		//System.out.print("Euclidean ==> ");
		findParameterJK(JKArrayT,JKArrayO,countTJK,countOJK);
		//findBestJK(avgAccEuc);

		//Find Empirical theta
		setEmpiricalTheta(findEmpiricalTheta(thetaArrayO,thetaArrayT,countO, countT));

		System.out.println("bestJ="+getBestJEuc()+" bestK="+getBestKEuc()+" empTheta="+getEmpiricalTheta());
	}

	//find Empirical theta
	private double findEmpiricalTheta(double [][] thetaArrayO, double [][] thetaArrayT, int countO, int countT) {
		double [] gmeanT = new double [countT];
		//Check emp threshold among all positive samples
		for (int i=0;i<countT;i++) {
			int TP=0;
			int FN=0;
			int TN=0;
			int FP=0;
			double threshold = thetaArrayT[i][2];
			for (int j=0;j<countT;j++){
				if (thetaArrayT[j][2]/threshold <= 1.0)
					TP++;
				else
					FN++;
			}
			for (int j=0;j<countO;j++) {
				if (thetaArrayO[j][2]/threshold <= 1.0)
					FP++;
				else 
					TN++;
			}
			double tpr=(double)TP/(TP+FN);
			double tnr=(double)TN/(TN+FP);
			gmeanT[i]=Math.sqrt(tpr*tnr);
		}
		double maxT=0.0;
		int maxInd=0;
		for (int i=0;i<gmeanT.length;i++){
			if (gmeanT[i] > maxT){
				maxT=gmeanT[i];
				maxInd=i;
			}
		}
		double eThetaT=thetaArrayT[maxInd][2];
		//System.out.println(gmeanT[maxInd]+" ethetaT="+eThetaT+" maxind="+maxInd);

		//Check emp threshold among all negative samples		
		double [] gmeanO = new double [countO];
		for (int i=0;i<countO;i++) {
			int TP=0;
			int FN=0;
			int TN=0;
			int FP=0;
			double threshold = thetaArrayO[i][2];
			for (int j=0;j<countT;j++){
				if (thetaArrayT[j][2]/threshold <= 1.0)
					TP++;
				else
					FN++;
			}
			for (int j=0;j<countO;j++) {
				if (thetaArrayO[j][2]/threshold <= 1.0)
					FP++;
				else 
					TN++;
			}
			double tpr=(double)TP/(TP+FN);
			double tnr=(double)TN/(TN+FP);
			gmeanO[i]=Math.sqrt(tpr*tnr);
		}
		double maxO=0.0;
		maxInd=0;
		for (int i=0;i<gmeanO.length;i++){
			if (gmeanO[i] > maxO){
				maxO=gmeanO[i];
				maxInd=i;
			}
		}
		double eThetaO=thetaArrayO[maxInd][2];
		//System.out.println(gmeanO[maxInd]+" ethetaO="+eThetaO+" maxind="+maxInd);

		double thetaRet=0;
		if (maxT >= maxO) {
			thetaRet=eThetaT;
		}
		else if (maxT < maxO) {
			thetaRet=eThetaO;
		}
		return thetaRet;
	}

	//find best J and K
	private void findParameterJK(double [][][] JKArrayT, double [][][] JKArrayO, int countT, int countO) {
		int TP [][] = new int [getMaxNN()][getMaxNN()];
		int FN [][] = new int [getMaxNN()][getMaxNN()];
		int TN [][] = new int [getMaxNN()][getMaxNN()];
		int FP [][] = new int [getMaxNN()][getMaxNN()];
		double gmean [][] = new double[getMaxNN()][getMaxNN()];
		//double [] gmean = new double [countO];
		for (int JNN=0;JNN<getMaxNN();JNN++) {
			for (int KNN=0;KNN<getMaxNN();KNN++) {
				for (int i=0;i<countT;i++) {
					if(JKArrayT[JNN][KNN][i] <=1.0)
						TP[JNN][KNN]++;
					else
						FN[JNN][KNN]++;					
				}			
				for (int i=0;i<countO;i++) {
					if(JKArrayO[JNN][KNN][i] <=1.0)
						FP[JNN][KNN]++;
					else
						TN[JNN][KNN]++;					
				}
				double tpr=(double)TP[JNN][KNN]/(TP[JNN][KNN]+FN[JNN][KNN]);
				double tnr=(double)TN[JNN][KNN]/(TN[JNN][KNN]+FP[JNN][KNN]);
				gmean[JNN][KNN]=Math.sqrt(tpr*tnr);
			}//End of KNN
		}//End of JNN
		//Find best J and K from the gmean array
		findBestJK(gmean);
	}

	//Generate index for random subspace, to remain same across all outer folds
	private int [][][] generateRsubIndex() throws Exception {
		int [][][] randomFeaturesIndex = new int [getRandomsubspace().length][getNumEnsemble()][data.numAttributes()-1];
		setSubspaceSize();
		for (int i=0;i<getRandomsubspace().length;i++) {
			if( getRandomsubspace()[i] >= 100  || getRandomsubspace()[i] <=0)
				throw new Exception("Subspace Size is out of range. Keep it between 0 and 100.\n");
			getSubspaceSize()[i]=(int) ((double)getRandomsubspace()[i]/100*(data.numAttributes()-1));
			System.out.println("subSpace size="+getSubspaceSize()[i]);			
			if (getSubspaceSize()[i]==0)
				throw new Exception("Subspace size is out of range. Increase its size\n");
			for (int j=0;j<getNumEnsemble();j++){
				//generate random features
				randomFeaturesIndex[i][j] = randomFeaturesIndex(subspaceSize[i]);				
			}
		}		
		return randomFeaturesIndex;
	}

	//Training Random subspace
	private void rsubTraining(Instances targetTrainSet,Instances targetTestSet,Instances 
			outlierTestSet, int fold, int [][][] randomFeaturesIndex) throws Exception {
		//System.out.println("*****Random Subspace*****");
		int subspaceArraySize = getRandomsubspace().length;//Total Number of subspaces to test
		for (int i=0;i<subspaceArraySize;i++) { //for every subspace
			System.out.println("\n####Random Subspace="+(i+1));
			rsubTargetLabelsEuc = new ArrayList<ArrayList<String>>();
			rsubOutlierLabelsEuc = new ArrayList<ArrayList<String>>();
			rsubTargetLabelsEuc11 = new ArrayList<ArrayList<String>>();
			rsubOutlierLabelsEuc11 = new ArrayList<ArrayList<String>>();
			rsubTargetLabelsEuc11Theta = new ArrayList<ArrayList<String>>();
			rsubOutlierLabelsEuc11Theta = new ArrayList<ArrayList<String>>();

			for (int j=0;j<getNumEnsemble();j++) { // for each subspace ensemble
				System.out.println("\n***Ensemble:"+(j+1));
				int [] rfeat = new int [getSubspaceSize()[i]+1];
				for (int k=0;k<getSubspaceSize()[i]+1;k++){
					rfeat [k] = randomFeaturesIndex[i][j][k];
					//System.out.print("*"+randomFeaturesIndex[i][j][k]+" ");
				}

				//Remove the attributes from data and create a new copy
				Instances newTargetTrainSet=createNewDataset(targetTrainSet, rfeat);
				Instances newTargetTestSet=createNewDataset(targetTestSet, rfeat);
				Instances newOutlierTestSet=createNewDataset(outlierTestSet, rfeat);

				//Perform Inner CV on this subspace for parameter Optimization
				innerCrossValidation(newTargetTrainSet);

				//Perform OCC with best J,K Parameters
				OCCJKNN(newTargetTrainSet, newTargetTestSet, newOutlierTestSet,getBestJEuc(),getBestKEuc(),1,j,true);//JKNN
				rsubTargetLabelsEuc.add(getTargetLabels());
				rsubOutlierLabelsEuc.add(getOutlierLabels());
				//Perform OCC with J=K=Theta=1
				OCCJKNN(newTargetTrainSet, newTargetTestSet, newOutlierTestSet,1,1,1,j,false);//1NN
				rsubTargetLabelsEuc11.add(getTargetLabels());
				rsubOutlierLabelsEuc11.add(getOutlierLabels());
				//Perform OCC with J=K=1 and Optimized Theta
				OCCJKNN(newTargetTrainSet, newTargetTestSet, newOutlierTestSet,1,1,getEmpiricalTheta(),j,false);//1NN+empirical theta
				rsubTargetLabelsEuc11Theta.add(getTargetLabels());
				rsubOutlierLabelsEuc11Theta.add(getOutlierLabels());
			} //End of each subspace ensemble
			rsubGmeanEuc[fold][i]=computeGmean(targetTestSet,outlierTestSet,getRsubTargetLabelsEuc(),getRsubOutlierLabelsEuc());
			rsubTprEuc[fold][i]=getRsubTpr();
			rsubTnrEuc[fold][i]=getRsubTnr();
			rsubPrecisionEuc[fold][i]=getRsubPrecision();
			rsubMCCEuc[fold][i]=getRsubMCC();
			//System.out.println("Rsub_gmeanEuc["+fold+"]["+i+"]="+rsubGmeanEuc[fold][i]);

			rsubGmeanEuc11[fold][i]=computeGmean(targetTestSet,outlierTestSet,getRsubTargetLabelsEuc11(),getRsubOutlierLabelsEuc11());
			rsubTprEuc11[fold][i]=getRsubTpr();
			rsubTnrEuc11[fold][i]=getRsubTnr();
			rsubPrecisionEuc11[fold][i]=getRsubPrecision();
			rsubMCCEuc11[fold][i]=getRsubMCC();
			//System.out.println("Rsub_gmeanEuc11["+fold+"]["+i+"]="+rsubGmeanEuc11[fold][i]);

			rsubGmeanEuc11Theta[fold][i]=computeGmean(targetTestSet,outlierTestSet,getRsubTargetLabelsEuc11Theta(),getRsubOutlierLabelsEuc11Theta());
			rsubTprEuc11Theta[fold][i]=getRsubTpr();
			rsubTnrEuc11Theta[fold][i]=getRsubTnr();
			rsubPrecisionEuc11Theta[fold][i]=getRsubPrecision();
			rsubMCCEuc11Theta[fold][i]=getRsubMCC();
			//System.out.println("Rsub_gmeanEuc11Theta["+fold+"]["+i+"]="+rsubGmeanEuc11Theta[fold][i]);
		} //End of every subspace
	} //end for rsub

	//Training Random Projection
	private void rprojTraining(Instances targetTrainSet,Instances targetTestSet,Instances 
			outlierTestSet, int fold) throws Exception {
		System.out.println("\n####Random Projection");
		rprojTargetLabelsEuc = new ArrayList<ArrayList<String>>();
		rprojOutlierLabelsEuc = new ArrayList<ArrayList<String>>();
		rprojTargetLabelsEuc11 = new ArrayList<ArrayList<String>>();
		rprojOutlierLabelsEuc11 = new ArrayList<ArrayList<String>>();
		rprojTargetLabelsEuc11Theta = new ArrayList<ArrayList<String>>();
		rprojOutlierLabelsEuc11Theta = new ArrayList<ArrayList<String>>();

		for (int j=0;j<getNumEnsemble();j++) { // for each RP ensemble
			System.out.println("\n***Ensemble:"+(j+1));
			RealMatrix RPMatrix = getRPMatrix();
			//create a new copy with random projection
			Instances newTargetTrainSet=createRPDataset(targetTrainSet, RPMatrix);
			Instances newTargetTestSet=createRPDataset(targetTestSet, RPMatrix);
			Instances newOutlierTestSet=createRPDataset(outlierTestSet, RPMatrix);

			//Perform Inner CV on this subspace for parameter Optimization
			innerCrossValidation(newTargetTrainSet);

			//Perform OCC with best J,K Parameters
			OCCJKNN(newTargetTrainSet, newTargetTestSet, newOutlierTestSet,getBestJEuc(),getBestKEuc(),1,j,true);//JKNN
			rprojTargetLabelsEuc.add(getTargetLabels());
			rprojOutlierLabelsEuc.add(getOutlierLabels());
			//Perform OCC J=K=1
			OCCJKNN(newTargetTrainSet, newTargetTestSet, newOutlierTestSet,1,1,1,j,false);//1NN
			rprojTargetLabelsEuc11.add(getTargetLabels());
			rprojOutlierLabelsEuc11.add(getOutlierLabels());
			//Perform OCC with J=K=1 and Optimized Theta
			OCCJKNN(newTargetTrainSet, newTargetTestSet, newOutlierTestSet,1,1,getEmpiricalTheta(),j,false);//1NN+empirical theta
			rprojTargetLabelsEuc11Theta.add(getTargetLabels());
			rprojOutlierLabelsEuc11Theta.add(getOutlierLabels());
		} //End of each random projection ensemble

		rprojGmeanEuc[fold]=computeGmean(targetTestSet,outlierTestSet,getRprojTargetLabelsEuc(),getRprojOutlierLabelsEuc());
		rprojTprEuc[fold]=getRsubTpr();
		rprojTnrEuc[fold]=getRsubTnr();
		rprojPrecisionEuc[fold]=getRsubPrecision();
		rprojMCCEuc[fold]=getRsubMCC();
		//System.out.println("Rproj_gmeanEuc["+fold+"]="+rprojGmeanEuc[fold]+" "+rprojTprEuc[fold]+" "+rprojTnrEuc[fold]);

		rprojGmeanEuc11[fold]=computeGmean(targetTestSet,outlierTestSet,getRprojTargetLabelsEuc11(),getRprojOutlierLabelsEuc11());
		rprojTprEuc11[fold]=getRsubTpr();
		rprojTnrEuc11[fold]=getRsubTnr();
		rprojPrecisionEuc11[fold]=getRsubPrecision();
		rprojMCCEuc11[fold]=getRsubMCC();
		//System.out.println("Rsub_gmeanEuc11["+fold+"]["+i+"]="+rsubGmeanEuc11[fold][i]);

		rprojGmeanEuc11Theta[fold]=computeGmean(targetTestSet,outlierTestSet,getRprojTargetLabelsEuc11Theta(),getRprojOutlierLabelsEuc11Theta());
		rprojTprEuc11Theta[fold]=getRsubTpr();
		rprojTnrEuc11Theta[fold]=getRsubTnr();
		rprojPrecisionEuc11Theta[fold]=getRsubPrecision();
		rprojMCCEuc11Theta[fold]=getRsubMCC();
		//System.out.println("Rsub_gmeanEuc11Theta["+fold+"]["+i+"]="+rsubGmeanEuc11Theta[fold][i]);
	} //end for random projection

	//Compute gmean
	private double computeGmean (Instances targetTestSet, Instances outlierTestSet,
			ArrayList<ArrayList<String>> ensembleTargetLabels,
			ArrayList<ArrayList<String>> ensembleOutlierLabels) {
		int TP=0;
		int FN=0;
		int TN=0;
		int FP=0;
		for (int k=0;k<targetTestSet.size();k++) {
			ArrayList<String> predictedLabels = new ArrayList<String>();
			for (int j=0;j<getNumEnsemble();j++){
				predictedLabels.add(j, ensembleTargetLabels.get(j).get(k));
				//System.out.print(ensembleTargetLabels.get(j).get(k)+" ");
			}
			String plabel=majorityVoting(predictedLabels);
			//System.out.println(" -> "+plabel+" ");
			if (plabel.equalsIgnoreCase("p"))
				TP++;
			else if (plabel.equalsIgnoreCase("n"))
				FN++;
		}
		//System.out.println();
		for (int k=0;k<outlierTestSet.size();k++) {
			ArrayList<String> predictedLabels = new ArrayList<String>();
			for (int j=0;j<getNumEnsemble();j++){
				predictedLabels.add(j, ensembleOutlierLabels.get(j).get(k));
				//System.out.print(ensembleOutlierLabels.get(j).get(k)+" ");
			}
			String plabel=majorityVoting(predictedLabels);
			//System.out.println(" -> "+plabel+" ");
			if (plabel.equalsIgnoreCase("n"))
				TN++;
			else if (plabel.equalsIgnoreCase("p"))
				FP++;
		}
		double tpr=(double)TP/(TP+FN);
		double tnr=(double)TN/(TN+FP);
		double precision = (double) TP/(TP+FP); 
		double gmean=Math.sqrt(tpr*tnr);
		double mcc = (double) (TP*TN-FP*FN)/Math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
		//System.out.println("gmean="+gmean+" tpr="+tpr+" tnr="+tnr+ " precision="+precision+ " mcc="+mcc);
		setRsubTpr(tpr);
		setRsubTnr(tnr);
		setRsubPrecision(precision);
		setRsubMCC(mcc);
		return gmean;
	}

	//Performs Majority Voting over the ensembles
	private String majorityVoting(ArrayList<String> labels) {
		int counter=0;
		for (int i=0;i<labels.size();i++) {
			if (labels.get(i).toString().equalsIgnoreCase("p")) {
				counter++;
			}
		}
		if (counter >= Math.ceil((double)getNumEnsemble()/2))
			return "p";
		else 
			return "n";
	}

	//Create new dataset based on random feature index
	private Instances createNewDataset(Instances dataset, int [] rfeat) throws Exception {
		Instances newDataset = new Instances(data,data.numAttributes());
		Remove re = new Remove();
		re.setAttributeIndicesArray(rfeat);
		re.setInvertSelection(true);
		re.setInputFormat(dataset);
		newDataset = Filter.useFilter(dataset, re);

		return newDataset;
	}

	//Non-repeated random features as per the size of subspace
	private int[] randomFeaturesIndex(int subspaceSize) {
		List<Integer> FeaturesIndex = new ArrayList<Integer>();
		int [] randomFeaturesIndex = new int [subspaceSize+1];
		System.out.print("Random features index: ");
		int range=data.numAttributes()-1;
		for (int i=0;i<range;i++){
			FeaturesIndex.add(i, i);
		}
		Collections.shuffle(FeaturesIndex);

		for(int i=0;i<subspaceSize;i++){
			randomFeaturesIndex[i]=FeaturesIndex.get(i);
			System.out.print(randomFeaturesIndex[i]+" ");
		}
		randomFeaturesIndex[subspaceSize]=data.numAttributes()-1;//to hold the class label
		System.out.println();
		return randomFeaturesIndex;
	}

	//Compute Mean Accuracy across all outer folds for best parameters
	private double computeMean(double accuracy []) {
		double avgAccuracy = 0;
		for(int i=0;i<getCVfoldsOuter();i++){
			avgAccuracy+=accuracy[i]/getCVfoldsOuter();
		}

		return avgAccuracy;
	} //end for computeMean

	//Compute Mean Accuracy across all outer fold for each random subspace
	private double [] computeMean(double accuracy [][]) {
		double [] avgAcc = new double [getRandomsubspace().length];
		for (int j=0;j<getRandomsubspace().length;j++) {
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (int i=0;i<getCVfoldsOuter();i++) {
				ds.addValue(accuracy[i][j]);
			}
			avgAcc[j] = ds.getMean();
		}	
		return avgAcc;
	}

	//Compute Standard Deviation across all outer fold for each random subspace
	private double [] computeStdDeviation(double accuracy [][]) {
		double [] sdAcc = new double [getRandomsubspace().length];
		for (int j=0;j<getRandomsubspace().length;j++) {
			DescriptiveStatistics ds = new DescriptiveStatistics();
			for (int i=0;i<getCVfoldsOuter();i++) {
				ds.addValue(accuracy[i][j]);
			}
			sdAcc[j] = ds.getStandardDeviation();
		}	
		return sdAcc;
	}	

	//Compute Std Deviation across all folds for best parameters
	private double computeStdDeviation(double accuracy [], double avgAccuracy) {
		double sdev = 0;
		for(int i=0;i<getCVfoldsOuter();i++) {
			sdev+=Math.pow(avgAccuracy-accuracy[i],2)/(getCVfoldsOuter()-1);
		}
		return Math.sqrt(sdev);
	} //end for computeStdDeviation

	//Print Results for best parameters
	private void printResults(double avgAccuracy, double sdev){
		DecimalFormat df = new DecimalFormat("0.0000");
		System.out.println(df.format(avgAccuracy)+"("+df.format(sdev)+")\t");

	} //end for printResults

}//End of class