package vision.src;

import java.util.Arrays;

import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import weka.core.Instance;
import weka.core.Instances;

//Class to implement One Class j+kNN, fixed Threshold
class JKNN {

	private int accept;
	private int reject ;
	private int [] index;
	private double avgD1;
	private double avgD2;
	private String label;

	private void setIndex(int ind, int val) {
		index [ind] = val;
	}
	
	public int [] getIndex() {
		return index;
	}
	
	int getaccept() {
		return accept;
	}

	void setaccept(int s){
		accept = s;
	}

	int getreject(){
		return reject;
	}

	void setreject(int r){
		reject = r;
	}

	public double getAvgD1() {
		return avgD1;
	}
	
	public void setAvgD1(double d) {
		avgD1=d;
	}
	
	public double getAvgD2() {
		return avgD2;
	}
	
	public void setAvgD2(double d) {
		avgD2=d;
	}
	public String getLabel() {
		return label;
	}

	public void setLabel(String s) {
		label=s;
	}

	void computeKNNmetric (Instances training, Instance testing, int JNN, int KNN, double theta) throws Exception {

		index = new int [JNN];
		//System.out.println("*t="+training.numInstances()+" ts="+testing.numInstances());
		//for(int i=0;i<testing.numInstances();i++) {
		//int val = 0; 
		//System.out.print("external"+i+" = ");
		double [] D1 = computeDistance(testing,training, JNN,0); //compute external neighbor distance
		//Take average of JNN
		DescriptiveStatistics ds = new DescriptiveStatistics();
		for (int i=0;i<D1.length;i++) {
			ds.addValue(D1[i]);
		}
		setAvgD1(ds.getMean());

		ds.clear();
		//System.out.println("internal"+(i+1));
		for(int i=0;i<JNN;i++) {
			//System.out.print("D1["+j+"]="+D1[j]+" "+index[j]+" ");
			double[] D2 = computeDistance(training.instance(getIndex()[i]),training, KNN,1);//compute internal neighbor distance
			for (int j=0;j<D2.length;j++) {
				ds.addValue(D2[j]);
			}
		} //end for JNN
		//Take average of KNN
		setAvgD2(ds.getMean());
		//Find threshold
		double threshold = getAvgD1()/getAvgD2();
		//if (ensemble==false) 
		//	out.write(classlabel+","+avgD1+","+avgD2+","+threshold+"\n");
		if (threshold <= theta) {
			accept++;
			setLabel("p");
			//System.out.println(" Accept="+accept);
		}
		else {
			reject++;
			setLabel("n");
			//System.out.println(" Reject="+reject);
		}
		//System.out.println("accept="+accept+" reject="+reject);
		//System.out.println();
	} //end computeKNNMetric

	private double[] computeDistance(Instance testing, Instances training, int nn, int level) throws Exception{
		double [] score = new double [nn];
		double[] distance = new double [training.numInstances()];
		double [] temp = new double [training.numInstances()];
		EuclideanDistance ed = new EuclideanDistance();
		//System.out.println("t="+training.numInstances());
		if (training.numInstances()<=nn)
			throw new Exception("Number of training data is less than nearest neighbours. Reduce Number of NN with cv.setMaxNN()");
		for(int j=0;j<training.numInstances();j++){
			//Compute 1-nearest neighbor
			/*double [] A = new double [testing.numAttributes()-1];
			double [] B = new double [training.instance(j).numAttributes()-1];
			for (int x=0;x<testing.numAttributes()-1;x++)
				A[x]=testing.toDoubleArray()[x];
			
			for (int x=0;x<training.instance(j).numAttributes()-1;x++)
				B[x]=training.instance(j).toDoubleArray()[x];
			
			distance[j]=ed.compute(B, A);
			*/
			distance[j]=ed.compute(training.instance(j).toDoubleArray(), testing.toDoubleArray());
			temp[j]=distance[j];
			//System.out.println("j="+j+" d="+distance[j]+" ");
		}//end for j
		Arrays.sort(temp);		//System.out.println("t0="+temp[0]+" t1="+temp[training.numInstances()-1]);
		/*for (int i=0;i<temp.length;i++){
			System.out.println(temp[i]);
		}*/
		
		if(level==0) {
			for(int i=0; i< nn; i++) {
				for(int j=0;j<training.numInstances();j++) {
					if(temp[i]==distance[j]){
						score[i]=distance[j];
						setIndex(i,j);
						break;
					}//end of if
				} //end for j
			} //end for i
		} //end of if level

		if(level==1) {
			//count number of zero distances
			int count0=0;
			for (int i=0;i<temp.length;i++) {
				if (temp[i]==0)
					count0++;
			}
			for(int i=0; i < nn; i++){
				if ((i+count0) >= temp.length) {
					throw new Exception("Number of duplicates exceeds the distance array. Reduce NN");
				}
				//System.out.print("t="+temp[i+1+count0]+" ");
				score[i]+=temp[i+count0];//skip zero distances
				//System.out.println("score["+i+"]="+score[i]);
			}
		}
		return score;
	} //end of computeDistance
}//end of class jkLSKNN
