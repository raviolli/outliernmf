class TONMF(object):
	'''
	https://arxiv.org/pdf/1701.01325v1.pdf
	'''
	def __init__(self, specs):
		self.logger = logging.getLogger(__name__)

		# required_specs=['vector_colname', 'output_colname']
		# if not all([rs in specs.keys() for rs in required_specs]):
		# 	raise AttributeError("Required Specifications Keys not found in spes.\n{}".format(required_specs))

		# self.vector_colname = specs['vector_colname']
		# self.output_colname = specs['output_colname']

		self.rank = specs.get('rank',100)
		self.a_reg = specs.get('alpha', 0.01)
		self.b_reg = specs.get('beta', 0.01)
		self.WH_iter = specs.get('wh_iter',10)
		self.iter = specs.get('iter',10)
		

	def hals(self, A,Winit,Hinit,k,tolerance,numIterations,beta):
		# solves A=WH 
		# A = mxn matrix
		# W = mxk 
		# H = kxn
		# k = low rank
		# implementation of the algorithm2 from 
		# http://www.bsp.brain.riken.jp/publications/2009/Cichocki-Phan-IEICE_col.pdf
		W=Winit
		H=Hinit
		prevError=np.linalg.norm(A-np.matmul(W,H), 'fro')
		currError = prevError+1
		currentIteration=0
		errChange=np.zeros(shape=(numIterations,1))
		while ( (abs(currError-prevError)>tolerance) & (currentIteration<numIterations) ):
			# update W
			AHt=np.matmul(A,H.T)
			HHt=np.matmul(H,H.T)
			# to avoid divide by zero error.
			HHtDiag=np.array(np.diag(HHt))
			HHtDiag[HHtDiag==0]=EPS

			for x in range(k): 
				Wx = W[:,x] + (AHt[:,x]- np.matmul(W,HHt[:,x]))/HHtDiag[x]
				Wx[Wx<EPS]=EPS
				W[:,x]=Wx

			# update H
			WtA=np.matmul(W.T,A)
			WtW=np.matmul(W.T,W)
			# to avoid divide by zero error.
			WtWDiag=np.array(np.diag(WtW))
			WtWDiag[WtWDiag==0]=EPS
			for x in range(k):
				Hx = H[x,:]+(WtA[x,:]- np.matmul(WtW[x,:],H))/WtWDiag[x]
				Hx=Hx-(beta/WtWDiag[x])
				Hx[Hx<EPS]=EPS
				H[x,:]=Hx
			if (currentIteration>0):
				prevError=currError
		
			errChange[currentIteration]=prevError
			currError=np.linalg.norm(A- np.matmul(W,H), 'fro')
			currentIteration+=1

		return ( W,H,errChange )


	def textoutliers(self, A,k,alpha,beta ):
		# %this function solves the equation
		# % A = ||A-Z-WH||_F^2 + alpha ||Z||_2,1 + beta ||H||_1
		# % A is a sparse matrix of size mxn
		# % the Z matrix is the outlier matrix.
		m,n=A.shape
		# %numIterations is used for convergence of W,H
		numIterationsWH = self.WH_iter
		numIterations = self.iter
		currentIteration=0
		# %first fix W,H and solve Z.
		W = np.random.rand(m,k)
		H = np.random.rand(k,n)
		D = A- np.matmul(W,H)
		Z = np.zeros(shape=(m,n))
		# %prevErr =  norm(D-Z,'fro')+alpha*sum(sqrt(sum(Z.^2)))+beta * norm(H,1)
		prevErr = np.linalg.norm(D,'fro') + beta * np.linalg.norm(H,1)
		currentErr = prevErr - 1
		errChange = np.zeros(shape=(numIterations+1,1))
		# %convergence is when A \approx WH
		while(currentIteration < numIterations and (np.abs(prevErr-currentErr) > 1e-3)):
			# Calculate Outlier portion given W,H
			colnormdi=np.sqrt(np.sum( np.square(D),0 ))
			colnormdi_factor=colnormdi-alpha
			colnormdi_factor[colnormdi_factor<0]=0 #Non-Negative restriction
			Z = D / colnormdi
			Z = Z * colnormdi_factor
			D = A-Z
			# Optimize Matrix Factors (W,H)
			W,H,h_err=self.hals(A=D,Winit=W,Hinit=H,k=k,tolerance=1e-6,numIterations=numIterationsWH,beta=beta)
			
			D = A - np.matmul(W,H) #A-W*H
			#Calculate new error
			if (currentIteration>0):
				prevErr = currentErr
			
			errChange[currentIteration]=prevErr
			currentErr = ( np.linalg.norm(D-Z,'fro') )+ ( alpha*np.sum(np.sqrt(np.sum( np.square(Z),0 ))) ) + ( beta*np.linalg.norm(H,1) )
			#print out progress
			print("{} {} {}".format(currentIteration,prevErr,np.linalg.norm(D-Z,'fro')/np.linalg.norm(A,'fro')))
			currentIteration += 1
		
		errChange[currentIteration]=currentErr
		return ( Z,W,H,errChange )

	def transform(self, data_df):

		# vector_df = data_df[self.vector_colname]
		#to matrix
		# vector_matrix = vector_df.to_numpy()
		#columns as documents
		# vector_matrix = vector_matrix.T

		vector_matrix = data_df

		Z,W,H,errChange = self.textoutliers( A=vector_matrix, k=self.rank, alpha=self.a_reg, beta=self.b_reg )
		# outliers = Z
		# data_df[self.output_colname] = outliers
		return Z