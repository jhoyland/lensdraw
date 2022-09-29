

class test:


	def __init__(self):


		self.functions =	{
			"a":self.A,
			"b":self.B,
			"c":self.C
		}


	def doit(self,f):

		self.functions[f]()


	def A(self):

		print("A")

	def B(self):

		print("B")

	def C(self):

		print("C")


atest = test()


atest.doit("b")

