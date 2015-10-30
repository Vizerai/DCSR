import os, sys, string

WarpSize = 32
SortSize = int(sys.argv[1])
if(len(sys.argv) == 3 and sys.argv[2] == "key"):
	key = 1;
else:
	key = 0;

print("INDEX_TYPE kvar;")
print("VALUE_TYPE dvar;")
print("int i,j;\n")

def BITONIC_INDEX_REVERSE(gs,tID):
	i = (tID/(gs>>1))*gs + (tID & ((gs>>1)-1))
	j = (tID/(gs>>1))*gs + (gs - (tID & ((gs>>1)-1)) - 1)
	if(i < 0 or i >= 128):
		print "i = ", i
	if(j < 0 or j >= 128):
		print "j = ", j

def BITONIC_INDEX(gs,tID):
	i = (tID/(gs>>1))*gs + (tID & ((gs>>1)-1))
	j = (gs>>1) + i
	if(i < 0 or i >= 128):
		print "i = ", i
	if(j < 0 or j >= 128):
		print "j = ", j

i = 2;
iter = SortSize / 2 / WarpSize

if key == 0:
	while i <= SortSize:
		j = i
		while j > 2:		
			if(iter > 1):
				if(j == i):
					print("BITONIC_INDEX_REVERSE(i,j," + str(j) + ",tID);")
					print("if(data[i] > data[j])")
					print("{	SWAP(data[i], data[j], var); }")
				else:
					print("BITONIC_INDEX(i,j," + str(j) + ",tID);")
					print("if(data[i] > data[j])")
					print("{	SWAP(data[i], data[j], var); }")
				for k in range(1,iter):
					if(j == i):
						print("BITONIC_INDEX_REVERSE(i,j," + str(j) + ",(tID+" + str(WarpSize*k) + "));")
						print("if(data[i] > data[j])")
						print("{	SWAP(data[i], data[j], var); }")
					else:
						print("BITONIC_INDEX(i,j," + str(j) + ",(tID+" + str(WarpSize*k) + "));")
						print("if(data[i] > data[j])")
						print("{	SWAP(data[i], data[j], var); }")
			else:
				if(j == i):
					print("BITONIC_INDEX_REVERSE(i,j," + str(j) + ",tID);")
					print("if(data[i] > data[j])")
					print("{	SWAP(data[i], data[j], var); }")
				else:
					print("BITONIC_INDEX(i,j," + str(j) + ",tID);")
					print("if(data[i] > data[j])")
					print("{	SWAP(data[i], data[j], var); }")
			j = j / 2

		
		print("BITONIC_INDEX(i,j,2,tID);")
		print("if(data[i] > data[j])")
		print("{	SWAP(data[i], data[j], var); }")
		if(iter > 1):
			for k in range(1,iter):
				print("BITONIC_INDEX(i,j,2,(tID+" + str(WarpSize*k) + "));")
				print("if(data[i] > data[j])")
				print("{	SWAP(data[i], data[j], var); }")

		print("")
		i = i * 2;
		
else:
	while i <= SortSize:
		j = i
		while j > 2:		
			if(iter > 1):
				if(j == i):
					print("BITONIC_INDEX_REVERSE(i,j," + str(j) + ",tID);")
				else:
					print("BITONIC_INDEX(i,j," + str(j) + ",tID);")
				print("if(key[i] > key[j])")
				print("{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }")
				for k in range(1,iter):
					if(j == i):
						print("BITONIC_INDEX_REVERSE(i,j," + str(j) + ",(tID+" + str(WarpSize*k) + "));")
					else:
						print("BITONIC_INDEX(i,j," + str(j) + ",(tID+" + str(WarpSize*k) + "));")
					print("if(key[i] > key[j])")
					print("{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }")
			else:
				if(j == i):
					print("BITONIC_INDEX_REVERSE(i,j," + str(j) + ",tID);")
				else:
					print("BITONIC_INDEX(i,j," + str(j) + ",tID);")
				print("if(key[i] > key[j])")
				print("{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }")
			j = j / 2

		
		print("BITONIC_INDEX(i,j,2,tID);")
		print("if(key[i] > key[j])")
		print("{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }")
		if(iter > 1):
			for k in range(1,iter):
				print("BITONIC_INDEX(i,j,2,(tID+" + str(WarpSize*k) + "));")
				print("if(key[i] > key[j])")
				print("{	SWAP(key[i], key[j], kvar);  SWAP(data[i], data[j], dvar); }")

		print("")
		i = i * 2;
