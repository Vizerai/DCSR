import os, sys, string
import subprocess as sub

test_files = [	"kron_g500-logn18.mtx", "internet.mtx", "webbase-1M.mtx", "rail4284.mtx", "eu-2005.mtx",
				"dblp-2010.mtx", "enron.mtx", "cnr-2000.mtx", "flickr.mtx", "in-2004.mtx", "amazon-2008.mtx", 
				"wikipedia-20051105.mtx", "ljournal-2008.mtx", "hollywood-2009.mtx", "soc-LiveJournal1.mtx",
				"indochina-2004.mtx"]
#test_files = ["as-Skitter.mtx", "offshore.mtx", "pkustk14.mtx", "ASIC_320k.mtx"]

test_data = open("tests.txt", 'w')

for tf in test_files:
	print "\nTesting: " + tf + "\n"
	test_data.write("Testing: " + tf + "\n")

	mesh = "../../../../Data/Matrices/" + tf
	out = sub.Popen(['./CFA', mesh], stdout=sub.PIPE, stderr=sub.PIPE)
	output, error = out.communicate()
	print output
	print error
	test_data.write(output);
	test_data.write(error);
	test_data.write("\n\n\n");
